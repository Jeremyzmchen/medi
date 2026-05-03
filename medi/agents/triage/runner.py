"""
TriageGraphRunner

每轮 handle() 调用都执行一次完整的图 invoke。
checkpointer（MemorySaver）以 session_id 为 thread_id 跨轮保存状态，
IntakeNode 从 checkpoint 恢复 messages 和 collection_status，
每轮只问一个问题，直到信息充分后进入 ClinicalNode。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from medi.agents.triage.graph.builder import build_triage_graph
from medi.agents.triage.graph.state import (
    TriageGraphState,
    empty_symptom_data,
    empty_collection_status,
    empty_medical_record,
)
from medi.agents.triage.task_definitions import initial_task_progress, task_ids
from medi.agents.triage.urgency_evaluator import (
    evaluate_urgency_by_rules,
    UrgencyLevel,
    EMERGENCY_RESPONSE,
)
from medi.agents.triage.symptom_utils import format_severity_line, looks_like_temperature
from medi.agents.triage.department_router import DepartmentRouter
from medi.core.context import UnifiedContext, DialogueState
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.memory.episodic import EpisodicMemory


class TriageGraphRunner:

    def __init__(
        self,
        ctx: UnifiedContext,
        bus: AsyncStreamBus,
        router: DepartmentRouter | None = None,
    ) -> None:
        self._ctx = ctx
        self._bus = bus
        self._router = router or DepartmentRouter()
        self._episodic = EpisodicMemory(ctx.user_id)
        self._checkpointer = MemorySaver()
        self._is_first_turn = True                       # 第一轮需要传完整 initial_state
        # 提供给意图识别agent作为上下文辅助，但是主要意图识别还是依赖于 graph state 里的 collection_status 来判断
        self._cached_symptom_data: dict = {}             

    async def handle(self, user_input: str) -> None:
        # 1. 前置疾病安全检查报警
        urgency = evaluate_urgency_by_rules(user_input)
        if urgency and urgency.level == UrgencyLevel.EMERGENCY:
            self._ctx.transition(DialogueState.ESCALATING)
            await self._bus.emit(StreamEvent(
                type=EventType.ESCALATION,
                data={"reason": urgency.reason},
                session_id=self._ctx.session_id,
            ))
            await self._bus.emit(StreamEvent(
                type=EventType.RESULT,
                data={"content": EMERGENCY_RESPONSE},
                session_id=self._ctx.session_id,
            ))
            self._ctx.transition(DialogueState.INIT)
            self.reset_graph_state()
            return

        # 2. 记录用户对话
        self._ctx.add_user_message(user_input)

        # 3. 存储会话id用于checkpoint恢复
        config = {"configurable": {"thread_id": self._ctx.session_id}}

        # 4. 历史就诊记录prompt作为软参考，health_profile里字段才是硬约束
        history_prompt = await self._episodic.build_history_prompt()

        # 5. 构图
        graph = self._build_graph(history_prompt)

        if self._is_first_turn:
            # 第一轮：传入完整 initial_state，checkpointer 创建新 thread
            initial_state = TriageGraphState(
                session_id=self._ctx.session_id,
                user_id=self._ctx.user_id,
                messages=[{"role": "user", "content": user_input}],
                symptom_data=empty_symptom_data(),
                collection_status=empty_collection_status(),
                intake_protocol_id="generic_opqrst",
                intake_overlays=[],
                intake_facts=[],
                requested_slots=[],
                monitor_result={},
                controller_decision={},
                medical_record=empty_medical_record(),
                task_progress=initial_task_progress(),
                pending_tasks=task_ids(),
                current_task=None,
                task_rounds={},
                triage_done=False,
                intake_complete=False,
                department_candidates=[],
                urgency_level="normal",
                urgency_reason="",
                differential_diagnoses=[],
                risk_factors_summary="",
                clinical_missing_slots=[],
                next_node="intake",
                patient_output=None,
                doctor_hpi=None,
                graph_iteration=0,
                error=None,
            )
            self._is_first_turn = False
            # 用户首次会话信息
            input_data = initial_state
        else:
            # 后续轮：只传新消息，checkpointer 自动恢复其余状态
            # messages 用 operator.add reducer，此处传增量
            input_data = {"messages": [{"role": "user", "content": user_input}]}

        self._ctx.transition(DialogueState.TRIAGE_GRAPH_RUNNING)

        try:
            # 带入graphstate(config里的线程id去寻找会话id)，同时增量input_data(用户新消息)
            result = await graph.ainvoke(input_data, config=config)
            self._process_result(result)
        except Exception as e:
            await self._emit_error(str(e))
            return

    def reset_graph_state(self) -> None:
        self._checkpointer = MemorySaver()
        self._is_first_turn = True
        self._cached_symptom_data = {}

    def symptom_summary(self) -> str:
        data = self._cached_symptom_data
        if not data:
            return ""
        parts = []
        if data.get("region"):
            parts.append(f"部位：{data['region']}")
        if data.get("time_pattern"):
            parts.append(f"时间：{data['time_pattern']}")
        if data.get("onset"):
            parts.append(f"诱因：{data['onset']}")
        if data.get("quality"):
            parts.append(f"性质：{data['quality']}")
        if data.get("max_temperature"):
            parts.append(f"最高体温：{data['max_temperature']}")
        if data.get("frequency"):
            parts.append(f"频率/次数：{data['frequency']}")
        if data.get("severity"):
            severity = str(data["severity"])
            if not (data.get("max_temperature") and looks_like_temperature(severity)):
                parts.append(format_severity_line(severity))
        if data.get("accompanying"):
            parts.append(f"伴随：{', '.join(data['accompanying'])}")
        raw = data.get("raw_descriptions") or []
        if raw and not parts:
            parts.append(raw[-1])
        return "\n".join(parts) if parts else ""

    def _build_graph(self, history_prompt: str):
        graph = build_triage_graph(
            bus=self._bus,
            router=self._router,
            smart_chain=self._ctx.model_config.smart_chain,
            fast_chain=self._ctx.model_config.fast_chain,
            health_profile=self._ctx.health_profile,
            constraint_prompt=self._ctx.build_constraint_prompt(),
            history_prompt=history_prompt,
            episodic=self._episodic,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
        )
        graph.checkpointer = self._checkpointer
        return graph

    def _process_result(self, result: dict) -> None:
        if "symptom_data" in result:
            self._cached_symptom_data = result.get("symptom_data") or {}

        # OutputNode 跑完才是真正结束
        if result.get("patient_output") is not None:
            self._ctx.transition(DialogueState.INIT)
            self.reset_graph_state()
        elif result.get("next_node") == "intake_wait":
            # IntakeNode 发出追问，本轮图到 END，等待用户下一条输入
            self._ctx.transition(DialogueState.INTAKE_WAITING)
            # 把护士最后一条问题同步到 ctx.messages，
            # 让 Orchestrator 的意图分类器能看到完整对话上下文
            graph_messages = result.get("messages") or []
            last_assistant = next(
                (m["content"] for m in reversed(graph_messages)
                 if m.get("role") == "assistant"),
                None,
            )
            if last_assistant:
                self._ctx.add_assistant_message(last_assistant)

    async def _emit_error(self, message: str) -> None:
        await self._bus.emit(StreamEvent(
            type=EventType.ERROR,
            data={"message": f"分诊处理出错：{message}"},
            session_id=self._ctx.session_id,
        ))
