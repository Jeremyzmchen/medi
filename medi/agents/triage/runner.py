"""
TriageGraphRunner

首轮 handle() 从 intake 启动分诊图；需要追问时 inquirer_node 通过
LangGraph interrupt 暂停图，Runner 把 interrupt payload 转成 FOLLOW_UP
事件。用户下一条回答会通过 Command(resume=...) 回到暂停点，然后继续
沿 inquirer -> intake 抽取新事实。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from medi.agents.triage.graph.builder import build_triage_graph
from medi.agents.triage.graph.state import (
    TriageGraphState,
    empty_intake_plan,
    empty_preconsultation_record,
    empty_task_board,
    empty_clinical_assessment,
    empty_workflow_control,
)
from medi.agents.triage.urgency_evaluator import (
    evaluate_urgency_by_rules,
    UrgencyLevel,
    EMERGENCY_RESPONSE,
)
from medi.agents.triage.clinical_summary import build_symptom_summary_from_record
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
        self._pending_interrupt = False
        self._pending_interrupt_payload: dict | None = None
        # 提供给意图识别 agent 作为上下文辅助。
        self._cached_symptom_summary = ""

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

        # 4. 历史就诊记录prompt作为软参考，profile_snapshot 作为本次会话的只读硬约束
        history_prompt = await self._episodic.build_history_prompt()

        # 5. 构图
        graph = self._build_graph(history_prompt)

        if self._pending_interrupt:
            # The graph is paused inside inquirer_node; resume hands this user
            # answer back to the interrupt() call instead of starting a new run.
            input_data = Command(resume=user_input)
        elif self._is_first_turn:
            # 第一轮：传入完整 initial_state，checkpointer 创建新 thread
            initial_state = TriageGraphState(
                session_id=self._ctx.session_id,
                user_id=self._ctx.user_id,
                messages=[{"role": "user", "content": user_input}],
                intake_plan=empty_intake_plan(),
                clinical_facts=[],
                preconsultation_record=empty_preconsultation_record(),
                task_board=empty_task_board(),
                clinical_assessment=empty_clinical_assessment(),
                triage_output=None,
                workflow_control=empty_workflow_control(),
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
            if await self._handle_interrupt(result):
                return
            self._pending_interrupt = False
            self._pending_interrupt_payload = None
            self._process_result(result)
        except Exception as e:
            await self._emit_error(str(e))
            return

    def reset_graph_state(self) -> None:
        self._checkpointer = MemorySaver()
        self._is_first_turn = True
        self._pending_interrupt = False
        self._pending_interrupt_payload = None
        self._cached_symptom_summary = ""

    def symptom_summary(self) -> str:
        return self._cached_symptom_summary

    def _build_graph(self, history_prompt: str):
        graph = build_triage_graph(
            bus=self._bus,
            router=self._router,
            smart_chain=self._ctx.model_config.smart_chain,
            fast_chain=self._ctx.model_config.fast_chain,
            profile_snapshot=self._ctx.profile_snapshot,
            constraint_prompt=self._ctx.build_constraint_prompt(),
            history_prompt=history_prompt,
            episodic=self._episodic,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
        )
        graph.checkpointer = self._checkpointer
        return graph

    async def _handle_interrupt(self, result: dict) -> bool:
        interrupts = result.get("__interrupt__") or []
        if not interrupts:
            return False

        payload = getattr(interrupts[0], "value", interrupts[0])
        if not isinstance(payload, dict):
            payload = {"question": str(payload)}

        question = str(payload.get("question") or "").strip()
        task = payload.get("task")
        round_number = payload.get("round")

        self._pending_interrupt = True
        self._pending_interrupt_payload = payload
        self._update_cached_symptom_summary(result)
        self._ctx.transition(DialogueState.INTAKE_WAITING)

        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "inquirer", "round": round_number},
            session_id=self._ctx.session_id,
        ))
        await self._bus.emit(StreamEvent(
            type=EventType.FOLLOW_UP,
            data={
                "question": question,
                "round": round_number,
                "task": task,
            },
            session_id=self._ctx.session_id,
        ))
        if question:
            self._ctx.add_assistant_message(question)
        return True

    def _update_cached_symptom_summary(self, result: dict) -> None:
        if "preconsultation_record" in result or "clinical_facts" in result:
            self._cached_symptom_summary = build_symptom_summary_from_record(
                result.get("preconsultation_record"),
                result.get("clinical_facts"),
                result.get("messages") or [],
            )

    def _process_result(self, result: dict) -> None:
        self._update_cached_symptom_summary(result)

        # OutputNode 跑完才是真正结束
        if result.get("triage_output") is not None:
            self._ctx.transition(DialogueState.INIT)
            self.reset_graph_state()

    async def _emit_error(self, message: str) -> None:
        await self._bus.emit(StreamEvent(
            type=EventType.ERROR,
            data={"message": f"分诊处理出错：{message}"},
            session_id=self._ctx.session_id,
        ))

