"""
TriageGraphRunner

首轮 handle() 从 intake 启动分诊图；需要追问时 inquirer_node 通过
LangGraph interrupt 暂停图，Runner 把 interrupt payload 转成 FOLLOW_UP
事件。用户下一条回答会通过 Command(resume=...) 回到暂停点，然后继续
沿 inquirer -> intake 抽取新事实。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

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
from medi.agents.triage.clinical_summary import build_symptom_summary_from_record
from medi.agents.triage.department_router import DepartmentRouter
from medi.core.context import UnifiedContext, DialogueState
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.memory.episodic import EpisodicMemory


class GraphThreadStatus(str, Enum):
    EMPTY = "empty"
    INTERRUPTED = "interrupted"
    ACTIVE = "active"
    COMPLETED = "completed"


class TriageGraphRunner:

    def __init__(
        self,
        ctx: UnifiedContext,
        bus: AsyncStreamBus,
        router: DepartmentRouter | None = None,
        checkpointer=None,
    ) -> None:
        self._ctx = ctx
        self._bus = bus
        self._router = router or DepartmentRouter()
        self._episodic = EpisodicMemory(ctx.user_id)
        self._checkpointer = checkpointer or MemorySaver()
        self._is_first_turn = True                       # 第一轮需要传完整 initial_state
        self._pending_interrupt = False
        self._pending_interrupt_payload: dict | None = None
        # 提供给意图识别 agent 作为上下文辅助。
        self._cached_symptom_summary = ""

    async def handle(self, user_input: str) -> None:
        # 1. 记录用户对话
        self._ctx.add_user_message(user_input)

        # 2. 存储会话id用于checkpoint恢复
        config = {"configurable": {"thread_id": self._ctx.session_id}}

        # 3. 历史就诊记录prompt作为软参考，profile_snapshot 作为本次会话的只读硬约束
        history_prompt = await self._episodic.build_history_prompt()

        # 4. 构图
        graph = self._build_graph(history_prompt)
        status = await self.inspect_thread_state(graph=graph, config=config)
        if status == GraphThreadStatus.COMPLETED:
            await self._delete_thread_checkpoint()
            status = GraphThreadStatus.EMPTY

        if self._pending_interrupt or status == GraphThreadStatus.INTERRUPTED:
            # The graph is paused inside inquirer_node; resume hands this user
            # answer back to the interrupt() call instead of starting a new run.
            input_data = Command(resume=user_input)
            self._is_first_turn = False
        elif status == GraphThreadStatus.ACTIVE:
            input_data = {"messages": [{"role": "user", "content": user_input}]}
            self._is_first_turn = False
        elif self._is_first_turn or status in {GraphThreadStatus.EMPTY, GraphThreadStatus.COMPLETED}:
            # 第一轮：传入完整 initial_state，checkpointer 创建新 thread
            initial_state = self._initial_state(user_input)
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
            await self._process_result(result)
        except Exception as e:
            await self._emit_error(str(e))
            return

    async def reset_graph_state(self, *, clear_checkpoint: bool = True) -> None:
        if clear_checkpoint:
            await self._delete_thread_checkpoint()
        self._is_first_turn = True
        self._pending_interrupt = False
        self._pending_interrupt_payload = None
        self._cached_symptom_summary = ""

    def symptom_summary(self) -> str:
        return self._cached_symptom_summary

    def _build_graph(self, history_prompt: str):
        return build_triage_graph(
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
            checkpointer=self._checkpointer,
        )

    async def inspect_thread_state(self, graph=None, config: dict | None = None) -> GraphThreadStatus:
        graph = graph or self._build_graph(history_prompt="")
        config = config or self._thread_config()
        snapshot = await graph.aget_state(config)
        status = self._status_from_snapshot(snapshot)
        values = getattr(snapshot, "values", {}) or {}
        if values:
            self._update_cached_symptom_summary(values)
        if status == GraphThreadStatus.INTERRUPTED:
            self._pending_interrupt = True
            self._pending_interrupt_payload = self._interrupt_payload(snapshot)
        elif status != GraphThreadStatus.INTERRUPTED:
            self._pending_interrupt = False
            self._pending_interrupt_payload = None
        return status

    async def resume_context(self) -> dict:
        graph = self._build_graph(history_prompt="")
        snapshot = await graph.aget_state(self._thread_config())
        status = self._status_from_snapshot(snapshot)
        values = getattr(snapshot, "values", {}) or {}
        payload = self._interrupt_payload(snapshot)
        messages = [
            dict(message)
            for message in (values.get("messages") or [])
            if isinstance(message, dict)
        ]
        summary = build_symptom_summary_from_record(
            values.get("preconsultation_record"),
            values.get("clinical_facts"),
            messages,
        ) if values else ""
        if values:
            self._update_cached_symptom_summary(values)

        task_id = payload.get("task") if payload else None
        return {
            "status": status.value,
            "session_id": self._ctx.session_id,
            "pending_question": payload.get("question") if payload else None,
            "pending_task": task_id,
            "why_needed": self._why_task_needed(values, task_id),
            "collected_summary": summary,
            "missing_summary": self._missing_summary(values),
            "recent_messages": messages[-6:],
            "last_updated_at": getattr(snapshot, "created_at", None),
            "expires_at": None,
            "recommended_action": self._recommended_action(status),
            "actions": self._resume_actions(status),
        }

    def _initial_state(self, user_input: str) -> TriageGraphState:
        return TriageGraphState(
            session_id=self._ctx.session_id,
            user_id=self._ctx.user_id,
            messages=[{"role": "user", "content": user_input}],
            safety_gate={},
            intake_plan=empty_intake_plan(),
            clinical_facts=[],
            preconsultation_record=empty_preconsultation_record(),
            task_board=empty_task_board(),
            clinical_assessment=empty_clinical_assessment(),
            triage_output=None,
            workflow_control=empty_workflow_control(),
        )

    def _thread_config(self) -> dict:
        return {"configurable": {"thread_id": self._ctx.session_id}}

    def _status_from_snapshot(self, snapshot) -> GraphThreadStatus:
        values = getattr(snapshot, "values", {}) or {}
        if not values:
            return GraphThreadStatus.EMPTY
        if getattr(snapshot, "interrupts", ()):
            return GraphThreadStatus.INTERRUPTED
        if values.get("triage_output") is not None:
            return GraphThreadStatus.COMPLETED
        return GraphThreadStatus.ACTIVE

    def _interrupt_payload(self, snapshot) -> dict:
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if not interrupts:
            return {}
        payload = getattr(interrupts[0], "value", interrupts[0])
        if isinstance(payload, dict):
            return payload
        return {"question": str(payload)}

    async def _delete_thread_checkpoint(self) -> None:
        delete = getattr(self._checkpointer, "adelete_thread", None)
        if delete is not None:
            await delete(self._ctx.session_id)
            return
        sync_delete = getattr(self._checkpointer, "delete_thread", None)
        if sync_delete is not None:
            sync_delete(self._ctx.session_id)

    def _why_task_needed(self, values: dict, task_id: str | None) -> str | None:
        if not task_id:
            return None
        task_board = values.get("task_board") or {}
        progress = task_board.get("progress") or {}
        item = progress.get(task_id) or {}
        details = [
            detail.get("description") or detail.get("id")
            for detail in item.get("requirement_details") or []
            if not detail.get("completed")
        ]
        if details:
            return "；".join(str(detail) for detail in details if detail)
        description = item.get("description")
        return str(description) if description else None

    def _missing_summary(self, values: dict) -> list[str]:
        if not values:
            return []
        task_board = values.get("task_board") or {}
        monitor = task_board.get("monitor") or {}
        clinical = values.get("clinical_assessment") or {}
        items: list[str] = []
        items.extend(str(item) for item in monitor.get("high_value_missing_slots") or [])
        items.extend(str(item) for item in clinical.get("missing_slots") or [])
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _recommended_action(self, status: GraphThreadStatus) -> str:
        if status == GraphThreadStatus.INTERRUPTED:
            return "answer"
        if status == GraphThreadStatus.ACTIVE:
            return "continue"
        if status == GraphThreadStatus.COMPLETED:
            return "restart"
        return "start"

    def _resume_actions(self, status: GraphThreadStatus) -> list[str]:
        if status in {GraphThreadStatus.INTERRUPTED, GraphThreadStatus.ACTIVE}:
            return ["continue", "details", "restart", "close"]
        if status == GraphThreadStatus.COMPLETED:
            return ["restart", "close"]
        return ["start"]

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

    def _update_cached_symptom_summary(self, result: dict[str, Any]) -> None:
        if "preconsultation_record" in result or "clinical_facts" in result:
            self._cached_symptom_summary = build_symptom_summary_from_record(
                result.get("preconsultation_record"),
                result.get("clinical_facts"),
                result.get("messages") or [],
            )

    async def _process_result(self, result: dict) -> None:
        self._update_cached_symptom_summary(result)

        # OutputNode 跑完才是真正结束
        if result.get("triage_output") is not None:
            self._ctx.transition(DialogueState.INIT)
            await self.reset_graph_state()

    async def _emit_error(self, message: str) -> None:
        await self._bus.emit(StreamEvent(
            type=EventType.ERROR,
            data={"message": f"分诊处理出错：{message}"},
            session_id=self._ctx.session_id,
        ))

