"""
IntakeInquirerNode - deliver the generated follow-up question.

The Inquirer owns the user-facing side effect: emitting FOLLOW_UP, appending
the nurse message, and recording that the selected task was attempted.
"""

from __future__ import annotations

from medi.agents.triage.graph.state import TriageGraphState
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


async def intake_inquirer_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
) -> dict:
    session_id = state["session_id"]
    messages = state.get("messages") or []
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    decision = state.get("controller_decision") or {}

    question = (
        decision.get("next_best_question")
        or "为了让医生更快了解情况，请再补充一个最重要的信息。"
    )
    task_id = decision.get("next_best_task") or state.get("current_task")

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "inquirer", "round": assistant_count + 1},
        session_id=session_id,
    ))
    await bus.emit(StreamEvent(
        type=EventType.FOLLOW_UP,
        data={
            "question": question,
            "round": assistant_count + 1,
            "task": task_id,
        },
        session_id=session_id,
    ))

    task_rounds = dict(state.get("task_rounds") or {})
    if task_id:
        task_rounds[task_id] = task_rounds.get(task_id, 0) + 1

    result = {
        "messages": [{"role": "assistant", "content": question}],
        "task_rounds": task_rounds,
        "intake_complete": False,
        "next_node": "intake_wait",
    }
    return result
