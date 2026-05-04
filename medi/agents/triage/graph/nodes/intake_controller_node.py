"""
IntakeControllerNode - global task scheduler.

Controller reads TaskBoard task completion scores and selects the next
subtask. It does not choose extraction fields, generate questions, or emit
patient-facing events.
"""

from __future__ import annotations

import sys

from langgraph.types import Command

from medi.agents.triage.graph.state import (
    ControllerDecision,
    MAX_INTAKE_ROUNDS,
    TriageGraphState,
)
from medi.agents.triage.task_definitions import (
    TASK_BY_ID,
    TASK_COMPLETION_THRESHOLD,
)
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


PREFERRED_MAX_QUESTIONS = 8
TASK_REPEAT_PENALTY = 18
CRITICAL_TASK_BONUS = 30


async def intake_controller_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    profile_snapshot=None,
    max_rounds: int = MAX_INTAKE_ROUNDS,
) -> Command:
    """Select the next subtask and route to Prompter or Clinical."""
    session_id = state["session_id"]
    messages = state.get("messages") or []
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

    task_board = dict(state.get("task_board") or {})
    monitor = task_board.get("monitor") or {}
    task_progress = task_board.get("progress") or {}
    pending_tasks = task_board.get("pending_tasks") or []
    task_rounds = dict(task_board.get("task_rounds") or {})

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "controller", "round": assistant_count + 1},
        session_id=session_id,
    ))

    selected_task, task_priority_score = _select_next_task(
        pending_tasks=pending_tasks,
        task_progress=task_progress,
        task_rounds=task_rounds,
    )
    task_instruction = _task_instruction(selected_task, task_progress)

    preferred_limit = min(max_rounds, PREFERRED_MAX_QUESTIONS)
    can_finish, finish_reason = _decide_finish(
        selected_task=selected_task,
        pending_tasks=pending_tasks,
        task_progress=task_progress,
        assistant_count=assistant_count,
        max_rounds=max_rounds,
        preferred_limit=preferred_limit,
        monitor_score=monitor.get("score", 0),
        red_flags_checked=bool(monitor.get("red_flags_checked")),
        safety_slots_covered=bool(monitor.get("safety_slots_covered")),
        doctor_summary_ready=bool(monitor.get("doctor_summary_ready")),
    )

    if can_finish:
        selected_task = None
        task_priority_score = None
        task_instruction = None

    decision = ControllerDecision(
        can_finish_intake=can_finish,
        next_best_task=selected_task,
        task_priority_score=task_priority_score,
        task_instruction=task_instruction,
        next_best_question=None,
        reason=finish_reason,
    )

    print(
        f"[controller] can_finish={can_finish} task={selected_task} "
        f"score={monitor.get('score', 0)} round={assistant_count + 1}",
        file=sys.stderr,
    )

    if can_finish:
        task_board.update({
            "controller": decision,
            "current_task": None,
        })
        return Command(
            update={
                "task_board": task_board,
                "workflow_control": {
                    "next_node": "clinical",
                    "intake_complete": True,
                    "graph_iteration": (state.get("workflow_control") or {}).get("graph_iteration", 0),
                },
            },
            goto="clinical",
        )

    task_board.update({
        "controller": decision,
        "current_task": selected_task,
    })
    return Command(
        update={
            "task_board": task_board,
            "workflow_control": {
                "next_node": "prompter",
                "intake_complete": False,
                "graph_iteration": (state.get("workflow_control") or {}).get("graph_iteration", 0),
            },
        },
        goto="prompter",
    )


def _decide_finish(
    *,
    selected_task: str | None,
    pending_tasks: list[str],
    task_progress: dict[str, dict],
    assistant_count: int,
    max_rounds: int,
    preferred_limit: int,
    monitor_score: int,
    red_flags_checked: bool = False,
    safety_slots_covered: bool = False,
    doctor_summary_ready: bool = False,
) -> tuple[bool, str]:
    if assistant_count >= max_rounds:
        return True, "达到最大追问轮数，未完成任务将交给医生侧总结"
    if not _blocking_pending_tasks(pending_tasks, task_progress):
        return True, "预问诊任务已达到完成阈值"
    if not selected_task:
        return True, "没有可继续调度的未完成任务"
    if (
        assistant_count >= preferred_limit
        and monitor_score >= 65
        and red_flags_checked
        and safety_slots_covered
        and doctor_summary_ready
        and not _critical_pending_tasks(pending_tasks, task_progress)
    ):
        return True, "核心预问诊信息已具备价值，结束继续追问"

    task_label = _task_label(selected_task)
    return False, f"继续推进任务：{task_label}"


def _blocking_pending_tasks(
    pending_tasks: list[str],
    task_progress: dict[str, dict],
) -> list[str]:
    blocking: list[str] = []
    for task_id in pending_tasks:
        if task_id == "T4_CHIEF_COMPLAINT_GENERATION":
            continue
        item = task_progress.get(task_id) or {}
        if _safe_float(item.get("score"), default=0.0) < TASK_COMPLETION_THRESHOLD:
            blocking.append(task_id)
    return blocking


def _critical_pending_tasks(
    pending_tasks: list[str],
    task_progress: dict[str, dict],
) -> list[str]:
    critical: list[str] = []
    for task_id in pending_tasks:
        item = task_progress.get(task_id) or {}
        if item.get("critical") and _safe_float(item.get("score"), default=0.0) < TASK_COMPLETION_THRESHOLD:
            critical.append(task_id)
    return critical


def _select_next_task(
    *,
    pending_tasks: list[str],
    task_progress: dict[str, dict],
    task_rounds: dict[str, int],
) -> tuple[str | None, float | None]:
    candidates: list[tuple[float, int, str]] = []
    ordered_task_ids = pending_tasks or list(task_progress)
    for order, task_id in enumerate(ordered_task_ids):
        if task_id == "T4_CHIEF_COMPLAINT_GENERATION":
            continue
        item = task_progress.get(task_id)
        if not item or _task_is_complete(item):
            continue
        priority_score = _task_dispatch_score(item, task_rounds)
        candidates.append((priority_score, -order, task_id))

    if not candidates:
        return None, None
    priority_score, _, task_id = max(candidates)
    return task_id, round(priority_score, 2)


def _task_dispatch_score(
    item: dict,
    task_rounds: dict[str, int],
) -> float:
    task_id = item.get("task_id") or ""
    score = _safe_float(item.get("score"), default=0.0)
    base_priority = int(item.get("base_priority") or 0)
    value = base_priority * max(0.0, 1.0 - score)
    if item.get("critical"):
        value += CRITICAL_TASK_BONUS
    value -= TASK_REPEAT_PENALTY * int(task_rounds.get(task_id, 0))
    return value


def _task_is_complete(item: dict) -> bool:
    return _safe_float(item.get("score"), default=0.0) >= TASK_COMPLETION_THRESHOLD


def _task_instruction(
    task_id: str | None,
    task_progress: dict[str, dict],
) -> str | None:
    if not task_id:
        return None
    spec = TASK_BY_ID.get(task_id)
    item = task_progress.get(task_id) or {}
    label = spec.label if spec else task_id
    description = spec.description if spec else item.get("description", "")
    details = item.get("requirement_details") or []
    missing = [
        detail.get("description") or detail.get("id")
        for detail in details
        if not detail.get("completed")
    ]

    parts = [f"推进任务：{label}。"]
    if description:
        parts.append(f"目标：{description}。")
    if missing:
        parts.append("当前缺口：" + "；".join(missing) + "。")
    return "".join(parts)


def _task_label(task_id: str | None) -> str:
    if not task_id:
        return "未完成任务"
    spec = TASK_BY_ID.get(task_id)
    return spec.label if spec else task_id


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
