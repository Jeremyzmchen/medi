"""
Subtask completion scoring for task-driven pre-consultation orchestration.

Monitor evaluates each subtask against the evolving PreconsultationRecord.
Each score combines two dimensions in a lightweight deterministic form:
whether the required clinical content exists, and whether the stored value is
usable as pre-consultation evidence.
"""

from __future__ import annotations

from medi.agents.triage.clinical_facts import ANSWERED_STATUSES
from medi.agents.triage.task_definitions import (
    TASK_COMPLETION_THRESHOLD,
    TASK_SPECS,
    PreconsultationTaskSpec,
    TaskRequirement,
)


def evaluate_task_progress(
    *,
    preconsultation_record: dict | None,
) -> tuple[dict[str, dict], list[str]]:
    record = preconsultation_record or {}
    progress = {
        spec.id: _evaluate_task(spec, record)
        for spec in TASK_SPECS
    }
    pending_tasks = [
        task_id
        for task_id, item in progress.items()
        if item["score"] < TASK_COMPLETION_THRESHOLD
    ]
    return progress, pending_tasks


def _evaluate_task(
    spec: PreconsultationTaskSpec,
    preconsultation_record: dict,
) -> dict:
    completed = [
        requirement.id
        for requirement in spec.requirements
        if _requirement_completed(requirement, preconsultation_record)
    ]
    total = len(spec.requirements)
    score = len(completed) / total if total else 1.0
    missing = [
        requirement.id
        for requirement in spec.requirements
        if requirement.id not in completed
    ]
    status = _status(score)
    return {
        "task_id": spec.id,
        "group_id": spec.group_id,
        "label": spec.label,
        "description": spec.description,
        "base_priority": spec.base_priority,
        "critical": spec.critical,
        "score": round(score, 2),
        "status": status,
        "completed_requirements": completed,
        "missing_requirements": missing,
        "requirement_details": [
            _requirement_detail(requirement, preconsultation_record)
            for requirement in spec.requirements
        ],
        "reason": _reason(status, missing),
    }


def _requirement_completed(
    requirement: TaskRequirement,
    preconsultation_record: dict,
) -> bool:
    values = [_record_value(path, preconsultation_record) for path in requirement.record_paths]
    completed = [_record_value_is_completed(value) for value in values]
    if requirement.completion_mode == "all":
        return all(completed) if completed else False
    return any(completed)


def _requirement_detail(
    requirement: TaskRequirement,
    preconsultation_record: dict,
) -> dict:
    completed_paths = [
        path for path in requirement.record_paths
        if _record_value_is_completed(_record_value(path, preconsultation_record))
    ]
    return {
        "id": requirement.id,
        "description": requirement.description,
        "record_paths": list(requirement.record_paths),
        "completed": bool(completed_paths),
        "completed_paths": completed_paths,
    }


def _record_value(path: str, preconsultation_record: dict):
    current = preconsultation_record
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _record_value_is_completed(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        status = str(value.get("status") or "").strip()
        if status in ANSWERED_STATUSES:
            return True
        if "value" in value:
            return _record_value_is_completed(value.get("value"))
        return any(_record_value_is_completed(v) for v in value.values())
    if isinstance(value, str):
        cleaned = value.strip()
        return bool(cleaned) and cleaned.lower() not in {"unknown", "none", "null"}
    if isinstance(value, (list, tuple, set)):
        return any(_record_value_is_completed(item) for item in value)
    return bool(value)


def _status(score: float) -> str:
    if score >= TASK_COMPLETION_THRESHOLD:
        return "complete"
    if score > 0:
        return "partial"
    return "pending"


def _reason(status: str, missing_requirements: list[str]) -> str:
    if status == "complete":
        return "任务已达到完成阈值"
    if missing_requirements:
        return "缺少：" + "、".join(missing_requirements)
    return "任务尚未采集到有效病历信息"
