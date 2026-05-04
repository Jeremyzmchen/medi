"""
Hierarchical pre-consultation task tree.

This module lifts the flat intake slots into pre-consultation task groups:
T1 triage, T2 HPI collection, T3 past/safety history, and T4 chief
complaint generation. It is intentionally derived from the existing
ClinicalFactStore and ResolvedIntakePlan so the current intake flow can adopt the
tree without rewriting fact extraction or protocol matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medi.agents.triage.clinical_facts import ClinicalFactStore, slot_label
from medi.agents.triage.intake_protocols import ResolvedIntakePlan


TaskGroupId = Literal["T1", "T2", "T3", "T4"]
CompletionMode = Literal["all", "any"]


@dataclass(frozen=True)
class TaskSpec:
    id: str
    group_id: TaskGroupId
    label: str
    slots: tuple[str, ...]
    priority: int
    required: bool = True
    mode: CompletionMode = "all"


TASK_GROUP_LABELS: dict[TaskGroupId, str] = {
    "T1": "分诊与风险识别",
    "T2": "现病史采集",
    "T3": "既往史与安全信息",
    "T4": "主诉生成",
}


TRIAGE_RISK_KEYS = {
    "max_temperature",
    "associated_fever_symptoms",
    "exertional_related",
    "age",
    "mental_status",
    "intake_urination",
    "baseline_function",
    "fall_or_confusion",
    "pregnancy_weeks",
    "vaginal_bleeding_or_abdominal_pain",
    "immunosuppression_status",
    "infection_exposure",
    "dyspnea_sweating",
    "rest_or_exertion",
    "chest_pain_or_wheeze",
    "cyanosis_or_spo2",
    "vomiting_diarrhea",
    "stool_or_bleeding",
    "frequency",
    "dehydration_signs",
    "blood_or_black_stool",
    "trigger_exposure",
    "mucosal_or_swelling",
    "loss_of_consciousness",
    "neuro_deficits",
    "fever_neck_stiffness",
    "palpitations_chest_pain",
    "sudden_or_worst",
    "bleeding_deformity",
}


HPI_PATTERN_KEYS = {
    "max_temperature",
    "antipyretics",
    "associated_fever_symptoms",
    "exertional_related",
    "rest_or_exertion",
    "chest_pain_or_wheeze",
    "vomiting_diarrhea",
    "food_related",
    "injury_mechanism",
    "function_or_weight_bearing",
    "frequency",
    "stool_or_vomit_character",
    "dehydration_signs",
    "itch_or_pain",
    "trigger_exposure",
    "loss_of_consciousness",
}


def build_intake_task_tree(
    store: ClinicalFactStore,
    plan: ResolvedIntakePlan,
    *,
    relaxed_low_value: bool = False,
    clinical_missing_slots: list[str] | None = None,
) -> dict:
    """Build a JSON-serializable task tree from current intake state."""
    clinical_missing = set(clinical_missing_slots or [])
    specs = _task_specs_for_plan(plan, relaxed_low_value)
    nodes = [
        _task_node_from_spec(store, plan, spec, clinical_missing)
        for spec in specs
    ]

    groups: list[dict] = []
    for group_id, group_label in TASK_GROUP_LABELS.items():
        children = [node for node in nodes if node["group_id"] == group_id]
        if not children:
            continue
        required_children = [node for node in children if node["required"]]
        required_count = len(required_children)
        complete_count = sum(1 for node in required_children if node["status"] == "complete")
        completion = complete_count / required_count if required_count else 1.0
        missing_slots = _unique(
            slot
            for node in required_children
            for slot in node["missing_slots"]
        )
        groups.append({
            "id": group_id,
            "label": group_label,
            "status": _status_from_completion(completion, required_count),
            "completion": round(completion, 2),
            "missing_slots": missing_slots,
            "subtasks": children,
        })

    pending_slots = _unique(
        slot
        for node in sorted(nodes, key=lambda n: n["priority"])
        if node["required"] and node["status"] != "complete"
        for slot in node["missing_slots"]
    )
    required_nodes = [node for node in nodes if node["required"]]
    completion = (
        sum(1 for node in required_nodes if node["status"] == "complete") / len(required_nodes)
        if required_nodes
        else 1.0
    )

    return {
        "protocol_id": plan.protocol_id,
        "protocol_label": plan.protocol_label,
        "overlay_ids": plan.overlay_ids,
        "status": _status_from_completion(completion, len(required_nodes)),
        "completion": round(completion, 2),
        "pending_slots": pending_slots,
        "groups": groups,
    }


def slot_task_priority(
    slot: str,
    plan: ResolvedIntakePlan,
    *,
    relaxed_low_value: bool = False,
) -> int:
    """Return task-tree priority for a slot, falling back to a safe default."""
    if slot.startswith("specific.") and slot.removeprefix("specific.") in TRIAGE_RISK_KEYS:
        return 10
    best = 999
    for spec in _task_specs_for_plan(plan, relaxed_low_value):
        if slot in spec.slots:
            best = min(best, spec.priority)
    return best


def _task_specs_for_plan(
    plan: ResolvedIntakePlan,
    relaxed_low_value: bool,
) -> list[TaskSpec]:
    specs: list[TaskSpec] = [
        TaskSpec(
            id="T4.chief_complaint",
            group_id="T4",
            label="主诉记录",
            slots=("hpi.chief_complaint",),
            priority=5,
        ),
        TaskSpec(
            id="T1.triage_basis",
            group_id="T1",
            label="分诊基础信息",
            slots=("hpi.chief_complaint", "hpi.associated_symptoms"),
            priority=15,
            mode="all",
        ),
        TaskSpec(
            id="T2.onset",
            group_id="T2",
            label="起病时间",
            slots=("hpi.onset", "hpi.timing"),
            priority=20,
            mode="any",
        ),
        TaskSpec(
            id="T2.severity",
            group_id="T2",
            label="严重程度或量化指标",
            slots=("hpi.severity", "specific.max_temperature", "specific.frequency"),
            priority=20,
            mode="any",
        ),
        TaskSpec(
            id="T2.associated_symptoms",
            group_id="T2",
            label="伴随症状",
            slots=("hpi.associated_symptoms", "specific.associated_fever_symptoms"),
            priority=35,
            mode="any",
        ),
        TaskSpec(
            id="T3.current_medications",
            group_id="T3",
            label="当前用药",
            slots=("safety.current_medications",),
            priority=45,
        ),
        TaskSpec(
            id="T3.allergies",
            group_id="T3",
            label="过敏史",
            slots=("safety.allergies",),
            priority=45,
        ),
        TaskSpec(
            id="T3.relevant_history",
            group_id="T3",
            label="相关既往史",
            slots=("hpi.relevant_history",),
            priority=70,
            required=not relaxed_low_value,
        ),
    ]

    if "opqrst.location" in plan.required_fields:
        specs.append(TaskSpec(
            id="T2.location",
            group_id="T2",
            label="症状部位",
            slots=("hpi.location",),
            priority=35,
        ))
    if "opqrst.quality" in plan.required_fields:
        specs.append(TaskSpec(
            id="T2.character",
            group_id="T2",
            label="症状性质",
            slots=("hpi.character",),
            priority=35,
            required=not relaxed_low_value,
        ))
    if "opqrst.provocation" in plan.required_fields:
        specs.append(TaskSpec(
            id="T2.aggravating_alleviating",
            group_id="T2",
            label="加重或缓解因素",
            slots=("hpi.aggravating_alleviating",),
            priority=35,
        ))
    if "opqrst.radiation" in plan.required_fields:
        specs.append(TaskSpec(
            id="T2.radiation",
            group_id="T2",
            label="放射或扩散",
            slots=("hpi.radiation",),
            priority=35,
        ))

    for key, label in plan.pattern_required:
        slot = f"specific.{key}"
        if slot in _slots_already_covered(specs):
            continue
        group_id: TaskGroupId = "T1" if key in TRIAGE_RISK_KEYS else "T2"
        if key not in TRIAGE_RISK_KEYS and key not in HPI_PATTERN_KEYS:
            group_id = "T2"
        priority = 10 if group_id == "T1" else 30
        specs.append(TaskSpec(
            id=f"{group_id}.specific.{key}",
            group_id=group_id,
            label=label,
            slots=(slot,),
            priority=priority,
        ))

    return specs


def _task_node_from_spec(
    store: ClinicalFactStore,
    plan: ResolvedIntakePlan,
    spec: TaskSpec,
    clinical_missing: set[str],
) -> dict:
    answered_slots = [slot for slot in spec.slots if store.is_answered(slot)]
    missing_slots = [slot for slot in spec.slots if not store.is_answered(slot)]
    if spec.mode == "any" and answered_slots:
        missing_slots = []

    if spec.mode == "all":
        completion = len(answered_slots) / len(spec.slots) if spec.slots else 1.0
    else:
        completion = 1.0 if answered_slots else 0.0

    labels = {slot: slot_label(slot, plan) for slot in spec.slots}
    return {
        "id": spec.id,
        "group_id": spec.group_id,
        "label": spec.label,
        "status": _status_from_completion(completion, len(spec.slots)),
        "completion": round(completion, 2),
        "required": spec.required,
        "priority": spec.priority,
        "slots": list(spec.slots),
        "slot_labels": labels,
        "answered_slots": answered_slots,
        "missing_slots": missing_slots,
        "clinical_missing": [slot for slot in spec.slots if slot in clinical_missing],
    }


def _slots_already_covered(specs: list[TaskSpec]) -> set[str]:
    return {slot for spec in specs for slot in spec.slots}


def _status_from_completion(completion: float, required_count: int) -> str:
    if required_count == 0 or completion >= 1:
        return "complete"
    if completion > 0:
        return "partial"
    return "missing"


def _unique(items) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
