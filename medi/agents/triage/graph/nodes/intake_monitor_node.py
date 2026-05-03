"""
IntakeMonitorNode — 预诊档案质量评估节点。

对应 Monitor 角色：
  只负责评估当前信息完整度，输出 0-100 分和缺失槽位列表。
  不做任何调度决策，不决定问什么、问不问——那是 Controller 的职责。

输入：state（intake_facts, intake_protocol_id, requested_slots, ...）
输出：写入 state["monitor_result"]（MonitorResult TypedDict）
"""

from __future__ import annotations

import sys

from medi.agents.triage.graph.state import (
    MAX_INTAKE_ROUNDS,
    MonitorResult,
    TriageGraphState,
)
from medi.agents.triage.intake_facts import (
    FactStore,
    collection_status_from_facts,
    required_slots_for_plan,
)
from medi.agents.triage.intake_protocols import (
    ResolvedIntakePlan,
    resolve_intake_plan,
)
from medi.agents.triage.task_progress import evaluate_task_progress
from medi.agents.triage.task_tree import build_intake_task_tree
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


HIGH_RISK_PROTOCOLS = {
    "chest_pain",
    "dyspnea",
    "abdominal_pain",
    "headache",
    "trauma",
}

# 每个协议的危险信号必查槽
CRITICAL_PATTERN_KEYS: dict[str, tuple[str, ...]] = {
    "fever": ("max_temperature", "associated_fever_symptoms"),
    "chest_pain": ("dyspnea_sweating", "exertional_related", "cardiovascular_history"),
    "dyspnea": ("rest_or_exertion", "chest_pain_or_wheeze", "cyanosis_or_spo2"),
    "abdominal_pain": ("vomiting_diarrhea", "stool_or_bleeding"),
    "headache": ("sudden_or_worst", "neuro_deficits", "fever_neck_stiffness"),
    "trauma": ("injury_mechanism", "function_or_weight_bearing", "bleeding_deformity"),
    "diarrhea_vomiting": ("frequency", "dehydration_signs", "blood_or_black_stool"),
    "rash_allergy": ("trigger_exposure", "mucosal_or_swelling"),
    "dizziness_syncope": ("loss_of_consciousness", "neuro_deficits", "palpitations_chest_pain"),
}

CRITICAL_OVERLAY_PATTERN_KEYS: dict[str, tuple[str, ...]] = {
    "pediatric": ("age", "mental_status", "intake_urination"),
    "elderly": ("baseline_function", "fall_or_confusion"),
    "pregnancy": ("pregnancy_weeks", "vaginal_bleeding_or_abdominal_pain"),
    "immunocompromised": ("immunosuppression_status", "infection_exposure"),
}


async def intake_monitor_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    health_profile=None,
) -> dict:
    """
    评估预诊档案当前质量，写入 monitor_result。
    不发出追问，不决定下一步——由 ControllerNode 读取结果后决策。
    """
    session_id = state["session_id"]
    messages = state.get("messages") or []
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

    fixed_protocol_id = _locked_protocol_id(state)
    plan = resolve_intake_plan(messages, health_profile, fixed_protocol_id=fixed_protocol_id)
    store = FactStore.from_state(state.get("intake_facts"))

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "monitor", "round": assistant_count + 1},
        session_id=session_id,
    ))

    relaxed_low_value = assistant_count >= 5
    clinical_missing = [
        slot for slot in (state.get("clinical_missing_slots") or [])
        if not store.is_answered(slot)
    ]

    score, missing, red_flags_checked, safety_covered = _score(
        store, plan, relaxed_low_value, clinical_missing
    )
    task_tree = build_intake_task_tree(
        store=store,
        plan=plan,
        relaxed_low_value=relaxed_low_value,
        clinical_missing_slots=clinical_missing,
    )
    task_progress, pending_tasks = evaluate_task_progress(
        medical_record=state.get("medical_record"),
    )
    required_covered = _required_slots_covered(store, plan, relaxed_low_value)
    doctor_ready = _has_core_doctor_summary(store, plan, relaxed_low_value)
    threshold = 80 if plan.protocol_id in HIGH_RISK_PROTOCOLS else 74
    doctor_summary_ready = score >= threshold and red_flags_checked and doctor_ready

    result = MonitorResult(
        score=score,
        red_flags_checked=red_flags_checked,
        safety_slots_covered=safety_covered,
        doctor_summary_ready=doctor_summary_ready,
        required_slots_covered=required_covered,
        high_value_missing_slots=_unique(missing),
        reason=_score_reason(score, red_flags_checked, safety_covered, plan),
    )

    collection_status = collection_status_from_facts(
        store=store,
        plan=plan,
        complete=False,
        reason="等待 Controller 调度决策",
    )

    print(
        f"[monitor] score={score} red_flags={red_flags_checked} "
        f"protocol={plan.protocol_id} missing={result['high_value_missing_slots']}",
        file=sys.stderr,
    )

    return {
        "monitor_result": result,
        "task_tree": task_tree,
        "task_progress": task_progress,
        "pending_tasks": pending_tasks,
        "triage_done": _triage_done(task_progress),
        "collection_status": collection_status,
        "next_node": "controller",
    }


# ─────────────────────────────────────────────
# 评分逻辑
# ─────────────────────────────────────────────

def _score(
    store: FactStore,
    plan: ResolvedIntakePlan,
    relaxed_low_value: bool,
    clinical_missing: list[str],
) -> tuple[int, list[str], bool, bool]:
    """
    返回 (score, missing_slots, red_flags_checked, safety_covered)。
    评分项与权重对应信息完整度维度。
    """
    missing: list[str] = list(clinical_missing)
    score = 0

    # 主诉 +12
    if store.is_answered("hpi.chief_complaint"):
        score += 12
    else:
        missing.append("hpi.chief_complaint")

    # 起病时间 +14（onset 或 timing 任一即可）
    if store.is_answered("hpi.onset") or store.is_answered("hpi.timing"):
        score += 14
    else:
        missing.append("hpi.onset")

    # 严重程度 / 量化指标 +12（发热用体温，腹泻用次数，其他用评分）
    if _has_severity_or_quantifier(store, plan):
        score += 12
    elif _severity_required(plan, relaxed_low_value):
        missing.append(_severity_slot_for_plan(plan))
    else:
        score += 6   # 宽松模式下无量化指标也给部分分

    # 部位 +8（仅协议要求时计入）
    if "opqrst.location" in plan.required_fields:
        if store.is_answered("hpi.location"):
            score += 8
        else:
            missing.append("hpi.location")

    # 症状性质 +6
    if "opqrst.quality" in plan.required_fields:
        if store.is_answered("hpi.character"):
            score += 6
        elif not relaxed_low_value:
            missing.append("hpi.character")
        else:
            score += 3

    # 伴随症状 +12
    if _has_associated_context(store, plan):
        score += 12
    else:
        missing.append(_associated_slot_for_plan(plan))

    # 症状特异性 pattern 槽 0-20（按完成比例）
    pattern_slots = [f"specific.{key}" for key, _ in plan.pattern_required]
    answered_pattern = [s for s in pattern_slots if store.is_answered(s)]
    if pattern_slots:
        score += round(20 * len(answered_pattern) / len(pattern_slots))
    else:
        score += 8

    # 危险信号槽：必须全部采集才算 red_flags_checked
    critical_slots = _critical_pattern_slots(plan)
    red_flag_slots = critical_slots or pattern_slots
    red_flags_checked = all(store.is_answered(s) for s in red_flag_slots)
    for s in red_flag_slots:
        if not store.is_answered(s):
            missing.append(s)

    # 安全信息：用药 +7，过敏 +8
    safety_covered = (
        store.is_answered("safety.current_medications")
        and store.is_answered("safety.allergies")
    )
    if store.is_answered("safety.current_medications"):
        score += 7
    else:
        missing.append("safety.current_medications")
    if store.is_answered("safety.allergies"):
        score += 8
    else:
        missing.append("safety.allergies")

    # 既往史 +8（前 4 轮强制要求，之后宽松）
    assistant_count_approx = len([s for s in missing if s])  # 仅用于判断
    if store.is_answered("hpi.relevant_history"):
        score += 8

    score = min(score, 100)
    return score, missing, red_flags_checked, safety_covered


def _has_severity_or_quantifier(store: FactStore, plan: ResolvedIntakePlan) -> bool:
    if store.is_answered("hpi.severity"):
        return True
    for key in ("max_temperature", "frequency"):
        if store.is_answered(f"specific.{key}"):
            return True
    return False


def _severity_slot_for_plan(plan: ResolvedIntakePlan) -> str:
    keys = {key for key, _ in plan.pattern_required}
    if "max_temperature" in keys:
        return "specific.max_temperature"
    if "frequency" in keys:
        return "specific.frequency"
    return "hpi.severity"


def _has_associated_context(store: FactStore, plan: ResolvedIntakePlan) -> bool:
    if store.is_answered("hpi.associated_symptoms"):
        return True
    for key, _ in plan.pattern_required:
        if "associated" in key and store.is_answered(f"specific.{key}"):
            return True
    return False


def _associated_slot_for_plan(plan: ResolvedIntakePlan) -> str:
    for key, _ in plan.pattern_required:
        if "associated" in key:
            return f"specific.{key}"
    return "hpi.associated_symptoms"


def _critical_pattern_slots(plan: ResolvedIntakePlan) -> list[str]:
    available = {key for key, _ in plan.pattern_required}
    keys: list[str] = []
    keys.extend(CRITICAL_PATTERN_KEYS.get(plan.protocol_id, ()))
    for overlay in plan.overlays:
        keys.extend(CRITICAL_OVERLAY_PATTERN_KEYS.get(overlay.id, ()))
    return [f"specific.{key}" for key in _unique(keys) if key in available]


def _severity_required(plan: ResolvedIntakePlan, relaxed_low_value: bool) -> bool:
    if not relaxed_low_value:
        return True
    keys = {key for key, _ in plan.pattern_required}
    if keys & {"max_temperature", "frequency"}:
        return True
    return plan.protocol_id in HIGH_RISK_PROTOCOLS


def _required_slots_covered(
    store: FactStore,
    plan: ResolvedIntakePlan,
    relaxed_low_value: bool,
) -> bool:
    for slot in required_slots_for_plan(plan):
        if slot == "hpi.relevant_history":
            continue
        if slot == "hpi.severity" and _has_severity_or_quantifier(store, plan):
            continue
        if slot in {"hpi.severity", "hpi.character"} and relaxed_low_value:
            continue
        if not store.is_answered(slot):
            return False
    return True


def _has_core_doctor_summary(
    store: FactStore,
    plan: ResolvedIntakePlan,
    relaxed_low_value: bool,
) -> bool:
    return (
        store.is_answered("hpi.chief_complaint")
        and (store.is_answered("hpi.onset") or store.is_answered("hpi.timing"))
        and (_has_severity_or_quantifier(store, plan) or not _severity_required(plan, relaxed_low_value))
        and _has_associated_context(store, plan)
    )


def _score_reason(
    score: int,
    red_flags_checked: bool,
    safety_covered: bool,
    plan: ResolvedIntakePlan,
) -> str:
    parts = [f"当前得分 {score}/100"]
    if not red_flags_checked:
        parts.append(f"{plan.protocol_label}危险信号槽尚未全部采集")
    if not safety_covered:
        parts.append("用药/过敏尚未采集")
    return "；".join(parts)


def _locked_protocol_id(state: TriageGraphState) -> str | None:
    protocol_id = state.get("intake_protocol_id")
    if protocol_id and protocol_id != "generic_opqrst":
        return protocol_id
    return None


def _triage_done(task_progress: dict[str, dict]) -> bool:
    return all(
        task_progress.get(task_id, {}).get("status") == "complete"
        for task_id in ("T1_PRIMARY_DEPARTMENT", "T1_SECONDARY_DEPARTMENT")
    )


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
