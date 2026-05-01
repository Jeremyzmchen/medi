"""
IntakeReviewNode - 预诊质量门节点。

职责不是判断最终科室，而是判断当前信息是否足以形成对医生有价值的
预诊档案；不足时只补问一个最高价值问题。
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

from medi.agents.triage.graph.state import (
    IntakeReviewStatus,
    MAX_INTAKE_ROUNDS,
    TriageGraphState,
)
from medi.agents.triage.intake_facts import (
    BASE_SLOT_SPECS,
    FactStore,
    collection_status_from_facts,
    required_slots_for_plan,
    slot_label,
)
from medi.agents.triage.intake_protocols import (
    ResolvedIntakePlan,
    resolve_intake_plan,
)
from medi.agents.triage.task_tree import (
    build_intake_task_tree,
    slot_task_priority,
)
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


PREFERRED_MAX_REVIEW_QUESTIONS = 6
HIGH_RISK_PROTOCOLS = {
    "chest_pain",
    "dyspnea",
    "abdominal_pain",
    "headache",
    "trauma",
}

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

REVIEW_SLOT_QUESTIONS: dict[str, str] = {
    "specific.max_temperature": "最高体温到多少度？大概是什么时候测到的？",
    "specific.antipyretics": "有没有用过退烧药？用了以后体温能降下来吗？",
    "specific.associated_fever_symptoms": "除了发热，还有咳嗽、咽痛、皮疹、腹泻、尿痛或呼吸困难吗？",
    "specific.exertional_related": "胸部不适和活动、情绪激动或休息有关系吗？",
    "specific.dyspnea_sweating": "胸痛时有没有呼吸困难、出汗、恶心或明显濒死感？",
    "specific.cardiovascular_history": "您有高血压、冠心病、糖尿病，或长期吸烟等心血管风险吗？",
    "specific.rest_or_exertion": "呼吸困难是在静息时也有，还是主要活动后出现？",
    "specific.chest_pain_or_wheeze": "有没有同时胸痛、喘鸣、咳嗽或咳痰？",
    "specific.cyanosis_or_spo2": "有没有口唇发紫，或者测过血氧饱和度？",
    "specific.vomiting_diarrhea": "腹痛时有没有恶心、呕吐或腹泻？",
    "specific.stool_or_bleeding": "有没有黑便、血便或呕血？",
    "specific.food_related": "这次不适和进食、饮酒或油腻饮食有关系吗？",
    "specific.sudden_or_worst": "头痛是突然一下达到最重，或像一生中最严重的头痛吗？",
    "specific.neuro_deficits": "有没有肢体无力、麻木、口角歪斜、说话不清或看东西异常？",
    "specific.fever_neck_stiffness": "有没有发热、脖子僵硬，或喷射样呕吐？",
    "specific.injury_mechanism": "是怎么受伤的？比如摔倒高度、撞击速度或扭伤经过。",
    "specific.function_or_weight_bearing": "受伤部位现在还能活动、负重或正常使用吗？",
    "specific.bleeding_deformity": "有没有明显出血、畸形、开放伤口、麻木或无力？",
    "specific.frequency": "从开始到现在大概腹泻或呕吐了几次？",
    "specific.stool_or_vomit_character": "大便或呕吐物是什么样的，比如水样、黏液、黑色或带血吗？",
    "specific.dehydration_signs": "有没有明显口渴、尿少、头晕乏力，或者喝水也留不住？",
    "specific.blood_or_black_stool": "有没有血便、黑便，或呕吐物里带血？",
    "specific.age": "孩子多大了？是几个月还是几岁？",
    "specific.mental_status": "孩子精神反应怎么样，有没有嗜睡、烦躁、抽搐或明显没精神？",
    "specific.intake_urination": "孩子现在喝水、进食和尿量怎么样？尿量有没有明显减少？",
    "specific.baseline_function": "平时活动能力怎么样？这次和往常相比变化大吗？",
    "specific.fall_or_confusion": "有没有跌倒、意识混乱或反应变差？",
    "specific.pregnancy_weeks": "目前怀孕多少周，或产后多久？",
    "specific.vaginal_bleeding_or_abdominal_pain": "有没有阴道流血、腹痛或胎动明显异常？",
    "specific.immunosuppression_status": "免疫抑制或肿瘤治疗是什么情况？最近一次治疗是什么时候？",
    "specific.infection_exposure": "近期有没有感染接触、留置导管、伤口红肿或其他感染线索？",
}


@dataclass(frozen=True)
class IntakeQualityReview:
    doctor_value_score: int
    can_finish_intake: bool
    doctor_summary_ready: bool
    red_flags_checked: bool
    required_slots_covered: bool
    safety_slots_covered: bool
    high_value_missing_slots: list[str]
    next_best_slot: str | None
    next_best_question: str | None
    reason: str
    task_tree: dict

    def to_state(self) -> IntakeReviewStatus:
        return IntakeReviewStatus(
            doctor_value_score=self.doctor_value_score,
            can_finish_intake=self.can_finish_intake,
            doctor_summary_ready=self.doctor_summary_ready,
            red_flags_checked=self.red_flags_checked,
            required_slots_covered=self.required_slots_covered,
            safety_slots_covered=self.safety_slots_covered,
            high_value_missing_slots=self.high_value_missing_slots,
            next_best_slot=self.next_best_slot,
            next_best_question=self.next_best_question,
            reason=self.reason,
            task_tree=self.task_tree,
        )


async def intake_review_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    health_profile=None,
    max_rounds: int = MAX_INTAKE_ROUNDS,
) -> dict:
    """评估预诊档案价值；不足时发出下一问并结束本轮图。"""
    session_id = state["session_id"]
    messages = state.get("messages") or []
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    fixed_protocol_id = _locked_protocol_id(state)
    plan = resolve_intake_plan(
        messages,
        health_profile,
        fixed_protocol_id=fixed_protocol_id,
    )
    store = FactStore.from_state(state.get("intake_facts"))

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "intake_review", "round": assistant_count + 1},
        session_id=session_id,
    ))

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=state.get("requested_slots") or [],
        assistant_count=assistant_count,
        max_rounds=max_rounds,
        clinical_missing_slots=state.get("clinical_missing_slots") or [],
    )
    collection_status = collection_status_from_facts(
        store=store,
        plan=plan,
        complete=review.can_finish_intake,
        reason=review.reason,
    )

    print(
        f"[intake_review] score={review.doctor_value_score} "
        f"ready={review.can_finish_intake} protocol={plan.protocol_id} "
        f"next_slot={review.next_best_slot} missing={review.high_value_missing_slots}",
        file=sys.stderr,
    )

    if review.can_finish_intake:
        return {
            "collection_status": collection_status,
            "intake_review": review.to_state(),
            "task_tree": review.task_tree,
            "intake_complete": True,
            "next_node": "clinical",
        }

    next_question = review.next_best_question or "为了让医生更快了解情况，请再补充一个最重要的信息。"
    nurse_message = {"role": "assistant", "content": next_question}

    await bus.emit(StreamEvent(
        type=EventType.FOLLOW_UP,
        data={"question": next_question, "round": assistant_count + 1},
        session_id=session_id,
    ))

    result = {
        "messages": [nurse_message],
        "collection_status": collection_status,
        "intake_review": review.to_state(),
        "task_tree": review.task_tree,
        "intake_complete": False,
        "next_node": "intake_wait",
    }
    if review.next_best_slot:
        result["requested_slots"] = [review.next_best_slot]
    return result


def review_intake_quality(
    store: FactStore,
    plan: ResolvedIntakePlan,
    requested_slots: list[str],
    assistant_count: int,
    max_rounds: int = MAX_INTAKE_ROUNDS,
    clinical_missing_slots: list[str] | None = None,
) -> IntakeQualityReview:
    """确定性评估：当前事实是否足以生成有医生价值的预诊档案。"""
    missing: list[str] = []
    score = 0
    relaxed_low_value = assistant_count >= 5
    clinical_missing = [
        slot for slot in clinical_missing_slots or []
        if not store.is_answered(slot)
    ]
    missing.extend(clinical_missing)
    task_tree = build_intake_task_tree(
        store=store,
        plan=plan,
        relaxed_low_value=relaxed_low_value,
        clinical_missing_slots=clinical_missing,
    )

    if store.is_answered("hpi.chief_complaint"):
        score += 12
    else:
        missing.append("hpi.chief_complaint")

    if store.is_answered("hpi.onset") or store.is_answered("hpi.timing"):
        score += 14
    else:
        missing.append("hpi.onset")

    if _has_severity_or_quantifier(store, plan):
        score += 12
    elif _severity_required(plan, relaxed_low_value):
        missing.append(_severity_slot_for_plan(plan))
    else:
        score += 6

    if "opqrst.location" in plan.required_fields:
        if store.is_answered("hpi.location"):
            score += 8
        else:
            missing.append("hpi.location")

    if "opqrst.quality" in plan.required_fields:
        if store.is_answered("hpi.character"):
            score += 6
        elif not relaxed_low_value:
            missing.append("hpi.character")
        else:
            score += 3

    if _has_associated_context(store, plan):
        score += 12
    else:
        missing.append(_associated_slot_for_plan(plan))

    pattern_slots = [f"specific.{key}" for key, _ in plan.pattern_required]
    answered_pattern = [slot for slot in pattern_slots if store.is_answered(slot)]
    if pattern_slots:
        score += round(20 * len(answered_pattern) / len(pattern_slots))
    else:
        score += 8

    critical_slots = _critical_pattern_slots(plan)
    red_flag_slots = critical_slots or pattern_slots
    red_flags_checked = all(store.is_answered(slot) for slot in red_flag_slots)
    for slot in red_flag_slots:
        if not store.is_answered(slot):
            missing.append(slot)

    safety_slots_covered = (
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

    if store.is_answered("hpi.relevant_history"):
        score += 8
    elif assistant_count < 4:
        missing.append("hpi.relevant_history")

    score = min(score, 100)
    high_value_missing = _unique(missing)
    required_slots_covered = _required_slots_covered(store, plan, relaxed_low_value)
    threshold = 80 if plan.protocol_id in HIGH_RISK_PROTOCOLS else 74
    doctor_summary_ready = (
        score >= threshold
        and red_flags_checked
        and _has_core_doctor_summary(store, plan, relaxed_low_value)
    )

    askable = [
        slot for slot in high_value_missing
        if requested_slots.count(slot) < 2
    ]
    next_best_slot = _pick_next_slot(askable, plan, clinical_missing)

    preferred_limit = min(max_rounds, PREFERRED_MAX_REVIEW_QUESTIONS)
    if assistant_count >= max_rounds:
        can_finish = True
        reason = "达到最大追问轮数，已将未采集信息标记给医生补问"
    elif doctor_summary_ready and safety_slots_covered:
        can_finish = True
        reason = "主诉、病程、严重程度、危险信号和安全信息已具备医生预诊价值"
    elif assistant_count >= preferred_limit and score >= 65 and red_flags_checked:
        can_finish = True
        reason = "核心预诊档案已具备价值，低收益信息不再继续打扰患者"
    elif not next_best_slot and score >= 55:
        can_finish = True
        reason = "剩余缺失项已多次追问或价值较低，进入医生侧总结"
    else:
        can_finish = False
        reason = _missing_reason(next_best_slot, plan)

    next_question = None
    if not can_finish and next_best_slot:
        next_question = _question_for_review_slot(next_best_slot, plan, store)

    return IntakeQualityReview(
        doctor_value_score=score,
        can_finish_intake=can_finish,
        doctor_summary_ready=doctor_summary_ready,
        red_flags_checked=red_flags_checked,
        required_slots_covered=required_slots_covered,
        safety_slots_covered=safety_slots_covered,
        high_value_missing_slots=high_value_missing,
        next_best_slot=None if can_finish else next_best_slot,
        next_best_question=next_question,
        reason=reason,
        task_tree=task_tree,
    )


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


def _required_slots_covered(
    store: FactStore,
    plan: ResolvedIntakePlan,
    relaxed_low_value: bool = False,
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
    relaxed_low_value: bool = False,
) -> bool:
    return (
        store.is_answered("hpi.chief_complaint")
        and (store.is_answered("hpi.onset") or store.is_answered("hpi.timing"))
        and (_has_severity_or_quantifier(store, plan) or not _severity_required(plan, relaxed_low_value))
        and _has_associated_context(store, plan)
    )


def _pick_next_slot(
    slots: list[str],
    plan: ResolvedIntakePlan,
    clinical_missing_slots: list[str] | None = None,
) -> str | None:
    if not slots:
        return None
    clinical_missing = set(clinical_missing_slots or [])
    return sorted(
        slots,
        key=lambda slot: (0 if slot in clinical_missing else 1, _review_slot_priority(slot, plan)),
    )[0]


def _review_slot_priority(slot: str, plan: ResolvedIntakePlan) -> int:
    tree_priority = slot_task_priority(slot, plan)
    if tree_priority < 999:
        return tree_priority
    if slot == "hpi.chief_complaint":
        return 5
    if slot in set(_critical_pattern_slots(plan)):
        return 10
    if slot in {"hpi.onset", "hpi.timing", "hpi.severity"}:
        return 20
    if slot.startswith("specific."):
        return 30
    if slot in {"hpi.location", "hpi.character", "hpi.associated_symptoms"}:
        return 35
    if slot.startswith("safety."):
        return 45
    if slot == "hpi.relevant_history":
        return 70
    return 99


def _question_for_review_slot(
    slot: str,
    plan: ResolvedIntakePlan,
    store: FactStore,
) -> str:
    """
    上下文感知的问题生成（Prompter 模式，参考论文 4.3 节）。

    优先生成引用已知信息的追问，使对话更自然、不显机械。
    只在有明确已知上下文时才附加引用，否则退回固定问题。
    """
    # ── 安全槽位特殊处理 ──────────────────────────────────────────────
    if slot == "safety.current_medications" and not store.is_answered("safety.allergies"):
        return "您现在为这个症状或平时在用什么药吗？另外有没有药物或食物过敏？"
    if slot == "safety.allergies" and store.is_answered("safety.current_medications"):
        meds = store.value("safety.current_medications") or "用药情况"
        return f"您提到{meds}，那有没有药物或食物过敏？"

    # ── onset：引用主诉 ───────────────────────────────────────────────
    if slot in ("hpi.onset", "hpi.timing"):
        chief = store.value("hpi.chief_complaint")
        if chief:
            return f"您提到{chief}，这个症状大概是什么时候开始的？"
        return REVIEW_SLOT_QUESTIONS.get(slot) or BASE_SLOT_SPECS[slot].question

    # ── 伴随症状：引用主诉或协议名 ──────────────────────────────────
    if slot in ("hpi.associated_symptoms", "specific.associated_fever_symptoms"):
        chief = store.value("hpi.chief_complaint")
        if chief:
            return f"除了{chief}，还有其他伴随症状吗，比如寒战、咳嗽、恶心或其他不适？"
        return REVIEW_SLOT_QUESTIONS.get(slot) or BASE_SLOT_SPECS.get(slot, BASE_SLOT_SPECS["hpi.associated_symptoms"]).question

    # ── severity/max_temperature：引用已知持续时间或主诉 ──────────────
    if slot in ("hpi.severity", "specific.max_temperature"):
        timing = store.value("hpi.timing") or store.value("hpi.onset")
        chief = store.value("hpi.chief_complaint")
        if slot == "specific.max_temperature":
            base = REVIEW_SLOT_QUESTIONS.get(slot, "最高体温到多少度？")
            if timing:
                return f"从{timing}开始，{base}"
            return base
        if chief and timing:
            return f"您{chief}已经{timing}了，严重程度大概怎么样？比如疼痛 0 到 10 分。"
        return REVIEW_SLOT_QUESTIONS.get(slot) or BASE_SLOT_SPECS["hpi.severity"].question

    # ── pattern_specific：引用协议名 + 已知主诉 ──────────────────────
    if slot.startswith("specific."):
        if slot in REVIEW_SLOT_QUESTIONS:
            chief = store.value("hpi.chief_complaint")
            base = REVIEW_SLOT_QUESTIONS[slot]
            # 只对较长的固定问题才不加前缀，短问题加主诉上下文更自然
            if chief and len(base) < 30:
                return f"关于您的{chief}，{base}"
            return base
        return f"关于{plan.protocol_label}，还需要了解{slot_label(slot, plan)}，方便的话请补充一下？"

    # ── 其余槽位：固定问题 ────────────────────────────────────────────
    spec = BASE_SLOT_SPECS.get(slot)
    if spec:
        return spec.question
    return "请问还有什么重要信息需要告诉我吗？"


def _missing_reason(slot: str | None, plan: ResolvedIntakePlan) -> str:
    if not slot:
        return "预诊档案仍有缺失项"
    return f"医生预诊档案还缺少：{slot_label(slot, plan)}"


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


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


def _locked_protocol_id(state: TriageGraphState) -> str | None:
    protocol_id = state.get("intake_protocol_id")
    if protocol_id and protocol_id != "generic_opqrst":
        return protocol_id
    return None
