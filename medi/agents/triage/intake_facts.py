"""
预问诊事实存储。

这一层把“护士已经知道什么”从 LLM prompt 中拿出来：
  1. LLM 只负责从对话中抽取事实
  2. FactStore 合并已知/否认/未知信息
  3. IntakeReviewNode 基于 FactStore 决定下一问

这样可以避免重复问已知信息，也避免把问诊流程写成大量主诉枚举。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from medi.agents.triage.graph.state import (
    CollectionStatus,
    GeneralCondition,
    OPQRSTStatus,
    SymptomData,
)
from medi.agents.triage.intake_protocols import ResolvedIntakePlan


ANSWERED_STATUSES = {"present", "absent", "not_applicable"}
UNKNOWN_STATUSES = {"unknown", "partial", "missing", ""}


@dataclass(frozen=True)
class SlotSpec:
    slot: str
    label: str
    question: str
    priority: int


@dataclass(frozen=True)
class ClinicalFact:
    slot: str
    status: str
    value: str | None = None
    evidence: str = ""
    confidence: float = 0.0
    source_turn: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalFact | None":
        slot = str(data.get("slot") or "").strip()
        if not slot:
            return None
        status = normalize_status(data.get("status"))
        return cls(
            slot=slot,
            status=status,
            value=_clean_value(data.get("value")),
            evidence=str(data.get("evidence") or ""),
            confidence=_safe_float(data.get("confidence"), default=0.0),
            source_turn=data.get("source_turn"),
        )

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "status": self.status,
            "value": self.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_turn": self.source_turn,
        }


class FactStore:
    def __init__(self, facts: Iterable[ClinicalFact] | None = None) -> None:
        self._facts: dict[str, ClinicalFact] = {}
        for fact in facts or ():
            self.merge_fact(fact)

    @classmethod
    def from_state(cls, raw_facts: list[dict] | None) -> "FactStore":
        facts = []
        for item in raw_facts or []:
            fact = ClinicalFact.from_dict(item)
            if fact is not None:
                facts.append(fact)
        return cls(facts)

    def merge_items(self, raw_facts: Iterable[dict], source_turn: int | None = None) -> None:
        for item in raw_facts:
            fact = ClinicalFact.from_dict({**item, "source_turn": source_turn})
            if fact is not None:
                self.merge_fact(fact)

    def merge_fact(self, fact: ClinicalFact) -> None:
        current = self._facts.get(fact.slot)
        if current is None or should_replace(current, fact):
            self._facts[fact.slot] = fact

    def get(self, slot: str) -> ClinicalFact | None:
        return self._facts.get(slot)

    def is_answered(self, slot: str) -> bool:
        fact = self.get(slot)
        if fact is None:
            return False
        if fact.status in ANSWERED_STATUSES:
            return True
        if fact.status == "partial":
            return False
        return bool(fact.value) and fact.status not in UNKNOWN_STATUSES

    def is_collected(self, slot: str) -> bool:
        """是否已经从用户那里得到过该槽位的响应，unknown 也算已收集。"""
        fact = self.get(slot)
        if fact is None:
            return False
        return fact.status not in {"", "missing", "partial"}

    def value(self, slot: str) -> str | None:
        fact = self.get(slot)
        if fact is None:
            return None
        return fact.value

    def to_state(self) -> list[dict]:
        return [fact.to_dict() for fact in self._facts.values()]

    def prompt_context(self) -> str:
        if not self._facts:
            return "（暂无已抽取事实）"
        lines = []
        for fact in sorted(self._facts.values(), key=lambda f: SLOT_ORDER.get(f.slot, 999)):
            label = slot_label(fact.slot)
            value = fact.value if fact.value is not None else fact.status
            lines.append(f"- {label}: {value}（{fact.status}；证据：{fact.evidence or '无'}）")
        return "\n".join(lines)

    def to_symptom_data(self, raw_descriptions: list[str]) -> SymptomData:
        gc = self._build_general_condition()
        return SymptomData(
            raw_descriptions=raw_descriptions,
            onset=self.value("hpi.onset"),
            provocation=self.value("hpi.aggravating_alleviating"),
            quality=self.value("hpi.character"),
            region=self.value("hpi.location"),
            severity=self.value("hpi.severity"),
            max_temperature=self.value("specific.max_temperature"),
            frequency=self.value("specific.frequency"),
            time_pattern=self.value("hpi.timing"),
            radiation=self.value("hpi.radiation"),
            accompanying=_split_list(
                self.value("hpi.associated_symptoms")
                or self.value("specific.associated_fever_symptoms")
            ),
            relevant_history=self.value("hpi.relevant_history"),
            medications=_split_list(self.value("safety.current_medications")),
            allergies=_split_list(self.value("safety.allergies")),
            general_condition=gc if any(v is not None for v in gc.values()) else None,
        )

    def _build_general_condition(self) -> GeneralCondition:
        """从 gc.* 槽位构建一般情况结构。"""
        return GeneralCondition(
            mental_status=self.value("gc.mental_status"),
            sleep=self.value("gc.sleep"),
            appetite=self.value("gc.appetite"),
            bowel=self.value("gc.bowel"),
            urination=self.value("gc.urination"),
            weight_change=self.value("gc.weight_change"),
        )


BASE_SLOT_SPECS: dict[str, SlotSpec] = {
    "hpi.chief_complaint": SlotSpec("hpi.chief_complaint", "主诉", "您今天主要是哪里不舒服，能描述一下吗？", 10),
    "hpi.onset": SlotSpec("hpi.onset", "发作时间", "这个症状是什么时候开始的？", 20),
    "hpi.location": SlotSpec("hpi.location", "部位", "您能告诉我具体是哪个部位不舒服吗？", 30),
    "hpi.severity": SlotSpec("hpi.severity", "严重程度", "这个不适大概有多严重？比如疼痛 0 到 10 分，或发热的最高体温、腹泻次数等。", 40),
    "hpi.timing": SlotSpec("hpi.timing", "时间特征", "这个症状是一直持续还是时好时坏？大概持续多久了？", 50),
    "hpi.character": SlotSpec("hpi.character", "症状性质", "您能描述一下这个不适的特点吗？比如疼痛性质、腹泻性状、咳嗽有无痰等。", 60),
    "hpi.aggravating_alleviating": SlotSpec("hpi.aggravating_alleviating", "加重/缓解因素", "有什么情况会让它加重或缓解吗？", 70),
    "hpi.radiation": SlotSpec("hpi.radiation", "放射/扩散", "这个不适会向其他部位扩散或放射吗？", 80),
    "hpi.associated_symptoms": SlotSpec("hpi.associated_symptoms", "伴随症状", "除了这个主要不适，还有其他伴随症状吗？", 90),
    "safety.current_medications": SlotSpec("safety.current_medications", "当前用药", "您目前为这个症状或平时在服用什么药物吗？没有也可以直接说没有。", 110),
    "safety.allergies": SlotSpec("safety.allergies", "过敏史", "有没有药物或食物过敏？", 111),
    "hpi.relevant_history": SlotSpec("hpi.relevant_history", "相关既往史", "以前有过类似情况，或有什么相关疾病史吗？", 140),
    # ── 一般情况（General Condition，论文 Table 5）────────────────────────────
    # 优先级 150+，不作为必填项，作为补充采集目标
    "gc.mental_status": SlotSpec("gc.mental_status", "精神状态", "精神状态怎么样，有没有嗜睡、烦躁或意识不清？", 150),
    "gc.sleep": SlotSpec("gc.sleep", "睡眠", "睡眠情况怎么样？", 155),
    "gc.appetite": SlotSpec("gc.appetite", "食欲", "食欲和进食情况怎么样，能正常吃东西吗？", 160),
    "gc.bowel": SlotSpec("gc.bowel", "大便", "大便情况有没有变化，比如便秘、腹泻或颜色异常？", 165),
    "gc.urination": SlotSpec("gc.urination", "小便", "小便情况有没有变化，比如尿量减少、尿频或尿痛？", 170),
    "gc.weight_change": SlotSpec("gc.weight_change", "体重变化", "近期体重有没有明显变化？", 175),
}


FIELD_TO_SLOT = {
    "chief_complaint": "hpi.chief_complaint",
    "opqrst.onset": "hpi.onset",
    "opqrst.location": "hpi.location",
    "opqrst.quality": "hpi.character",
    "opqrst.severity": "hpi.severity",
    "opqrst.time_pattern": "hpi.timing",
    "opqrst.provocation": "hpi.aggravating_alleviating",
    "opqrst.radiation": "hpi.radiation",
    "associated_symptoms": "hpi.associated_symptoms",
    "relevant_history": "hpi.relevant_history",
}


SLOT_ORDER = {slot: spec.priority for slot, spec in BASE_SLOT_SPECS.items()}


def required_slots_for_plan(plan: ResolvedIntakePlan) -> list[str]:
    slots: list[str] = []
    for field in plan.required_fields:
        if field == "medications_allergies":
            slots.extend(["safety.current_medications", "safety.allergies"])
            continue
        mapped = FIELD_TO_SLOT.get(field)
        if mapped:
            slots.append(mapped)

    # 医生 HPI 至少需要知道伴随症状；相关病史可在轮数允许时补充。
    slots.append("hpi.associated_symptoms")
    for key, _ in plan.pattern_required:
        slots.append(f"specific.{key}")
    slots.append("hpi.relevant_history")
    return _unique(slots)


def collection_status_from_facts(
    store: FactStore,
    plan: ResolvedIntakePlan,
    complete: bool,
    reason: str,
) -> CollectionStatus:
    opqrst = OPQRSTStatus(
        onset=_slot_status(store, "hpi.onset"),
        provocation=_slot_status(store, "hpi.aggravating_alleviating"),
        quality=_slot_status(store, "hpi.character"),
        location=_slot_status(store, "hpi.location"),
        severity=_slot_status(store, "hpi.severity"),
        time_pattern=_slot_status(store, "hpi.timing"),
        radiation=_slot_status(store, "hpi.radiation"),
    )

    meds = _slot_status(store, "safety.current_medications")
    allergies = _slot_status(store, "safety.allergies")
    if meds == "complete" and allergies == "complete":
        med_allergy_status = "complete"
    elif meds != "missing" or allergies != "missing":
        med_allergy_status = "partial"
    else:
        med_allergy_status = "missing"

    pattern = {
        key: _slot_status(store, f"specific.{key}")
        for key, _ in plan.pattern_required
    }

    return CollectionStatus(
        chief_complaint=_slot_status(store, "hpi.chief_complaint"),
        opqrst=opqrst,
        associated_symptoms=_slot_status(store, "hpi.associated_symptoms"),
        relevant_history=_slot_status(store, "hpi.relevant_history"),
        medications_allergies=med_allergy_status,
        pattern_specific=pattern,
        can_conclude=complete,
        reason=reason,
    )


def extraction_slot_prompt(plan: ResolvedIntakePlan) -> str:
    lines = []
    all_slots = _unique(list(BASE_SLOT_SPECS), required_slots_for_plan(plan))
    for slot in all_slots:
        lines.append(f"- {slot}: {slot_label(slot, plan)}")
    return "\n".join(lines)


def normalize_status(value) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "yes": "present",
        "no": "absent",
        "none": "absent",
        "complete": "present",
        "denied": "absent",
        "否认": "absent",
        "没有": "absent",
        "无": "absent",
        "unknown": "unknown",
        "missing": "unknown",
        "partial": "partial",
        "not_applicable": "not_applicable",
    }
    return mapping.get(raw, raw if raw in ANSWERED_STATUSES | UNKNOWN_STATUSES else "present")


def should_replace(current: ClinicalFact, new: ClinicalFact) -> bool:
    if current.status in UNKNOWN_STATUSES and new.status not in UNKNOWN_STATUSES:
        return True
    if new.confidence > current.confidence + 0.15:
        return True
    if new.value and new.value != current.value and new.status in ANSWERED_STATUSES:
        return True
    return False


def slot_label(slot: str, plan: ResolvedIntakePlan | None = None) -> str:
    spec = BASE_SLOT_SPECS.get(slot)
    if spec:
        return spec.label
    if slot.startswith("specific.") and plan is not None:
        key = slot.removeprefix("specific.")
        return dict(plan.pattern_required).get(key, key)
    return slot


def _slot_status(store: FactStore, slot: str) -> str:
    fact = store.get(slot)
    if fact is None:
        return "missing"
    if fact.status in ANSWERED_STATUSES:
        return "complete"
    if fact.status == "partial":
        return "partial"
    return "missing"


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    separators = [",", "，", "、", ";", "；"]
    items = [value]
    for sep in separators:
        next_items = []
        for item in items:
            next_items.extend(item.split(sep))
        items = next_items
    return [item.strip() for item in items if item.strip() and item.strip() not in {"无", "没有"}]


def _clean_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown"}:
        return None
    return text


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique(items: Iterable[str], extra: Iterable[str] | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in (items, extra or ()):
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result
