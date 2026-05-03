"""
Structured medical record projection for the pre-consultation flow.

The Recipient role keeps an evolving CC/HPI/PH/Triage draft across dialogue
turns. The current implementation derives that draft from FactStore so the
record stays deterministic and evidence-linked while the graph still uses the
existing intake node.
"""

from __future__ import annotations

from copy import deepcopy

from medi.agents.triage.intake_facts import ClinicalFact, FactStore
from medi.agents.triage.intake_protocols import ResolvedIntakePlan


TRIAGE_DEPARTMENT_HINTS: dict[str, tuple[str, str]] = {
    "chest_pain": ("内科", "心血管内科"),
    "dyspnea": ("内科", "呼吸内科"),
    "abdominal_pain": ("内科", "消化内科"),
    "headache": ("内科", "神经内科"),
    "trauma": ("外科", "创伤外科"),
    "diarrhea_vomiting": ("内科", "消化内科"),
    "rash_allergy": ("皮肤科", "过敏反应门诊"),
    "dizziness_syncope": ("内科", "神经内科"),
    "fever": ("内科", "发热门诊"),
    "ear_pain": ("五官科", "耳鼻喉科"),
    "generic_opqrst": ("全科医学科", "普通门诊"),
}


HPI_SLOT_FIELDS: dict[str, str] = {
    "hpi.chief_complaint": "chief_complaint",
    "hpi.onset": "onset",
    "hpi.exposure_event": "exposure_event",
    "hpi.exposure_symptoms": "exposure_symptoms",
    "hpi.location": "location",
    "hpi.character": "character",
    "hpi.duration": "duration",
    "hpi.severity": "severity",
    "hpi.timing": "timing",
    "hpi.progression": "progression",
    "hpi.aggravating_alleviating": "aggravating_alleviating",
    "hpi.radiation": "radiation",
    "hpi.associated_symptoms": "associated_symptoms",
    "hpi.diagnostic_history": "diagnostic_history",
    "hpi.therapeutic_history": "therapeutic_history",
    "hpi.relevant_history": "relevant_history",
}

GENERAL_CONDITION_SLOT_FIELDS: dict[str, str] = {
    "gc.mental_status": "mental_status",
    "gc.sleep": "sleep",
    "gc.appetite": "appetite",
    "gc.bowel": "bowel",
    "gc.urination": "urination",
    "gc.weight_change": "weight_change",
    "specific.mental_status": "mental_status",
    "specific.intake_urination": "urination",
}

PAST_HISTORY_SLOT_FIELDS: dict[str, str] = {
    "ph.disease_history": "disease_history",
    "ph.immunization_history": "immunization_history",
    "ph.surgical_history": "surgical_history",
    "ph.trauma_history": "trauma_history",
    "ph.blood_transfusion_history": "blood_transfusion_history",
    "ph.allergy_history": "allergy_history",
    "hpi.relevant_history": "disease_history",
    "safety.current_medications": "current_medications",
    "safety.allergies": "allergy_history",
}


def update_medical_record(
    current: dict | None,
    *,
    store: FactStore,
    plan: ResolvedIntakePlan,
) -> dict:
    """Build an evidence-linked medical record snapshot from accumulated facts."""
    record = _base_record(current)

    record["triage"].update({
        "protocol_id": plan.protocol_id,
        "protocol_label": plan.protocol_label,
        "overlay_ids": plan.overlay_ids,
    })
    _update_triage_hint(record, plan)

    for slot, field in HPI_SLOT_FIELDS.items():
        _copy_fact(store, record["hpi"], field, slot)

    for slot, field in GENERAL_CONDITION_SLOT_FIELDS.items():
        _copy_fact(store, record["hpi"]["general_condition"], field, slot)

    for key, label in plan.pattern_required:
        _copy_fact(
            store,
            record["hpi"]["specific"],
            key,
            f"specific.{key}",
            label=label,
        )

    for slot, field in PAST_HISTORY_SLOT_FIELDS.items():
        _copy_fact(store, record["ph"], field, slot)

    chief_complaint = store.get("hpi.chief_complaint")
    if chief_complaint is not None:
        record["cc"]["source"] = _fact_payload(chief_complaint)
        if chief_complaint.value:
            record["cc"]["draft"] = chief_complaint.value
        else:
            record["cc"].setdefault("draft", "")

    generated_cc = _generate_chief_complaint(record)
    if generated_cc and not _payload_value(record["cc"].get("generated")):
        record["cc"]["generated"] = generated_cc

    return record


def _base_record(current: dict | None) -> dict:
    record = deepcopy(current) if current else {}
    record.setdefault("triage", {})
    record.setdefault("hpi", {})
    record.setdefault("ph", {})
    record.setdefault("cc", {})
    record["hpi"].setdefault("general_condition", {})
    record["hpi"].setdefault("specific", {})
    return record


def _update_triage_hint(record: dict, plan: ResolvedIntakePlan) -> None:
    primary, secondary = TRIAGE_DEPARTMENT_HINTS.get(
        plan.protocol_id,
        TRIAGE_DEPARTMENT_HINTS["generic_opqrst"],
    )
    record["triage"].setdefault(
        "primary_department",
        {
            "department": primary,
            "confidence": 0.55,
            "reason": f"根据当前预问诊主题初步判断为{primary}",
        },
    )
    record["triage"].setdefault(
        "secondary_department",
        {
            "department": secondary,
            "confidence": 0.55,
            "reason": f"根据当前预问诊主题初步判断为{secondary}",
        },
    )


def _generate_chief_complaint(record: dict) -> str:
    hpi = record.get("hpi") or {}
    complaint = _payload_value(hpi.get("chief_complaint"))
    if not complaint:
        return ""
    duration = (
        _payload_value(hpi.get("duration"))
        or _payload_value(hpi.get("timing"))
        or _payload_value(hpi.get("onset"))
    )
    if duration and duration not in complaint:
        return f"{complaint}{duration}"
    return complaint


def _copy_fact(
    store: FactStore,
    target: dict,
    field: str,
    slot: str,
    *,
    label: str | None = None,
) -> None:
    fact = store.get(slot)
    if fact is None:
        return
    target[field] = _fact_payload(fact, label=label)


def _fact_payload(fact: ClinicalFact, *, label: str | None = None) -> dict:
    payload = {
        "slot": fact.slot,
        "status": fact.status,
        "value": fact.value,
        "evidence": fact.evidence,
        "confidence": fact.confidence,
        "source_turn": fact.source_turn,
    }
    if label is not None:
        payload["label"] = label
    return payload


def _payload_value(value) -> str:
    if isinstance(value, dict):
        raw = value.get("value")
        return str(raw).strip() if raw is not None else ""
    if value is None:
        return ""
    return str(value).strip()
