"""Clinical summary text builders shared by triage nodes.

This module builds runtime-only text from structured encounter data. It does
not define or persist clinical data models.
"""

from __future__ import annotations

import re

from medi.agents.triage.clinical_facts import ClinicalFactStore


def _format_severity_line(value: str) -> str:
    """Format pain scores separately from temperatures and frequencies."""
    text = value.strip()
    if _looks_like_temperature(text):
        return f"最高体温：{text}"
    if re.search(r"\d+\s*(次|回)", text):
        return f"频率/次数：{text}"
    if re.fullmatch(r"(10|[0-9])", text):
        return f"疼痛评分：{text}/10"
    if re.fullmatch(r"(10|[0-9])\s*/\s*10", text):
        return f"疼痛评分：{text}"
    return f"严重程度：{text}"


def _looks_like_temperature(value: str) -> bool:
    return bool(re.search(r"(体温|发热|发烧|高烧|低烧|℃|度|[3-4]\d(?:\.\d+)?)", value))


def build_symptom_summary_from_record(
    preconsultation_record: dict | None,
    clinical_facts: list[dict] | None,
    messages: list[dict] | None = None,
) -> str:
    """Build a readable clinical summary directly from record/facts."""
    record = preconsultation_record or {}
    store = ClinicalFactStore.from_state(clinical_facts or [])

    lines: list[str] = []
    _append_line(lines, "主诉", _first_text(
        _record_value(record, "t4_chief_complaint.generated"),
        _record_value(record, "t4_chief_complaint.draft"),
        _record_value(record, "t2_hpi.chief_complaint"),
        store.value("hpi.chief_complaint"),
    ))
    _append_line(lines, "起病时间", _first_text(
        _record_value(record, "t2_hpi.onset"),
        store.value("hpi.onset"),
    ))
    _append_line(lines, "相关暴露", _first_text(
        _record_value(record, "t2_hpi.exposure_event"),
        store.value("hpi.exposure_event"),
    ))
    _append_line(lines, "暴露后症状", _first_text(
        _record_value(record, "t2_hpi.exposure_symptoms"),
        store.value("hpi.exposure_symptoms"),
    ))
    _append_line(lines, "部位", _first_text(
        _record_value(record, "t2_hpi.location"),
        store.value("hpi.location"),
    ))
    _append_line(lines, "性质", _first_text(
        _record_value(record, "t2_hpi.character"),
        store.value("hpi.character"),
    ))
    _append_line(lines, "持续时间", _first_text(
        _record_value(record, "t2_hpi.duration"),
        store.value("hpi.duration"),
    ))
    _append_line(lines, "时间特征", _first_text(
        _record_value(record, "t2_hpi.timing"),
        store.value("hpi.timing"),
    ))

    max_temperature = _first_text(
        _record_value(record, "t2_hpi.specific.max_temperature"),
        store.value("specific.max_temperature"),
    )
    if max_temperature:
        _append_line(lines, "最高体温", max_temperature)

    frequency = _first_text(
        _record_value(record, "t2_hpi.specific.frequency"),
        store.value("specific.frequency"),
    )
    if frequency:
        _append_line(lines, "频率/次数", frequency)

    severity = _first_text(
        _record_value(record, "t2_hpi.severity"),
        store.value("hpi.severity"),
    )
    if severity and not (max_temperature and _looks_like_temperature(severity)):
        lines.append(_format_severity_line(severity))

    _append_line(lines, "放射痛", _first_text(
        _record_value(record, "t2_hpi.radiation"),
        store.value("hpi.radiation"),
    ))
    _append_line(lines, "加重/缓解", _first_text(
        _record_value(record, "t2_hpi.aggravating_alleviating"),
        store.value("hpi.aggravating_alleviating"),
    ))

    associated = _dedupe_text(
        _text_list(_record_value(record, "t2_hpi.associated_symptoms"))
        + _text_list(_record_value(record, "t2_hpi.specific.associated_fever_symptoms"))
        + _text_list(store.value("hpi.associated_symptoms"))
        + _text_list(store.value("specific.associated_fever_symptoms"))
    )
    if associated:
        _append_line(lines, "伴随症状", "、".join(associated))

    _append_line(lines, "检查/诊断经过", _first_text(
        _record_value(record, "t2_hpi.diagnostic_history"),
        store.value("hpi.diagnostic_history"),
    ))
    _append_line(lines, "治疗经过", _first_text(
        _record_value(record, "t2_hpi.therapeutic_history"),
        store.value("hpi.therapeutic_history"),
    ))
    _append_line(lines, "相关既往史", _first_text(
        _record_value(record, "t2_hpi.relevant_history"),
        store.value("hpi.relevant_history"),
    ))
    _append_line(lines, "用药", _first_text(
        _record_value(record, "t3_background.current_medications"),
        store.value("safety.current_medications"),
    ))
    _append_line(lines, "过敏", _first_text(
        _record_value(record, "t3_background.allergy_history"),
        store.value("safety.allergies"),
    ))

    general_condition = _general_condition_line(record, store)
    if general_condition:
        _append_line(lines, "一般情况", general_condition)

    if not lines:
        fallback = _recent_user_messages(messages or [])
        if fallback:
            return " ".join(fallback)
    return "\n".join(lines) if lines else "（症状信息待采集）"


def build_department_query_from_record(
    preconsultation_record: dict | None,
    clinical_facts: list[dict] | None,
    messages: list[dict] | None = None,
) -> str:
    """Build search text for department retrieval from record/facts."""
    summary = build_symptom_summary_from_record(preconsultation_record, clinical_facts, messages)
    if summary != "（症状信息待采集）":
        return summary.replace("\n", " ")
    return " ".join(_recent_user_messages(messages or []))


def _append_line(lines: list[str], label: str, value: str | None) -> None:
    if value:
        lines.append(f"{label}：{value}")


def _first_text(*values) -> str:
    for value in values:
        text = _payload_text(value)
        if text:
            return text
    return ""


def _record_value(record: dict, path: str):
    current = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _payload_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("value", "generated", "draft", "department"):
            if key in value:
                return _payload_text(value.get(key))
        return ""
    if isinstance(value, (list, tuple, set)):
        return "、".join(_text_list(value))
    return str(value).strip()


def _text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_text_list(item))
        return result
    text = _payload_text(value)
    if not text:
        return []
    return [item.strip() for item in re.split(r"[、,，;；\n]+", text) if item.strip()]


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _general_condition_line(record: dict, store: ClinicalFactStore) -> str:
    fields = (
        ("mental_status", "精神/意识", "gc.mental_status"),
        ("sleep", "睡眠", "gc.sleep"),
        ("appetite", "食欲", "gc.appetite"),
        ("bowel", "大便", "gc.bowel"),
        ("urination", "小便", "gc.urination"),
        ("weight_change", "体重", "gc.weight_change"),
    )
    parts = []
    for key, label, slot in fields:
        value = _first_text(
            _record_value(record, f"t2_hpi.general_condition.{key}"),
            store.value(slot),
        )
        if value:
            parts.append(f"{label}{value}")
    return "，".join(parts)


def _recent_user_messages(messages: list[dict]) -> list[str]:
    return [
        str(m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user" and m.get("content")
    ][-3:]
