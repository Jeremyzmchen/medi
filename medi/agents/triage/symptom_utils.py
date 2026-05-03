"""Symptom presentation helpers shared by triage nodes."""

from __future__ import annotations

import re


def format_severity_line(value: str) -> str:
    """Format pain scores separately from temperatures and frequencies."""
    text = value.strip()
    if looks_like_temperature(text):
        return f"最高体温：{text}"
    if re.search(r"\d+\s*(次|回)", text):
        return f"频率/次数：{text}"
    if re.fullmatch(r"(10|[0-9])", text):
        return f"疼痛评分：{text}/10"
    if re.fullmatch(r"(10|[0-9])\s*/\s*10", text):
        return f"疼痛评分：{text}"
    return f"严重程度：{text}"


def looks_like_temperature(value: str) -> bool:
    return bool(re.search(r"(体温|发热|发烧|高烧|低烧|℃|度|[3-4]\d(?:\.\d+)?)", value))


def build_symptom_summary(symptom_data: dict) -> str:
    """Build a readable OPQRST summary from structured symptom data."""
    lines = []
    if symptom_data.get("region"):
        lines.append(f"部位：{symptom_data['region']}")
    if symptom_data.get("time_pattern"):
        lines.append(f"时间：{symptom_data['time_pattern']}")
    if symptom_data.get("onset"):
        lines.append(f"起病时间：{symptom_data['onset']}")
    if symptom_data.get("exposure_event"):
        lines.append(f"相关暴露：{symptom_data['exposure_event']}")
    if symptom_data.get("exposure_symptoms"):
        lines.append(f"暴露后症状：{symptom_data['exposure_symptoms']}")
    if symptom_data.get("quality"):
        lines.append(f"性质：{symptom_data['quality']}")
    max_temperature = symptom_data.get("max_temperature")
    if max_temperature:
        lines.append(f"最高体温：{max_temperature}")
    if symptom_data.get("frequency"):
        lines.append(f"频率/次数：{symptom_data['frequency']}")
    severity = symptom_data.get("severity")
    if severity and not (max_temperature and looks_like_temperature(str(severity))):
        lines.append(format_severity_line(str(severity)))
    if symptom_data.get("radiation"):
        lines.append(f"放射痛：{symptom_data['radiation']}")
    if symptom_data.get("provocation"):
        lines.append(f"加重/缓解：{symptom_data['provocation']}")
    if symptom_data.get("accompanying"):
        lines.append(f"伴随症状：{', '.join(symptom_data['accompanying'])}")
    if symptom_data.get("relevant_history"):
        lines.append(f"既往史：{symptom_data['relevant_history']}")
    if symptom_data.get("medications"):
        lines.append(f"用药：{', '.join(symptom_data['medications'])}")
    if symptom_data.get("allergies"):
        lines.append(f"过敏：{', '.join(symptom_data['allergies'])}")
    gc = symptom_data.get("general_condition")
    if gc:
        gc_parts = []
        if gc.get("mental_status"):
            gc_parts.append(f"精神{gc['mental_status']}")
        if gc.get("sleep"):
            gc_parts.append(f"睡眠{gc['sleep']}")
        if gc.get("appetite"):
            gc_parts.append(f"食欲{gc['appetite']}")
        if gc.get("bowel"):
            gc_parts.append(f"大便{gc['bowel']}")
        if gc.get("urination"):
            gc_parts.append(f"小便{gc['urination']}")
        if gc.get("weight_change"):
            gc_parts.append(f"体重{gc['weight_change']}")
        if gc_parts:
            lines.append(f"一般情况：{', '.join(gc_parts)}")
    if not lines and symptom_data.get("raw_descriptions"):
        return " ".join(symptom_data["raw_descriptions"][-3:])
    return "\n".join(lines) if lines else "（症状信息待采集）"
