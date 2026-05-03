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
