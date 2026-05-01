"""
确定性预诊事实抽取规则。

LLM 负责理解开放表达，但药名、否认过敏这类高价值短语应该有规则兜底，
避免因为一次抽取遗漏导致重复追问。
"""

from __future__ import annotations

import re
from typing import Iterable


MEDICATION_KEYWORDS: tuple[str, ...] = (
    "布洛芬",
    "对乙酰氨基酚",
    "扑热息痛",
    "泰诺",
    "散利痛",
    "阿司匹林",
    "头孢",
    "阿莫西林",
    "左氧氟沙星",
    "蒙脱石散",
    "思密达",
    "黄连素",
    "洛哌丁胺",
    "奥美拉唑",
    "吗丁啉",
    "氯雷他定",
    "西替利嗪",
    "退烧药",
    "退热药",
    "止痛药",
    "感冒药",
    "消炎药",
    "抗生素",
    "降压药",
    "降糖药",
    "胰岛素",
)

ANTIPYRETIC_KEYWORDS: tuple[str, ...] = (
    "布洛芬",
    "对乙酰氨基酚",
    "扑热息痛",
    "泰诺",
    "退烧药",
    "退热药",
)

NEGATED_MEDICATION_PATTERNS: tuple[str, ...] = (
    "没吃{med}",
    "没有吃{med}",
    "没用{med}",
    "没有用{med}",
    "未吃{med}",
    "未用{med}",
)

NO_MEDICATION_PATTERNS: tuple[str, ...] = (
    "没吃药",
    "没有吃药",
    "没用药",
    "没有用药",
    "没服药",
    "没有服药",
    "未用药",
)

NO_ALLERGY_PATTERNS: tuple[str, ...] = (
    "没有药物过敏",
    "无药物过敏",
    "没有食物过敏",
    "无食物过敏",
    "没有过敏史",
    "无过敏史",
    "不过敏",
)


def extract_deterministic_facts(
    messages: Iterable[dict],
    protocol_id: str | None = None,
) -> list[dict]:
    """从用户原文里抽取高置信度事实。"""
    text = _user_text(messages)
    facts: list[dict] = []

    meds, medication_evidence = _extract_medications(text)
    if meds:
        value = "，".join(meds)
        facts.append({
            "slot": "safety.current_medications",
            "status": "present",
            "value": value,
            "evidence": medication_evidence or value,
            "confidence": 0.99,
        })
        antipyretics = [med for med in meds if med in ANTIPYRETIC_KEYWORDS]
        if protocol_id == "fever" and antipyretics:
            facts.append({
                "slot": "specific.antipyretics",
                "status": "present",
                "value": _antipyretic_value(text, antipyretics),
                "evidence": medication_evidence or "，".join(antipyretics),
                "confidence": 0.98,
            })
    elif _has_no_medication(text):
        facts.append({
            "slot": "safety.current_medications",
            "status": "absent",
            "value": "没有用药",
            "evidence": _first_matching_phrase(text, NO_MEDICATION_PATTERNS) or "没有用药",
            "confidence": 0.95,
        })

    if _has_no_allergy(text):
        facts.append({
            "slot": "safety.allergies",
            "status": "absent",
            "value": "无药物或食物过敏",
            "evidence": _first_matching_phrase(text, NO_ALLERGY_PATTERNS) or "无过敏史",
            "confidence": 0.95,
        })

    return facts


def _user_text(messages: Iterable[dict]) -> str:
    return "。".join(
        str(m.get("content") or "")
        for m in messages
        if m.get("role") == "user" and m.get("content")
    )


def _extract_medications(text: str) -> tuple[list[str], str]:
    meds: list[str] = []
    evidence_parts: list[str] = []
    for med in MEDICATION_KEYWORDS:
        if med not in text or _medication_is_negated(text, med):
            continue
        meds.append(med)
        evidence_parts.append(_sentence_containing(text, med))
    return _unique(meds), "；".join(_unique(evidence_parts))


def _medication_is_negated(text: str, med: str) -> bool:
    return any(pattern.format(med=med) in text for pattern in NEGATED_MEDICATION_PATTERNS)


def _has_no_medication(text: str) -> bool:
    return any(pattern in text for pattern in NO_MEDICATION_PATTERNS)


def _has_no_allergy(text: str) -> bool:
    return any(pattern in text for pattern in NO_ALLERGY_PATTERNS)


def _first_matching_phrase(text: str, patterns: tuple[str, ...]) -> str | None:
    return next((pattern for pattern in patterns if pattern in text), None)


def _sentence_containing(text: str, keyword: str) -> str:
    parts = re.split(r"[。！？!?；;\n\r]", text)
    for part in parts:
        if keyword in part:
            return part.strip()
    return keyword


def _antipyretic_value(text: str, meds: list[str]) -> str:
    value = "，".join(meds)
    effect = _extract_effect_phrase(text)
    if effect:
        return f"{value}，{effect}"
    return value


def _extract_effect_phrase(text: str) -> str | None:
    patterns = (
        r"(退了[一点些]?)",
        r"(能退[烧热]?)",
        r"(有效)",
        r"(没效果)",
        r"(没有效果)",
        r"(几小时后又[发]?烧)",
        r"(反复[发]?烧)",
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
    return "，".join(_unique(matches)) if matches else None


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
