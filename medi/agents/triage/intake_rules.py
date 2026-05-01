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
    facts.extend(_extract_exposure_timeline(text))

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


def _extract_exposure_timeline(text: str) -> list[dict]:
    """抽取“相关暴露”和“暴露后阴性”时间线，避免误当作症状起病。"""
    facts: list[dict] = []
    exposure = _extract_exposure_event(text)
    if exposure:
        facts.append({
            "slot": "hpi.exposure_event",
            "status": "present",
            "value": exposure,
            "evidence": exposure,
            "confidence": 0.96,
        })

    exposure_negative = _extract_exposure_negative(text)
    if exposure_negative:
        facts.append({
            "slot": "hpi.exposure_symptoms",
            "status": "absent",
            "value": exposure_negative,
            "evidence": exposure_negative,
            "confidence": 0.96,
        })

    onset = _extract_today_onset(text)
    if onset:
        facts.append({
            "slot": "hpi.onset",
            "status": "present",
            "value": onset,
            "evidence": onset,
            "confidence": 0.92,
        })

    return facts


def _user_text(messages: Iterable[dict]) -> str:
    return "。".join(
        str(m.get("content") or "")
        for m in messages
        if m.get("role") == "user" and m.get("content")
    )


def _extract_exposure_event(text: str) -> str | None:
    exposure_keywords = ("潜水", "游泳", "坐飞机", "飞行", "高铁", "爬山", "外伤", "撞到", "摔倒")
    if not any(keyword in text for keyword in exposure_keywords):
        return None
    match = re.search(
        r"((?:上周|上星期|上个月|昨天|前天|今天|刚才|最近|[一二三四五六七八九十\d]+天前)[^。！？!?；;\n\r]{0,24}?"
        r"(?:潜水|游泳|坐飞机|飞行|高铁|爬山|外伤|撞到|摔倒))",
        text,
    )
    if match:
        return match.group(1).strip("，,。；; ")
    return _sentence_containing_any(text, exposure_keywords)


def _extract_exposure_negative(text: str) -> str | None:
    if not any(keyword in text for keyword in ("潜水", "游泳", "坐飞机", "飞行", "外伤")):
        return None
    symptom_keywords = ("耳朵痛", "耳痛", "疼", "痛", "不适", "听力下降", "耳闷")
    negative_markers = ("没发现", "没有", "无", "没觉得", "未出现", "不觉得")
    for sentence in re.split(r"[。！？!?；;\n\r]", text):
        if not sentence.strip():
            continue
        if any(marker in sentence for marker in negative_markers) and any(symptom in sentence for symptom in symptom_keywords):
            exposure_context = "暴露当时或之后"
            if "潜水" in sentence:
                exposure_context = "潜水当时或之后"
            elif "游泳" in sentence:
                exposure_context = "游泳当时或之后"
            elif "飞机" in sentence or "飞行" in sentence:
                exposure_context = "飞行当时或之后"
            symptom = _negated_symptom_label(sentence)
            return f"{exposure_context}{symptom}"
    return None


def _negated_symptom_label(sentence: str) -> str:
    if "听力下降" in sentence:
        return "无听力下降"
    if "耳闷" in sentence:
        return "无耳闷"
    if "耳朵痛" in sentence or "耳痛" in sentence:
        return "无耳痛"
    if "不适" in sentence:
        return "无明显不适"
    if "疼" in sentence or "痛" in sentence:
        return "无疼痛"
    return "无相关症状"


def _extract_today_onset(text: str) -> str | None:
    match = re.search(r"(今天[^。！？!?；;\n\r]{0,20}?(?:痛|疼|刺痛|不适|听力下降|耳闷|耳鸣))", text)
    if match:
        return match.group(1).strip("，,。；; ")
    return None


def _sentence_containing_any(text: str, keywords: Iterable[str]) -> str | None:
    for sentence in re.split(r"[。！？!?；;\n\r]", text):
        if any(keyword in sentence for keyword in keywords):
            return sentence.strip()
    return None


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
