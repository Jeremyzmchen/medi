"""
UrgencyEvaluator — 紧急程度评估器

核心设计：规则层前置，不依赖 LLM 判断红旗症状。
LLM 只处理规则层通过后的普通症状分级。

红旗症状（立即拨打 120，不等 LLM）：
- 胸痛 / 胸闷
- 呼吸困难 / 喘不过气
- 意识丧失 / 昏迷
- 突发剧烈头痛
- 口角歪斜 / 肢体无力（卒中征兆）
- 大量出血
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UrgencyLevel(Enum):
    EMERGENCY  = "emergency"    # 立即拨打 120
    URGENT     = "urgent"       # 尽快就医（当天）
    NORMAL     = "normal"       # 近期就医（3 天内）
    WATCHFUL   = "watchful"     # 可先观察


# 红旗关键词 -> 触发 EMERGENCY，规则层直接返回，不走 LLM
_RED_FLAG_KEYWORDS: list[str] = [
    "胸痛", "胸闷", "心痛",
    "呼吸困难", "喘不过气", "憋气", "呼吸急促",
    "昏迷", "失去意识", "晕倒", "意识不清",
    "突然头痛", "剧烈头痛", "雷击样头痛",
    "口角歪斜", "面瘫", "肢体无力", "半身不遂",
    "大量出血", "咳血", "呕血", "便血大量",
]


@dataclass
class UrgencyResult:
    level: UrgencyLevel
    reason: str
    triggered_by_rule: bool   # True = 规则层触发，False = LLM 判断


def evaluate_urgency_by_rules(symptom_text: str) -> UrgencyResult | None:
    """
    规则层前置扫描。
    命中红旗关键词 -> 返回 EMERGENCY。
    未命中 -> 返回 None，交由 LLM 处理。
    """
    for keyword in _RED_FLAG_KEYWORDS:
        if keyword in symptom_text:
            return UrgencyResult(
                level=UrgencyLevel.EMERGENCY,
                reason=f"检测到红旗症状关键词：「{keyword}」",
                triggered_by_rule=True,
            )
    return None


EMERGENCY_RESPONSE = (
    "您描述的症状可能存在紧急情况，请立即拨打 120 或前往最近医院急诊。\n"
    "在等待救援期间，请保持平卧、不要独自行动。"
)
