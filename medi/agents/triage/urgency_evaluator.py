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
# 在act -> observe 阶段间还有一层llm根据收集的信息判断紧急情况
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


async def evaluate_urgency_by_llm(
    symptom_text: str,
    call_with_fallback,
    fast_chain,
    bus,
    session_id: str,
    obs=None,
) -> UrgencyResult:
    """
    LLM 层紧急程度评估，规则层未命中时调用。
    只判断 urgent / normal / watchful 三级（emergency 已由规则层拦截）。
    """
    response = await call_with_fallback(
        chain=fast_chain,
        bus=bus,
        session_id=session_id,
        obs=obs,
        call_type="evaluate_urgency",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位急诊分诊护士。根据患者症状判断就医紧急程度，"
                    "只输出以下三个等级之一，不要输出其他内容：\n"
                    "urgent（尽快就医，当天内）\n"
                    "normal（近期就医，3天内）\n"
                    "watchful（可先观察，症状加重再就医）"
                ),
            },
            {"role": "user", "content": symptom_text},
        ],
        max_tokens=10,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip().lower()

    if "urgent" in raw:
        level = UrgencyLevel.URGENT
        reason = "智能分诊护士评估为较急，建议当天就医"
    elif "watchful" in raw:
        level = UrgencyLevel.WATCHFUL
        reason = "智能分诊护士评估可先观察，症状加重再就医"
    else:
        level = UrgencyLevel.NORMAL
        reason = "智能分诊护士评估为普通，建议近期就医"

    return UrgencyResult(level=level, reason=reason, triggered_by_rule=False)


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

# TODO: 后续增加其他呼救报警模块
EMERGENCY_RESPONSE = (
    "已为您触发急诊优先处理流程，正在为您连接急诊科室挂号。\n"
    "请按照后续挂号提示或医院急诊流程立即就医；如果症状正在加重、出现呼吸困难、胸痛加重或意识不清，请立即拨打 120。"
)
