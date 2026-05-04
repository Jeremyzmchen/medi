"""
DietAgent — 膳食建议 Agent

输入：ReportAnalysis（来自 HealthReportAgent）
输出：DietPlan（传给 ScheduleAgent）

设计要点：
- 接收结构化的异常指标，而非自由文本，避免信息丢失
- 约束：同时注入 ProfileSnapshot（过敏史、慢性病、当前用药）
- 输出也是结构化 DietPlan，供 ScheduleAgent 直接使用
"""

from __future__ import annotations

import json

from medi.agents.health_report.schemas import AbnormalIndicator, DietPlan, ReportAnalysis
from medi.core.context import UnifiedContext
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus

_DIET_SYSTEM_PROMPT = """你是一位专业的临床营养师，根据用户的体检异常指标和健康档案，制定个性化膳食建议。

输出要求（严格 JSON，不要输出其他内容）：
{
  "summary": "整体膳食方案说明（1-2句）",
  "suggestions": [
    {
      "category": "宜食/忌食/限量",
      "items": ["食物1", "食物2"],
      "reason": "针对哪项指标、为什么"
    }
  ],
  "daily_schedule": [
    "早餐：...",
    "午餐：...",
    "晚餐：...",
    "加餐：..."
  ]
}"""


def _format_indicators(indicators: list[AbnormalIndicator]) -> str:
    lines = []
    for ind in indicators:
        lines.append(
            f"- {ind.name}：{ind.value}（参考 {ind.reference}）→ {ind.interpretation}，严重程度：{ind.severity}"
        )
    return "\n".join(lines) if lines else "无明显异常"


class DietAgent:
    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus

    async def handle(self, analysis: ReportAnalysis) -> DietPlan:
        """
        根据体检解读结果生成膳食方案。

        :param analysis: HealthReportAgent 输出的结构化解读
        :return: DietPlan，供 ScheduleAgent 使用
        """
        indicators_text = _format_indicators(analysis.abnormal_indicators)
        constraint_prompt = self._ctx.build_constraint_prompt()

        system = _DIET_SYSTEM_PROMPT
        if constraint_prompt:
            system += f"\n\n{constraint_prompt}"

        user_content = (
            f"[体检异常指标]\n{indicators_text}\n\n"
            f"[就医建议]\n" + "\n".join(f"- {r}" for r in analysis.recommendations)
        )

        response = await call_with_fallback(
            chain=self._ctx.model_config.smart_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="diet_plan",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=800,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()

        try:
            # 去掉可能的 markdown 代码块
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            from medi.agents.health_report.schemas import DietSuggestion
            suggestions = [
                DietSuggestion(
                    category=s.get("category", ""),
                    items=s.get("items", []),
                    reason=s.get("reason", ""),
                )
                for s in data.get("suggestions", [])
            ]
            return DietPlan(
                summary=data.get("summary", ""),
                suggestions=suggestions,
                daily_schedule=data.get("daily_schedule", []),
            )
        except (json.JSONDecodeError, KeyError):
            # JSON 解析失败：降级为纯文本摘要
            return DietPlan(summary=raw, suggestions=[], daily_schedule=[])
