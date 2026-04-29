"""
ScheduleAgent — 健康日程规划 Agent

输入：DietPlan（来自 DietAgent）+ ReportAnalysis（来自 HealthReportAgent）
输出：SchedulePlan，包含一周日程安排和提醒事项

设计要点：
- 整合膳食建议 + 运动建议 + 复查提醒，生成可执行的周计划
- 输出结构化 SchedulePlan，最终由 HealthReportAgent 汇总后推给用户
"""

from __future__ import annotations

import json

from medi.agents.health_report.schemas import DietPlan, ReportAnalysis, SchedulePlan
from medi.core.context import UnifiedContext
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus

_SCHEDULE_SYSTEM_PROMPT = """你是一位健康管理顾问，根据用户的膳食方案和体检建议，制定一周健康日程。

输出要求（严格 JSON，不要输出其他内容）：
{
  "summary": "日程方案总结（1-2句）",
  "weekly_plan": [
    "周一：早餐 ... | 运动：... | 注意事项：...",
    "周二：...",
    "周三：...",
    "周四：...",
    "周五：...",
    "周六：...",
    "周日：..."
  ],
  "reminders": [
    "每天早晨空腹测血糖",
    "4周后复查空腹血糖和糖化血红蛋白"
  ]
}"""


class ScheduleAgent:
    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus

    async def handle(self, diet_plan: DietPlan, analysis: ReportAnalysis) -> SchedulePlan:
        """
        根据膳食方案和体检解读生成一周健康日程。

        :param diet_plan: DietAgent 输出
        :param analysis: HealthReportAgent 输出（用于提取复查建议）
        :return: SchedulePlan
        """
        # 格式化膳食摘要
        diet_text = diet_plan.summary
        if diet_plan.daily_schedule:
            diet_text += "\n\n[每日饮食要点]\n" + "\n".join(diet_plan.daily_schedule)
        if diet_plan.suggestions:
            diet_text += "\n\n[膳食建议]\n"
            for s in diet_plan.suggestions:
                items_str = "、".join(s.items)
                diet_text += f"- {s.category}：{items_str}（{s.reason}）\n"

        # 复查建议
        followup_text = "\n".join(f"- {r}" for r in analysis.recommendations) or "无特殊复查要求"

        response = await call_with_fallback(
            chain=self._ctx.model_config.smart_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="schedule_plan",
            messages=[
                {"role": "system", "content": _SCHEDULE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"[膳食方案]\n{diet_text}\n\n"
                        f"[就医/复查建议]\n{followup_text}"
                    ),
                },
            ],
            max_tokens=1000,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()

        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return SchedulePlan(
                summary=data.get("summary", ""),
                weekly_plan=data.get("weekly_plan", []),
                reminders=data.get("reminders", []),
            )
        except (json.JSONDecodeError, KeyError):
            return SchedulePlan(summary=raw, weekly_plan=[], reminders=[])
