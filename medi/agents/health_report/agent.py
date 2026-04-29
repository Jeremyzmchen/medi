"""
HealthReportAgent — 体检报告解读 Agent（多 Agent Pipeline 入口）

Pipeline 流程：
  用户输入（体检报告文字）
      ↓
  HealthReportAgent._analyze()    # 解读异常指标，输出 ReportAnalysis
      ↓
  DietAgent.handle(analysis)      # 生成膳食方案，输出 DietPlan
      ↓
  ScheduleAgent.handle(diet, analysis)  # 生成日程安排，输出 SchedulePlan
      ↓
  汇总三份输出，emit RESULT 事件

Agent 间通过结构化数据（ReportAnalysis / DietPlan / SchedulePlan）传递，
避免自由文本传递导致的信息丢失和歧义。

免责：回答仅供参考，具体诊断和治疗请遵医嘱。
"""

from __future__ import annotations

import json

from medi.agents.health_report.diet_agent import DietAgent
from medi.agents.health_report.schedule_agent import ScheduleAgent
from medi.agents.health_report.schemas import (
    AbnormalIndicator,
    DietPlan,
    ReportAnalysis,
    SchedulePlan,
)
from medi.core.context import UnifiedContext
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent

_ANALYZE_SYSTEM_PROMPT = """你是一位经验丰富的全科医生，专注于体检报告解读。

根据用户提供的体检数据，识别所有异常指标，给出临床解读和就医建议。

输出要求（严格 JSON，不要输出其他内容）：
{
  "summary": "整体健康评估（2-3句，通俗易懂）",
  "abnormal_indicators": [
    {
      "name": "指标名称",
      "value": "实测值",
      "reference": "参考范围",
      "interpretation": "临床含义（通俗描述）",
      "severity": "low/medium/high"
    }
  ],
  "recommendations": [
    "建议1：如'建议4周后复查空腹血糖'",
    "建议2：如'建议内分泌科就诊'"
  ]
}

severity 说明：
- low：轻微偏离，生活方式调整即可
- medium：需要关注，建议就医复查
- high：显著异常，建议尽快就医"""


class HealthReportAgent:
    """
    体检报告多 Agent Pipeline 的编排入口。

    职责：
      1. 解读体检报告（自身完成）
      2. 编排 DietAgent 和 ScheduleAgent
      3. 汇总三份结果，推送给用户
    """

    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus
        self._diet_agent = DietAgent(ctx=ctx, bus=bus)
        self._schedule_agent = ScheduleAgent(ctx=ctx, bus=bus)

    async def handle(self, user_input: str) -> None:
        """
        Pipeline 入口：解读 → 膳食 → 日程 → 汇总输出。
        每个阶段完成后通过 RESULT 事件通知进度。
        """
        self._ctx.add_user_message(user_input)

        # ── 阶段一：体检解读 ──────────────────────────────
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": "正在解读体检报告，请稍候..."},
            session_id=self._ctx.session_id,
        ))

        analysis = await self._analyze(user_input)

        # ── 阶段二：膳食建议 ──────────────────────────────
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": "正在制定膳食方案..."},
            session_id=self._ctx.session_id,
        ))

        diet_plan = await self._diet_agent.handle(analysis)

        # ── 阶段三：日程安排 ──────────────────────────────
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": "正在规划健康日程..."},
            session_id=self._ctx.session_id,
        ))

        schedule_plan = await self._schedule_agent.handle(diet_plan, analysis)

        # ── 汇总输出 ──────────────────────────────────────
        final_content = self._format_final_report(analysis, diet_plan, schedule_plan)
        self._ctx.add_assistant_message(final_content)

        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": final_content},
            session_id=self._ctx.session_id,
        ))

    async def _analyze(self, user_input: str) -> ReportAnalysis:
        """调用 LLM 解读体检报告，输出结构化 ReportAnalysis"""
        constraint_prompt = self._ctx.build_constraint_prompt()
        system = _ANALYZE_SYSTEM_PROMPT
        if constraint_prompt:
            system += f"\n\n{constraint_prompt}"

        response = await call_with_fallback(
            chain=self._ctx.model_config.smart_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="report_analyze",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            indicators = [
                AbnormalIndicator(
                    name=ind.get("name", ""),
                    value=ind.get("value", ""),
                    reference=ind.get("reference", ""),
                    interpretation=ind.get("interpretation", ""),
                    severity=ind.get("severity", "low"),
                )
                for ind in data.get("abnormal_indicators", [])
            ]
            return ReportAnalysis(
                summary=data.get("summary", ""),
                abnormal_indicators=indicators,
                recommendations=data.get("recommendations", []),
            )
        except (json.JSONDecodeError, KeyError):
            return ReportAnalysis(summary=raw, abnormal_indicators=[], recommendations=[])

    def _format_final_report(
        self,
        analysis: ReportAnalysis,
        diet_plan: DietPlan,
        schedule_plan: SchedulePlan,
    ) -> str:
        """将三份结构化数据格式化为 Markdown 报告"""
        lines = ["## 体检报告综合健康方案\n"]

        # 一、体检解读
        lines.append("### 一、体检解读\n")
        lines.append(analysis.summary)
        if analysis.abnormal_indicators:
            lines.append("\n**异常指标：**\n")
            for ind in analysis.abnormal_indicators:
                severity_label = {"low": "⚠️ 轻微", "medium": "🔶 中度", "high": "🔴 严重"}.get(ind.severity, "")
                lines.append(
                    f"- **{ind.name}**：{ind.value}（参考 {ind.reference}）{severity_label}\n"
                    f"  {ind.interpretation}"
                )
        if analysis.recommendations:
            lines.append("\n**就医/复查建议：**\n")
            for r in analysis.recommendations:
                lines.append(f"- {r}")

        # 二、膳食方案
        lines.append("\n### 二、个性化膳食方案\n")
        lines.append(diet_plan.summary)
        if diet_plan.suggestions:
            lines.append("")
            for s in diet_plan.suggestions:
                items_str = "、".join(s.items)
                lines.append(f"**{s.category}**：{items_str}")
                lines.append(f"  原因：{s.reason}")
        if diet_plan.daily_schedule:
            lines.append("\n**每日饮食安排：**\n")
            for item in diet_plan.daily_schedule:
                lines.append(f"- {item}")

        # 三、健康日程
        lines.append("\n### 三、一周健康日程\n")
        lines.append(schedule_plan.summary)
        if schedule_plan.weekly_plan:
            lines.append("")
            for day in schedule_plan.weekly_plan:
                lines.append(f"- {day}")
        if schedule_plan.reminders:
            lines.append("\n**提醒事项：**\n")
            for reminder in schedule_plan.reminders:
                lines.append(f"- {reminder}")

        lines.append("\n---\n⚠️ 以上内容仅供参考，具体诊断和治疗请遵医嘱。")

        return "\n".join(lines)
