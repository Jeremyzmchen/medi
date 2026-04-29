"""
体检报告多 Agent Pipeline 的中间数据结构

Agent 间通过结构化数据传递，而非自由文本：
  HealthReportAgent → ReportAnalysis → DietAgent → DietPlan → ScheduleAgent
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AbnormalIndicator:
    """单个异常指标"""
    name: str           # 指标名称，如"空腹血糖"
    value: str          # 实测值，如"7.2 mmol/L"
    reference: str      # 参考范围，如"3.9-6.1 mmol/L"
    interpretation: str # 临床含义，如"提示糖尿病前期"
    severity: str       # low / medium / high


@dataclass
class ReportAnalysis:
    """HealthReportAgent 输出 → 下游 Agent 的输入"""
    summary: str                                        # 整体评估文字（给用户看）
    abnormal_indicators: list[AbnormalIndicator] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)  # 就医/复查建议


@dataclass
class DietSuggestion:
    """单条膳食建议"""
    category: str   # 宜食 / 忌食 / 限量
    items: list[str]
    reason: str


@dataclass
class DietPlan:
    """DietAgent 输出 → ScheduleAgent 的输入"""
    summary: str
    suggestions: list[DietSuggestion] = field(default_factory=list)
    daily_schedule: list[str] = field(default_factory=list)  # 每日饮食安排要点


@dataclass
class SchedulePlan:
    """ScheduleAgent 最终输出"""
    summary: str
    weekly_plan: list[str] = field(default_factory=list)   # 按天排列的日程
    reminders: list[str] = field(default_factory=list)     # 提醒事项
