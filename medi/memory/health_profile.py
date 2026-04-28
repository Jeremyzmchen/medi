"""
HealthProfile — 用户健康档案（SQLite 持久化）

存储医疗事实（硬性数据），区别于向量库存储的行为习惯（软性数据）。
档案内容通过 UnifiedContext.build_constraint_prompt() 作为硬约束注入每次 LLM 调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VisitRecord:
    visit_date: datetime
    department: str
    chief_complaint: str
    conclusion: str


@dataclass
class HealthProfile:
    user_id: str
    age: int | None = None
    gender: str | None = None                          # "男" / "女"
    chronic_conditions: list[str] = field(default_factory=list)   # 慢性病史
    allergies: list[str] = field(default_factory=list)            # 过敏史
    current_medications: list[str] = field(default_factory=list)  # 当前用药
    visit_history: list[VisitRecord] = field(default_factory=list)

    # TODO: Phase 2 接入 SQLite 持久化
    # 目前为内存对象，Phase 1 用 mock 数据测试约束注入逻辑
