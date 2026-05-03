"""
EpisodicMemory — 分诊记录的读写封装

职责：
  - 保存每次分诊结果（症状摘要 + 建议科室 + 紧急程度）
  - 查询最近 N 条记录，供下次会话参考
  - 构建历史摘要 prompt 片段，注入 system prompt（软参考，非硬约束）

与 HealthProfile 的区别：
  HealthProfile  — 用户的静态医疗事实（过敏史、慢性病），硬约束
  EpisodicMemory — 历次就诊的动态记录（什么时候、什么症状、建议哪里），软参考
"""

from __future__ import annotations

from datetime import datetime

from medi.memory.health_profile import DB_PATH, VisitRecord, _ensure_tables, add_visit_record

import aiosqlite


class EpisodicMemory:
    def __init__(self, user_id: str) -> None:
        self._user_id = user_id

    async def save(self, symptom_summary: str, advice: str, department: str = "待确认") -> None:
        """
        保存一次分诊记录。
        symptom_summary — OPQRST 摘要（来自 SymptomInfo.to_summary()）
        advice          — triage flow 给出的完整建议文本
        department      — 置信度最高的科室（来自 candidates[0].department）
        """
        record = VisitRecord(
            visit_date=datetime.now(),
            department=department,
            chief_complaint=symptom_summary,
            conclusion=advice[:500],  # 截断避免存太长
        )
        await add_visit_record(self._user_id, record)

    async def recent(self, limit: int = 5) -> list[VisitRecord]:
        """查询最近 N 条分诊记录"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_tables(db)
            async with db.execute(
                "SELECT visit_date, department, chief_complaint, conclusion "
                "FROM visit_records WHERE user_id = ? "
                "ORDER BY visit_date DESC LIMIT ?",
                (self._user_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            VisitRecord(
                visit_date=datetime.fromisoformat(r[0]),
                department=r[1],
                chief_complaint=r[2],
                conclusion=r[3],
            )
            for r in rows
        ]

    async def build_history_prompt(self, limit: int = 3) -> str:
        """
        构建历史就诊摘要 prompt 片段，供 system prompt 软参考。
        只取最近 3 条，避免 prompt 过长。
        """
        records = await self.recent(limit)
        if not records:
            return ""

        lines = ["[历史就诊记录（供参考，非硬约束）]"]
        for r in records:
            date_str = r.visit_date.strftime("%Y-%m-%d")
            lines.append(f"- {date_str} | {r.department} | {r.chief_complaint[:50]}")

        return "\n".join(lines)

