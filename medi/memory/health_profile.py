"""
HealthProfile — 用户健康档案（SQLite 持久化）

存储医疗事实（硬性数据），区别于向量库存储的行为习惯（软性数据）。
档案内容通过 UnifiedContext.build_constraint_prompt() 作为硬约束注入每次 LLM 调用。

表结构：
  profiles       — 用户基本信息（一人一行）
  visit_records  — 就诊记录（一人多行）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiosqlite

# C:\...\health\medi\medi\memory\health_profile.py
DB_PATH = Path(__file__).parents[3] / "data" / "medi.db"


@dataclass
class VisitRecord:
    """问诊记录"""
    visit_date: datetime       # 问诊时间
    department: str            # 部门
    chief_complaint: str       # 主诉
    conclusion: str            # 结论


@dataclass
class HealthProfile:
    user_id: str                                                    # 用户id
    age: int | None = None                                          # 年龄
    gender: str | None = None                                       # 性别
    chronic_conditions: list[str] = field(default_factory=list)     # 慢性病史
    allergies: list[str] = field(default_factory=list)              # 过敏史
    current_medications: list[str] = field(default_factory=list)    # 当前用药
    visit_history: list[VisitRecord] = field(default_factory=list)  # 分诊记录

    def is_complete(self) -> bool:
        """判断基本信息是否已填写（年龄和性别是最低要求）"""
        return self.age is not None and self.gender is not None


async def _ensure_tables(db: aiosqlite.Connection) -> None:
    """初始化表格，确保数据库表已存在"""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id    TEXT PRIMARY KEY,
            age        INTEGER,
            gender     TEXT,
            chronic_conditions  TEXT DEFAULT '[]',
            allergies           TEXT DEFAULT '[]',
            current_medications TEXT DEFAULT '[]',
            updated_at TEXT
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS visit_records (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        TEXT NOT NULL,
            visit_date     TEXT NOT NULL,
            department     TEXT NOT NULL,
            chief_complaint TEXT NOT NULL,
            conclusion     TEXT NOT NULL
        )
    """)
    await db.commit()


async def load_profile(user_id: str) -> HealthProfile:
    """从 SQLite 加载用户档案，不存在时返回空档案"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        
        # 根据userId查询用户档案 
        async with db.execute(
            "SELECT age, gender, chronic_conditions, allergies, current_medications "
            "FROM profiles WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return HealthProfile(user_id=user_id)
        # 解包
        age, gender, chronic_json, allergy_json, meds_json = row
        # 建档
        profile = HealthProfile(
            user_id=user_id,
            age=age,
            gender=gender,
            chronic_conditions=json.loads(chronic_json or "[]"),
            allergies=json.loads(allergy_json or "[]"),
            current_medications=json.loads(meds_json or "[]"),
        )

        # 加载就诊记录
        async with db.execute(
            "SELECT visit_date, department, chief_complaint, conclusion "
            "FROM visit_records WHERE user_id = ? ORDER BY visit_date DESC LIMIT 10",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        profile.visit_history = [
            VisitRecord(
                visit_date=datetime.fromisoformat(visit_date),
                department=department,
                chief_complaint=chief_complaint,
                conclusion=conclusion,
            )
            for visit_date, department, chief_complaint, conclusion in rows
        ]

        return profile


async def save_profile(profile: HealthProfile) -> None:
    """保存或更新用户档案"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute("""
            INSERT INTO profiles
                (user_id, age, gender, chronic_conditions, allergies, current_medications, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age=excluded.age,
                gender=excluded.gender,
                chronic_conditions=excluded.chronic_conditions,
                allergies=excluded.allergies,
                current_medications=excluded.current_medications,
                updated_at=excluded.updated_at
        """, (
            profile.user_id,
            profile.age,
            profile.gender,
            json.dumps(profile.chronic_conditions, ensure_ascii=False),
            json.dumps(profile.allergies, ensure_ascii=False),
            json.dumps(profile.current_medications, ensure_ascii=False),
            datetime.now().isoformat(),
        ))
        await db.commit()


async def add_visit_record(user_id: str, record: VisitRecord) -> None:
    """追加一条就诊记录"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        # 问诊记录写入数据库
        await db.execute("""
            INSERT INTO visit_records (user_id, visit_date, department, chief_complaint, conclusion)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            record.visit_date.isoformat(),
            record.department,
            record.chief_complaint,
            record.conclusion,
        ))
        await db.commit()
