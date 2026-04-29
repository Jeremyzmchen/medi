"""
ObservabilityStore — 可观测性数据采集与存储

记录每次请求的完整链路：
  - LLM 调用（供应商、模型、token、耗时、是否降级）
  - 工具调用（工具名、耗时、成功/失败）
  - 阶段耗时（think / act / observe / respond）

数据持久化到 SQLite，通过 `medi observe` CLI 命令查询。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent.parent / "data" / "observability.db"


@dataclass
class LLMTrace:
    """单次 LLM 调用记录"""
    session_id: str
    timestamp: datetime
    provider: str           # e.g. "openai/gpt-4o"
    call_type: str          # intent_classify / enrich_region / follow_up / act_search / respond / medication / followup_answer / decompose
    is_fallback: bool       # 是否是降级后的调用
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    success: bool
    error_type: str | None = None   # RateLimitError / APITimeoutError 等


@dataclass
class ToolTrace:
    """单次工具调用记录"""
    session_id: str
    timestamp: datetime
    tool_name: str
    latency_ms: int
    success: bool
    error_msg: str | None = None


@dataclass
class StageTrace:
    """TAOR 单阶段耗时记录"""
    session_id: str
    timestamp: datetime
    stage: str      # think / act / observe / respond
    latency_ms: int


class ObservabilityStore:
    """
    可观测性数据存储。
    每个 session 对应一组 trace 记录，写入 SQLite。
    """

    def __init__(self) -> None:
        self._llm_traces: list[LLMTrace] = []
        self._tool_traces: list[ToolTrace] = []
        self._stage_traces: list[StageTrace] = []

    # ---------- 采集接口 ----------

    def record_llm(self, trace: LLMTrace) -> None:
        self._llm_traces.append(trace)

    def record_tool(self, trace: ToolTrace) -> None:
        self._tool_traces.append(trace)

    def record_stage(self, trace: StageTrace) -> None:
        self._stage_traces.append(trace)

    # ---------- 持久化 ----------

    async def flush(self) -> None:
        """将内存中的 trace 写入 SQLite，清空缓冲"""
        if not self._llm_traces and not self._tool_traces and not self._stage_traces:
            return

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_tables(db)
            await _insert_llm_traces(db, self._llm_traces)
            await _insert_tool_traces(db, self._tool_traces)
            await _insert_stage_traces(db, self._stage_traces)
            await db.commit()

        self._llm_traces.clear()
        self._tool_traces.clear()
        self._stage_traces.clear()


# ---------- 查询接口（供 CLI 使用）----------

async def query_recent_sessions(limit: int = 10) -> list[dict]:
    """查询最近 N 个 session 的汇总信息"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT
                session_id,
                MIN(timestamp) as start_time,
                COUNT(*) as llm_calls,
                SUM(prompt_tokens + completion_tokens) as total_tokens,
                SUM(latency_ms) as total_llm_ms,
                SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END) as fallback_count,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as error_count
            FROM llm_traces
            GROUP BY session_id
            ORDER BY start_time DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def query_session_detail(session_id: str) -> dict:
    """查询单个 session 的完整链路"""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM llm_traces WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        llm = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT * FROM tool_traces WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        tools = [dict(r) for r in await cursor.fetchall()]

        cursor = await db.execute(
            "SELECT * FROM stage_traces WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        stages = [dict(r) for r in await cursor.fetchall()]

    return {"llm_calls": llm, "tool_calls": tools, "stages": stages}


# ---------- 内部 SQL 辅助 ----------

async def _ensure_tables(db: aiosqlite.Connection) -> None:
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS llm_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            provider TEXT NOT NULL,
            call_type TEXT NOT NULL DEFAULT '',
            is_fallback INTEGER NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            success INTEGER NOT NULL,
            error_type TEXT
        );

        CREATE TABLE IF NOT EXISTS tool_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            success INTEGER NOT NULL,
            error_msg TEXT
        );

        CREATE TABLE IF NOT EXISTS stage_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            stage TEXT NOT NULL,
            latency_ms INTEGER NOT NULL
        );
    """)


async def _insert_llm_traces(db: aiosqlite.Connection, traces: list[LLMTrace]) -> None:
    await db.executemany("""
        INSERT INTO llm_traces
            (session_id, timestamp, provider, call_type, is_fallback, prompt_tokens,
             completion_tokens, latency_ms, success, error_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            t.session_id,
            t.timestamp.isoformat(),
            t.provider,
            t.call_type,
            int(t.is_fallback),
            t.prompt_tokens,
            t.completion_tokens,
            t.latency_ms,
            int(t.success),
            t.error_type,
        )
        for t in traces
    ])


async def _insert_tool_traces(db: aiosqlite.Connection, traces: list[ToolTrace]) -> None:
    await db.executemany("""
        INSERT INTO tool_traces (session_id, timestamp, tool_name, latency_ms, success, error_msg)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (t.session_id, t.timestamp.isoformat(), t.tool_name, t.latency_ms, int(t.success), t.error_msg)
        for t in traces
    ])


async def _insert_stage_traces(db: aiosqlite.Connection, traces: list[StageTrace]) -> None:
    await db.executemany("""
        INSERT INTO stage_traces (session_id, timestamp, stage, latency_ms)
        VALUES (?, ?, ?, ?)
    """, [
        (t.session_id, t.timestamp.isoformat(), t.stage, t.latency_ms)
        for t in traces
    ])
