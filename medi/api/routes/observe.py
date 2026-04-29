"""
可观测性路由

GET /observe                    — 最近 10 个 session 摘要
GET /observe/{session_id}       — 单个 session 完整链路
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from medi.api.schemas import (
    LLMCallInfo,
    ObserveDetailResponse,
    ObserveListResponse,
    SessionInfo,
    StageInfo,
    ToolCallInfo,
)
from medi.core.observability import query_recent_sessions, query_session_detail

router = APIRouter(prefix="/observe", tags=["observe"])


@router.get("", response_model=ObserveListResponse)
async def list_sessions() -> ObserveListResponse:
    """返回最近 10 个会话的可观测性摘要"""
    rows = await query_recent_sessions(limit=10)
    sessions = [
        SessionInfo(
            session_id=r["session_id"],
            start_time=r["start_time"][:16],
            llm_calls=r["llm_calls"],
            total_tokens=r["total_tokens"] or 0,
            total_llm_ms=r["total_llm_ms"] or 0,
            fallback_count=r["fallback_count"] or 0,
            error_count=r["error_count"] or 0,
        )
        for r in rows
    ]
    return ObserveListResponse(sessions=sessions)


@router.get("/{session_id}", response_model=ObserveDetailResponse)
async def session_detail(session_id: str) -> ObserveDetailResponse:
    """返回单个 session 的完整链路详情（LLM 调用 + 工具调用 + 阶段耗时）"""
    detail = await query_session_detail(session_id)

    if not detail["stages"] and not detail["llm_calls"] and not detail["tool_calls"]:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' 暂无可观测性数据")

    return ObserveDetailResponse(
        session_id=session_id,
        stages=[StageInfo(**s) for s in detail["stages"]],
        llm_calls=[
            LLMCallInfo(
                call_type=c.get("call_type", "unknown"),
                provider=c["provider"],
                is_fallback=c["is_fallback"],
                prompt_tokens=c["prompt_tokens"],
                completion_tokens=c["completion_tokens"],
                latency_ms=c["latency_ms"],
                success=c["success"],
            )
            for c in detail["llm_calls"]
        ],
        tool_calls=[
            ToolCallInfo(
                tool_name=t["tool_name"],
                latency_ms=t["latency_ms"],
                success=t["success"],
                error_msg=t["error_msg"],
            )
            for t in detail["tool_calls"]
        ],
    )
