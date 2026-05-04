"""
FastAPI 请求/响应 Schema

设计原则：
- 请求简洁，session_id 由服务端管理
- 流式响应用 SSE，每条事件对应一个 StreamEvent
- 错误统一用 HTTPException + ErrorResponse
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────
# 请求体
# ──────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = Field(default="guest", description="用户 ID，决定加载哪个健康档案")
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID。首轮不传（服务端生成），后续轮次传入以继续对话",
    )
    message: str = Field(..., min_length=1, description="用户输入")


# ──────────────────────────────────────────
# 响应体
# ──────────────────────────────────────────

class ChatResponse(BaseModel):
    session_id: str
    event_type: str        # follow_up / result / escalation / error
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResumeStateResponse(BaseModel):
    status: str
    session_id: str
    pending_question: Optional[str] = None
    pending_task: Optional[str] = None
    why_needed: Optional[str] = None
    collected_summary: str = ""
    missing_summary: list[str] = Field(default_factory=list)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    last_updated_at: Optional[str] = None
    expires_at: Optional[str] = None
    recommended_action: str = "start"
    actions: list[str] = Field(default_factory=list)


class SessionActionResponse(BaseModel):
    session_id: str
    status: str
    message: str


class SessionInfo(BaseModel):
    session_id: str
    start_time: str
    llm_calls: int
    total_tokens: int
    total_llm_ms: int
    fallback_count: int
    error_count: int


class ObserveListResponse(BaseModel):
    sessions: list[SessionInfo]


class LLMCallInfo(BaseModel):
    call_type: str
    provider: str
    is_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    success: bool


class ToolCallInfo(BaseModel):
    tool_name: str
    latency_ms: int
    success: bool
    error_msg: Optional[str] = None


class StageInfo(BaseModel):
    stage: str
    latency_ms: int


class ObserveDetailResponse(BaseModel):
    session_id: str
    stages: list[StageInfo]
    llm_calls: list[LLMCallInfo]
    tool_calls: list[ToolCallInfo]


class ErrorResponse(BaseModel):
    detail: str
