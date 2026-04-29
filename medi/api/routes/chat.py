"""
聊天路由

POST /chat          — 单轮对话，收集所有事件后返回完整 JSON 列表
GET  /chat/stream   — SSE 流式推送，每个 StreamEvent 一条 data: 行

SSE 格式（每条事件）：
  data: {"event_type": "follow_up", "content": "...", "session_id": "..."}

前端消费示例（EventSource）：
  const es = new EventSource('/chat/stream?user_id=u1&session_id=abc&message=...')
  es.onmessage = e => console.log(JSON.parse(e.data))
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from medi.agents.orchestrator import Intent
from medi.agents.triage.symptom_collector import SymptomInfo
from medi.api.schemas import ChatRequest, ChatResponse
from medi.api.session_store import get_or_create_session, rebind_bus
from medi.core.stream_bus import AsyncStreamBus, EventType

router = APIRouter(prefix="/chat", tags=["chat"])


def _event_to_dict(event_type: str, content: str, session_id: str, metadata: dict | None = None) -> dict:
    return {
        "event_type": event_type,
        "content": content,
        "session_id": session_id,
        "metadata": metadata or {},
    }


async def _run_turn(session_id: str | None, user_id: str, message: str) -> tuple[str, list[dict]]:
    """
    执行一轮对话，返回 (session_id, events_list)。
    events_list 是所有收集到的事件，每个是 dict。
    """
    session = await get_or_create_session(session_id, user_id)
    bus = AsyncStreamBus()
    rebind_bus(session, bus)

    events: list[dict] = []

    async def consume() -> None:
        async for event in bus.stream():
            if event.type == EventType.FOLLOW_UP:
                events.append(_event_to_dict(
                    "follow_up",
                    event.data.get("question", ""),
                    session.session_id,
                ))
            elif event.type == EventType.RESULT:
                events.append(_event_to_dict(
                    "result",
                    event.data.get("content", ""),
                    session.session_id,
                ))
            elif event.type == EventType.ESCALATION:
                events.append(_event_to_dict(
                    "escalation",
                    event.data.get("reason", ""),
                    session.session_id,
                ))
            elif event.type == EventType.ERROR:
                events.append(_event_to_dict(
                    "error",
                    event.data.get("message", "未知错误"),
                    session.session_id,
                ))

    async def produce() -> None:
        ctx = session.ctx
        agent = session.agent
        orchestrator = session.orchestrator
        medication_agent = session.medication_agent

        symptom_summary = agent._symptom_info.to_summary()
        intent = await orchestrator.classify_intent(message, symptom_summary)

        if intent == Intent.OUT_OF_SCOPE:
            await orchestrator.handle_out_of_scope()
        elif intent == Intent.FOLLOWUP:
            await orchestrator.handle_followup(message)
        elif intent == Intent.NEW_SYMPTOM:
            ctx.messages.clear()
            agent._symptom_info = SymptomInfo()
            await agent.handle(message)
        elif intent == Intent.MEDICATION:
            await medication_agent.handle(message)
        elif intent == Intent.HEALTH_REPORT:
            await session.health_report_agent.handle(message)
        else:
            await agent.handle(message)

        await bus.close()

    await asyncio.gather(consume(), produce())
    await session.obs.flush()

    return session.session_id, events


@router.post("", response_model=list[ChatResponse])
async def chat(req: ChatRequest) -> list[ChatResponse]:
    """
    单轮对话接口。

    - 首轮不传 session_id，服务端生成并在响应中返回
    - 后续轮次传入 session_id 继续同一会话
    - 返回该轮所有事件列表（通常 1 条，追问时也是 1 条 follow_up）
    """
    try:
        sid, events = await _run_turn(req.session_id, req.user_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not events:
        # 兜底：不应发生，Agent 至少会给 result
        events = [_event_to_dict("result", "（无响应）", req.session_id or "")]

    return [ChatResponse(**e) for e in events]


@router.get("/stream")
async def chat_stream(
    message: str = Query(..., description="用户输入"),
    user_id: str = Query(default="guest"),
    session_id: str | None = Query(default=None),
) -> StreamingResponse:
    """
    SSE 流式对话接口。

    每个 StreamEvent 对应一条 `data: {...}\\n\\n`。
    最后发送 `data: {"event_type": "done", "content": "", ...}` 表示结束。

    前端用 EventSource 消费：
      const es = new EventSource(`/chat/stream?message=...&user_id=...&session_id=...`)
    """

    async def event_generator() -> AsyncIterator[str]:
        try:
            session = await get_or_create_session(session_id, user_id)
            bus = AsyncStreamBus()
            rebind_bus(session, bus)

            async def produce() -> None:
                ctx = session.ctx
                agent = session.agent
                orchestrator = session.orchestrator
                medication_agent = session.medication_agent

                symptom_summary = agent._symptom_info.to_summary()
                intent = await orchestrator.classify_intent(message, symptom_summary)

                if intent == Intent.OUT_OF_SCOPE:
                    await orchestrator.handle_out_of_scope()
                elif intent == Intent.FOLLOWUP:
                    await orchestrator.handle_followup(message)
                elif intent == Intent.NEW_SYMPTOM:
                    ctx.messages.clear()
                    agent._symptom_info = SymptomInfo()
                    await agent.handle(message)
                elif intent == Intent.MEDICATION:
                    await medication_agent.handle(message)
                elif intent == Intent.HEALTH_REPORT:
                    await session.health_report_agent.handle(message)
                else:
                    await agent.handle(message)

                await bus.close()

            produce_task = asyncio.create_task(produce())

            async for event in bus.stream():
                if event.type == EventType.FOLLOW_UP:
                    payload = _event_to_dict("follow_up", event.data.get("question", ""), session.session_id)
                elif event.type == EventType.RESULT:
                    payload = _event_to_dict("result", event.data.get("content", ""), session.session_id)
                elif event.type == EventType.ESCALATION:
                    payload = _event_to_dict("escalation", event.data.get("reason", ""), session.session_id)
                elif event.type == EventType.ERROR:
                    payload = _event_to_dict("error", event.data.get("message", ""), session.session_id)
                else:
                    continue

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            await produce_task
            await session.obs.flush()

            # 结束标记
            done_payload = _event_to_dict("done", "", session.session_id)
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            err_payload = {"event_type": "error", "content": str(e), "session_id": session_id or "", "metadata": {}}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，确保实时推送
        },
    )
