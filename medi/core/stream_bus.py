"""
AsyncStreamBus — 异步事件总线

ESCALATION 事件类型用于红旗症状升级通知。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class EventType(Enum):
    STAGE_START  = "stage_start"
    STAGE_END    = "stage_end"
    THINKING     = "thinking"
    OBSERVATION  = "observation"
    CONTENT      = "content"
    TOOL_CALL    = "tool_call"
    TOOL_RESULT  = "tool_result"
    FOLLOW_UP    = "follow_up"       # 追问用户
    ESCALATION   = "escalation"      # 红旗症状，立即升级（Medi 特有）
    RESULT       = "result"
    ERROR        = "error"
    DONE         = "done"


@dataclass
class StreamEvent:
    type: EventType
    data: dict = field(default_factory=dict)
    session_id: str = ""


class AsyncStreamBus:
    """
    Agent 通过 emit() 发布事件，CLI/API 通过 stream() 消费。
    支持多消费者互不干扰（每个消费者有独立队列）。
    """

    def __init__(self) -> None:
        # list: 不同消费者有自己独立的队列, 目前只有cli一个消费者
        self._queues: list[asyncio.Queue[StreamEvent | None]] = []

    def _make_queue(self) -> asyncio.Queue[StreamEvent | None]:
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._queues.append(q)
        return q

    async def emit(self, event: StreamEvent) -> None:
        for q in self._queues:
            await q.put(event)

    async def stream(self) -> AsyncIterator[StreamEvent]:
        # 消费时分配一个独立队列，同时注册到queues
        q = self._make_queue()
        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        finally:
            self._queues.remove(q)

    async def close(self) -> None:
        for q in self._queues:
            await q.put(None)
