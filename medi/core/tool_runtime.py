"""
ToolRuntime — 统一工具执行层

与 Weave 的 ToolRuntime 相比，Medi 新增：
1. 工具优先级分级（CRITICAL / STANDARD / OPTIONAL）
2. 审计日志（CRITICAL 工具调用强制记录，合规要求）
3. 超时分级
4. 降级策略（OPTIONAL 工具失败静默跳过）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from medi.core.context import UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent

logger = logging.getLogger(__name__)


class ToolPriority(Enum):
    CRITICAL = "critical"   # 失败必须告知用户，超时 10s
    STANDARD = "standard"   # 可重试 3 次，超时 5s
    OPTIONAL = "optional"   # 失败静默跳过，超时 3s


TIMEOUT_BY_PRIORITY = {
    ToolPriority.CRITICAL: 10.0,
    ToolPriority.STANDARD: 5.0,
    ToolPriority.OPTIONAL: 3.0,
}

MAX_RETRY_BY_PRIORITY = {
    ToolPriority.CRITICAL: 1,   # 不重试，失败立即告知
    ToolPriority.STANDARD: 3,
    ToolPriority.OPTIONAL: 1,
}


@dataclass
class AuditRecord:
    """审计记录 — CRITICAL 工具调用的完整溯源信息"""
    session_id: str
    timestamp: datetime
    tool_name: str
    priority: ToolPriority
    input_params: dict
    output_result: dict | None
    latency_ms: int
    success: bool
    error_msg: str | None = None


@dataclass
class ToolDefinition:
    name: str
    priority: ToolPriority
    fn: Callable[..., Awaitable[Any]]
    description: str = ""


class ToolRuntime:
    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus
        self._tools: dict[str, ToolDefinition] = {}
        self._audit_log: list[AuditRecord] = []   # 内存暂存，后续持久化到 SQLite

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    async def call(self, tool_name: str, **kwargs: Any) -> Any:
        if not self._ctx.has_tool(tool_name):
            raise PermissionError(f"Tool '{tool_name}' not enabled for this session")

        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not registered")

        await self._bus.emit(StreamEvent(
            type=EventType.TOOL_CALL,
            data={"tool": tool_name, "params": kwargs},
            session_id=self._ctx.session_id,
        ))

        timeout = TIMEOUT_BY_PRIORITY[tool.priority]
        max_retry = MAX_RETRY_BY_PRIORITY[tool.priority]
        start = datetime.now()
        last_error: Exception | None = None

        for attempt in range(max_retry):
            try:
                result = await asyncio.wait_for(tool.fn(**kwargs), timeout=timeout)
                latency = int((datetime.now() - start).total_seconds() * 1000)

                await self._bus.emit(StreamEvent(
                    type=EventType.TOOL_RESULT,
                    data={"tool": tool_name, "result": result},
                    session_id=self._ctx.session_id,
                ))

                if tool.priority == ToolPriority.CRITICAL:
                    self._audit_log.append(AuditRecord(
                        session_id=self._ctx.session_id,
                        timestamp=start,
                        tool_name=tool_name,
                        priority=tool.priority,
                        input_params=kwargs,
                        output_result=result,
                        latency_ms=latency,
                        success=True,
                    ))

                # 可观测性 trace
                if self._ctx.observability is not None:
                    from medi.core.observability import ToolTrace
                    self._ctx.observability.record_tool(ToolTrace(
                        session_id=self._ctx.session_id,
                        timestamp=start,
                        tool_name=tool_name,
                        latency_ms=latency,
                        success=True,
                    ))

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Tool '{tool_name}' attempt {attempt + 1} failed: {e}")

        # 所有重试耗尽
        latency = int((datetime.now() - start).total_seconds() * 1000)

        # 可观测性 trace（失败）
        if self._ctx.observability is not None:
            from medi.core.observability import ToolTrace
            self._ctx.observability.record_tool(ToolTrace(
                session_id=self._ctx.session_id,
                timestamp=start,
                tool_name=tool_name,
                latency_ms=latency,
                success=False,
                error_msg=str(last_error),
            ))

        if tool.priority == ToolPriority.CRITICAL:
            self._audit_log.append(AuditRecord(
                session_id=self._ctx.session_id,
                timestamp=start,
                tool_name=tool_name,
                priority=tool.priority,
                input_params=kwargs,
                output_result=None,
                latency_ms=latency,
                success=False,
                error_msg=str(last_error),
            ))
            await self._bus.emit(StreamEvent(
                type=EventType.ERROR,
                data={"tool": tool_name, "error": str(last_error), "fatal": True},
                session_id=self._ctx.session_id,
            ))
            raise RuntimeError(f"Critical tool '{tool_name}' failed: {last_error}")

        if tool.priority == ToolPriority.OPTIONAL:
            logger.info(f"Optional tool '{tool_name}' failed, skipping silently")
            return None

        raise RuntimeError(f"Tool '{tool_name}' failed after {max_retry} attempts: {last_error}")

    @property
    def audit_log(self) -> list[AuditRecord]:
        return list(self._audit_log)
