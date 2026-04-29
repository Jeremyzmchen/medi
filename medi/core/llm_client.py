"""
LLM 调用封装 — 跨供应商降级策略

按供应商链依次尝试，捕获以下错误时自动切换下一个供应商：
- RateLimitError（429 限流）
- APIStatusError（500/503 服务不可用）
- APITimeoutError（超时）

调用方完全透明，不感知当前使用的是哪个供应商。

使用方式：
    response = await call_with_fallback(
        chain=self._ctx.model_config.smart_chain,
        bus=self._bus,
        session_id=self._ctx.session_id,
        messages=messages,
        max_tokens=1000,
    )
    content = response.choices[0].message.content
"""

from __future__ import annotations

import logging

from openai import RateLimitError, APIStatusError, APITimeoutError

from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent

logger = logging.getLogger(__name__)

_FALLBACK_ERRORS = (RateLimitError, APIStatusError, APITimeoutError)


async def call_with_fallback(
    chain: list,           # list[LLMProvider]
    bus: AsyncStreamBus,
    session_id: str,
    messages: list[dict],
    max_tokens: int,
    **kwargs,
):
    """
    按供应商链依次调用，任一供应商失败则切换下一个。
    全部失败则抛出最后一个异常。
    """
    last_error: Exception | None = None

    for i, provider in enumerate(chain):
        try:
            if i > 0:
                await bus.emit(StreamEvent(
                    type=EventType.ERROR,
                    data={
                        "reason": f"{chain[i-1].name} 不可用（{type(last_error).__name__}），降级至 {provider.name}",
                        "fallback_provider": provider.name,
                    },
                    session_id=session_id,
                ))
                logger.warning(
                    "LLM fallback: %s -> %s (%s)",
                    chain[i - 1].name, provider.name, last_error,
                )

            response = await provider.create(
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response

        except _FALLBACK_ERRORS as e:
            last_error = e
            continue

    raise last_error
