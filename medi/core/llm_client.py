"""
LLM 调用封装 — 跨供应商降级策略 + 可观测性采集

按供应商链依次尝试，捕获以下错误时自动切换下一个供应商：
- RateLimitError（429 限流）
- APIStatusError（500/503 服务不可用）
- APITimeoutError（超时）

每次调用（成功或失败）都写入 ObservabilityStore，记录：
- 供应商名、是否降级、token 消耗、耗时、错误类型

使用方式：
    response = await call_with_fallback(
        chain=self._ctx.model_config.smart_chain,
        bus=self._bus,
        session_id=self._ctx.session_id,
        obs=self._ctx.observability,
        messages=messages,
        max_tokens=1000,
    )
    content = response.choices[0].message.content
"""

from __future__ import annotations

import logging
from datetime import datetime

from openai import RateLimitError, APIStatusError, APITimeoutError

from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent

logger = logging.getLogger(__name__)

_FALLBACK_ERRORS = (RateLimitError, APIStatusError, APITimeoutError)


async def call_with_fallback(
    chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    messages: list[dict],
    max_tokens: int,
    obs=None,           # ObservabilityStore | None
    call_type: str = "unknown",
    **kwargs,
):
    """
    按供应商链依次调用，任一供应商失败则切换下一个。
    每次调用结果（含失败）都记录到 ObservabilityStore。
    全部失败则抛出最后一个异常。
    """
    from medi.core.observability import LLMTrace

    last_error: Exception | None = None

    for i, provider in enumerate(chain):
        is_fallback = i > 0
        start = datetime.now()

        if is_fallback:
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

        try:
            response = await provider.create(
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency_ms = int((datetime.now() - start).total_seconds() * 1000)

            # 采集成功 trace
            if obs is not None:
                usage = getattr(response, "usage", None)
                obs.record_llm(LLMTrace(
                    session_id=session_id,
                    timestamp=start,
                    provider=provider.name,
                    call_type=call_type,
                    is_fallback=is_fallback,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                    success=True,
                ))

            return response

        except _FALLBACK_ERRORS as e:
            latency_ms = int((datetime.now() - start).total_seconds() * 1000)

            # 采集失败 trace
            if obs is not None:
                obs.record_llm(LLMTrace(
                    session_id=session_id,
                    timestamp=start,
                    provider=provider.name,
                    call_type=call_type,
                    is_fallback=is_fallback,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                    success=False,
                    error_type=type(e).__name__,
                ))

            last_error = e
            continue

    raise last_error
