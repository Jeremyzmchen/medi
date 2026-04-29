"""
LLM Provider 抽象层 — 跨供应商模型切换

定义统一接口 LLMProvider，每个供应商实现适配器。
调用方只依赖接口，不感知具体供应商。

支持的供应商：
  - OpenAIProvider  — OpenAI 官方 API（gpt-4o / gpt-4o-mini 等）
  - QwenProvider    — 阿里云通义千问（兼容 OpenAI 格式，需 DASHSCOPE_API_KEY）
  - LocalProvider   — 本地 Ollama（无需 API Key，离线兜底）

降级链示例：
  smart: OpenAI gpt-4o → Qwen qwen-max → Local qwen2.5:7b
  fast:  OpenAI gpt-4o-mini → Local qwen2.5:7b
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from openai import AsyncOpenAI


class LLMProvider(ABC):
    """统一 LLM 调用接口，所有供应商适配器必须实现"""

    @property
    @abstractmethod
    def name(self) -> str:
        """供应商名称，用于日志和事件"""
        ...

    @abstractmethod
    async def create(self, messages: list[dict], max_tokens: int, **kwargs) -> str:
        """
        调用 LLM，返回文本内容。
        kwargs 透传 temperature、tools、tool_choice 等参数。
        注意：tools 参数并非所有供应商都支持，适配器内部处理兼容性。
        """
        ...


class OpenAIProvider(LLMProvider):
    """
    OpenAI 官方 API 适配器。
    支持 function calling / tool use，完整兼容所有参数。
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    async def create(self, messages: list[dict], max_tokens: int, **kwargs):
        """返回原始 response 对象（供 tool_calls 解析使用）"""
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )


class QwenProvider(LLMProvider):
    """
    阿里云通义千问适配器。
    通义千问兼容 OpenAI API 格式，base_url 指向阿里云端点即可。
    需要环境变量：DASHSCOPE_API_KEY
    """

    def __init__(self, model: str = "qwen-max") -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @property
    def name(self) -> str:
        return f"qwen/{self._model}"

    async def create(self, messages: list[dict], max_tokens: int, **kwargs):
        # 通义千问不支持 tool_choice="auto" 以外的值，过滤掉 tools 参数避免报错
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )


class LocalProvider(LLMProvider):
    """
    本地 Ollama 适配器（离线兜底）。
    Ollama 兼容 OpenAI API 格式，base_url 指向本地服务。
    无需 API Key，适合断网或 API 配额耗尽场景。
    """

    def __init__(self, model: str = "qwen2.5:7b") -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key="ollama",  # Ollama 不校验 key，填占位符即可
            base_url="http://localhost:11434/v1",
        )

    @property
    def name(self) -> str:
        return f"local/{self._model}"

    async def create(self, messages: list[dict], max_tokens: int, **kwargs):
        # 本地模型不支持 tool use，过滤掉相关参数
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )


def build_smart_chain() -> list[LLMProvider]:
    """
    根据可用的环境变量，动态构建 smart 模型的供应商降级链。
    没有对应 API Key 的供应商自动跳过（LocalProvider 除外，本地始终可尝试）。
    """
    chain: list[LLMProvider] = []

    if os.getenv("OPENAI_API_KEY"):
        chain.append(OpenAIProvider(model="gpt-4o"))

    if os.getenv("DASHSCOPE_API_KEY"):
        chain.append(QwenProvider(model="qwen-max"))

    # 本地 Ollama 作为最后兜底，不需要 API Key
    chain.append(LocalProvider(model="qwen2.5:7b"))

    return chain


def build_fast_chain() -> list[LLMProvider]:
    """
    根据可用的环境变量，动态构建 fast 模型的供应商降级链。
    """
    chain: list[LLMProvider] = []

    if os.getenv("OPENAI_API_KEY"):
        chain.append(OpenAIProvider(model="gpt-4o-mini"))

    if os.getenv("DASHSCOPE_API_KEY"):
        chain.append(QwenProvider(model="qwen-turbo"))

    chain.append(LocalProvider(model="qwen2.5:7b"))

    return chain
