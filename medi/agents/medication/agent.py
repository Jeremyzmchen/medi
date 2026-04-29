"""
MedicationAgent — 用药咨询 Agent

处理三类用药问题：
  1. 药物查询   — 这个药是干什么的、怎么吃
  2. 副作用咨询 — 吃了 XX 有什么副作用
  3. 药物冲突   — XX 和 YY 能一起吃吗

当前实现：GPT-4o 直接回答，注入 HealthProfile 硬约束。
免责：回答仅供参考，以药品说明书或医生/药师建议为准。

后续扩展（未实现）：
  - 接药物数据库 API（NMPA/RxNorm/OpenFDA），需加追问提取标准化药物名
  - 多模态输入：拍药盒/说明书图片，OCR 提取药物名后查库
"""

from __future__ import annotations

from medi.core.context import UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback

MEDICATION_SYSTEM_PROMPT = """你是一位专业的用药咨询助手，帮助用户了解药物信息。

你可以回答：
- 药物的用途和基本用法
- 常见副作用和注意事项
- 药物之间是否存在冲突

你不能做的：
- 给出具体用药剂量（请遵医嘱或参考说明书）
- 替代医生/药师的专业诊断
- 判断用户是否需要用某种药

⚠️ 重要声明：以下信息仅供参考，实际用药请以药品说明书、医生或执业药师的建议为准。"""


class MedicationAgent:
    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus

    async def handle(self, user_input: str) -> None:
        """处理一次用药咨询，单轮 GPT-4o 调用"""
        self._ctx.add_user_message(user_input)

        constraint_prompt = self._ctx.build_constraint_prompt()
        system = MEDICATION_SYSTEM_PROMPT
        if constraint_prompt:
            system += f"\n\n{constraint_prompt}"

        messages = (
            [{"role": "system", "content": system}]
            + self._ctx.messages
        )

        response = await call_with_fallback(
            chain=self._ctx.model_config.smart_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="medication",
            messages=messages,
            max_tokens=600,
        )

        content = response.choices[0].message.content
        self._ctx.add_assistant_message(content)

        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": content},
            session_id=self._ctx.session_id,
        ))
