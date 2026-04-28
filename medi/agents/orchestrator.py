"""
OrchestratorAgent — 意图识别 + 子 Agent 路由

职责：
  1. 对每条用户输入做意图分类（gpt-4o-mini，轻量快速）
  2. 根据意图路由到对应 Agent，或直接回复边界提示

意图类别（人工定义，LLM 做分类）：
  symptom     — 描述身体不适、症状、疼痛，需要分诊
  medication  — 咨询药物、用量、副作用、药物冲突
  followup    — 追问刚才给出的建议（科室详情、紧急程度解释等）
  out_of_scope — 医院推荐、挂号、天气等超出范围的问题
"""

from __future__ import annotations

from enum import Enum

from openai import AsyncOpenAI

from medi.core.context import UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent

# 意图描述（注入 LLM prompt，直接影响分类准确率）
_INTENT_DESCRIPTIONS = {
    "symptom": "用户在描述身体不适、症状、疼痛、发烧、头晕等，需要分诊指导",
    "medication": "用户在咨询药物名称、用药剂量、副作用、两种药能否同时吃等用药问题",
    "followup": "用户在追问刚才给出的分诊建议，例如询问科室在哪、紧急程度是什么意思、还需要做什么检查等",
    "out_of_scope": "用户问了超出医疗咨询范围的问题，例如推荐医院、如何挂号、天气、闲聊等",
}

_OUT_OF_SCOPE_REPLY = (
    "我是分诊助手，只能帮您判断应就诊的科室和紧急程度。\n"
    "您的问题超出了我的服务范围，建议通过搜索引擎或医院官网获取相关信息。\n\n"
    "如果您有身体不适需要分诊，请描述您的症状。"
)


class Intent(Enum):
    SYMPTOM      = "symptom"
    MEDICATION   = "medication"
    FOLLOWUP     = "followup"
    OUT_OF_SCOPE = "out_of_scope"


class OrchestratorAgent:
    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus
        self._client = AsyncOpenAI()
        self._last_response: str = ""   # 上一次给出的建议，供 followup 时作上下文

    async def classify_intent(self, user_input: str) -> Intent:
        """用 gpt-4o-mini 做意图分类，返回 Intent 枚举"""
        intent_list = "\n".join(
            f"- {name}: {desc}"
            for name, desc in _INTENT_DESCRIPTIONS.items()
        )

        # 带上对话历史，让分类器能判断"我是摔倒的"是 followup 还是新症状
        history_context = ""
        if self._ctx.messages:
            history_context = "\n\n[当前对话历史]\n" + "\n".join(
                f"{m['role']}: {m['content'][:100]}"
                for m in self._ctx.messages[-6:]  # 最近 3 轮
            )

        response = await self._client.chat.completions.create(
            model=self._ctx.model_config.fast,  # gpt-4o-mini
            max_tokens=10,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个意图分类器。根据用户最新输入和对话历史，从以下类别中选择最匹配的一个，"
                        "只输出类别名，不要输出其他内容。\n\n"
                        f"{intent_list}"
                        f"{history_context}"
                    ),
                },
                {"role": "user", "content": user_input},
            ],
        )

        raw = response.choices[0].message.content.strip().lower()

        # 容错：LLM 输出可能带标点或多余空格
        for intent in Intent:
            if intent.value in raw:
                return intent

        # 默认当症状处理，避免误拒
        return Intent.SYMPTOM

    async def handle_followup(self, user_input: str) -> None:
        """处理对上一条建议的追问，带完整对话历史回答"""
        if not self._ctx.messages:
            # 没有对话历史，当新症状处理（不应发生）
            return

        self._ctx.add_user_message(user_input)

        response = await self._client.chat.completions.create(
            model=self._ctx.model_config.fast,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位分诊助手。用户正在追问你之前给出的分诊建议，"
                        "请基于完整对话历史简洁回答，不要做超出分诊范围的建议。"
                    ),
                },
            ] + self._ctx.messages,
        )

        content = response.choices[0].message.content
        self._ctx.add_assistant_message(content)
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": content},
            session_id=self._ctx.session_id,
        ))

    async def handle_out_of_scope(self) -> None:
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": _OUT_OF_SCOPE_REPLY},
            session_id=self._ctx.session_id,
        ))

    def update_last_response(self, content: str) -> None:
        """TriageAgent 给出建议后，记录到 Orchestrator 供 followup 使用"""
        self._last_response = content
