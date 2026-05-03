"""
OrchestratorAgent — 意图识别 + 子 Agent 路由

职责：
  1. 对用户输入做意图分类（gpt-4o-mini，轻量快速）
  2. 根据意图路由到对应 Agent，或直接回复边界提示
  3. 混合输入场景引导用户分步提问（不做拆分）

意图类别（人工定义，LLM 做分类）：
  symptom      — 补充当前分诊过程中的症状信息（分诊进行中）
  new_symptom  — 全新的、与当前分诊无关的新主诉（分诊已完成后）
  medication   — 咨询药物、用量、副作用、药物冲突
  followup     — 追问对话中任何已有内容（科室位置、紧急程度、就诊流程等）
  health_report — 用户上传或描述体检报告，需要解读 + 膳食 + 日程建议
  out_of_scope — 医院推荐、挂号、天气等超出范围的问题
"""

from __future__ import annotations

from enum import Enum

from medi.core.context import DialogueState, UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback

# 意图描述（注入 LLM prompt，直接影响分类准确率）
_INTENT_DESCRIPTIONS = {
    "symptom": "用户在补充当前分诊过程中的症状信息，包括伴随症状、发作时间、诱因等",
    "new_symptom": "用户描述了与之前完全无关的新身体不适，或上一次分诊已完成后重新开始描述新症状",
    "medication": "用户在咨询药物名称、用药剂量、副作用、两种药能否同时吃等用药问题",
    "followup": "用户在追问对话中任何已有内容，包括科室位置、就诊流程、紧急程度含义、需要做什么检查等",
    "health_report": "用户提供了体检报告内容（血糖、血脂、血压等指标）或要求解读体检报告，需要异常解读、膳食建议和日程安排",
    "out_of_scope": "用户问了超出医疗咨询范围的问题，例如推荐医院、如何挂号、天气等。注意：简单问候（你好、您好、hi、hello）不属于此类，应归为 followup",
}

_OUT_OF_SCOPE_REPLY = (
    "我是分诊助手，只能帮您判断应就诊的科室和紧急程度。\n"
    "您的问题超出了我的服务范围，建议通过搜索引擎或医院官网获取相关信息。\n\n"
    "如果您有身体不适需要分诊，请描述您的症状。"
)

_GREETING_REPLY = (
    "您好！我是 Medi 分诊助手，请描述您的症状，我来帮您判断应就诊的科室和紧急程度。"
)

_NO_CONTEXT_FOLLOWUP_REPLY = (
    "我目前还没有上一轮分诊结果可以解释。请先描述您的症状；"
    "如果您想咨询用药或解读健康报告，也可以直接说明具体问题。"
)

_GREETING_TEXTS = {
    "你好",
    "您好",
    "hi",
    "hello",
    "hey",
    "嗨",
    "哈喽",
}


def _is_greeting(text: str) -> bool:
    normalized = text.strip().lower()
    normalized = normalized.strip(" ，。！？!?,~～")
    return normalized in {item.lower() for item in _GREETING_TEXTS}


class Intent(Enum):
    SYMPTOM       = "symptom"
    NEW_SYMPTOM   = "new_symptom"
    MEDICATION    = "medication"
    FOLLOWUP      = "followup"
    HEALTH_REPORT = "health_report"
    OUT_OF_SCOPE  = "out_of_scope"


class OrchestratorAgent:
    def __init__(self, ctx: UnifiedContext, bus: AsyncStreamBus) -> None:
        self._ctx = ctx
        self._bus = bus

    async def classify_intent(
        self,
        user_input: str,
        symptom_summary: str = "",
    ) -> Intent:
        """
        用 gpt-4o-mini 做意图分类，返回 Intent 枚举。

        注入三类上下文信号：
          1. dialogue_state — 分诊是否进行中（TRIAGE_GRAPH_RUNNING / INTAKE_WAITING）或已完成（INIT）
          2. symptom_summary — 当前已收集的 OPQRST 摘要
          3. 近期对话历史 — 让分类器感知上下文
        """
        # 问诊意图
        _intake_active_states = {
            DialogueState.TRIAGE_GRAPH_RUNNING,
            DialogueState.INTAKE_WAITING,
        }
        if self._ctx.dialogue_state in _intake_active_states:
            return Intent.SYMPTOM

        # 意图类别
        intent_list = "\n".join(
            f"- {name}: {desc}"
            for name, desc in _INTENT_DESCRIPTIONS.items()
        )

        # 会话状态 (INIT, RUNNING, ESCALATING, WAITTING)
        state = self._ctx.dialogue_state.value
        state_hint = (
            "（注意：当前分诊护士正在采集病史，用户新输入是对护士问题的回答，"
            "应归为 symptom，即使内容很短（如数字、单词、'不清楚'等））"
            if self._ctx.dialogue_state in _intake_active_states
            else "（注意：上一次分诊已完成，用户新输入更可能是新主诉）"
        )

        # 症状上下文
        symptom_context = ""
        if symptom_summary and symptom_summary != "（无结构化信息）":
            symptom_context = f"\n\n[当前已收集的症状信息]\n{symptom_summary}"

        # 对话上下文（最近10条）
        history_context = ""
        if self._ctx.messages:
            history_context = "\n\n[近期对话历史]\n" + "\n".join(
                f"{m['role']}: {m['content'][:100]}"
                for m in self._ctx.messages[-10:]
            )

        response = await call_with_fallback(
            chain=self._ctx.model_config.fast_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="intent_classify",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个医疗对话意图分类器。根据对话状态、已收集症状和用户最新输入，"
                        "从以下类别中选择最匹配的一个，只输出类别名，不要输出其他内容。\n\n"
                        f"[对话状态] {state} {state_hint}\n\n"
                        f"[意图类别]\n{intent_list}"
                        f"{symptom_context}"
                        f"{history_context}"
                    ),
                },
                {"role": "user", "content": user_input},
            ],
            # 控制字数，防止模型幻觉瞎输出
            max_tokens=15,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip().lower()

        # 校验意图分类回应
        for intent in Intent:
            if intent.value in raw:
                return intent

        # 默认当症状处理，避免误拒
        return Intent.SYMPTOM

    async def handle_followup(self, user_input: str) -> None:
        """处理对上一条建议的追问，带完整对话历史回答"""

        if not self._ctx.messages:
            content = (
                _GREETING_REPLY
                if _is_greeting(user_input)
                else _NO_CONTEXT_FOLLOWUP_REPLY
            )
            await self._bus.emit(StreamEvent(
                type=EventType.RESULT,
                data={"content": content},
                session_id=self._ctx.session_id,
            ))
            return

        self._ctx.add_user_message(user_input)

        response = await call_with_fallback(
            chain=self._ctx.model_config.fast_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="followup_answer",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位分诊助手。用户正在追问你之前给出的分诊建议，"
                        "请基于完整对话历史简洁回答，不要做超出分诊范围的建议。"
                    ),
                },
            ] + self._ctx.messages,
            max_tokens=300,
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

