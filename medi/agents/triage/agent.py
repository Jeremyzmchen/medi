"""
TriageAgent — 智能分诊 Agent

TAOR 四阶段驱动，结合显式对话状态机。
安全第一：规则层前置扫描红旗症状，不依赖 LLM 做紧急判断。
"""

from __future__ import annotations

from transformers import pipeline as hf_pipeline
from openai import AsyncOpenAI

from medi.core.context import DialogueState, UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.agents.triage.symptom_collector import SymptomInfo, build_follow_up_question
from medi.agents.triage.urgency_evaluator import (
    UrgencyLevel,
    evaluate_urgency_by_rules,
    EMERGENCY_RESPONSE,
)
from medi.agents.triage.department_router import DepartmentRouter

TRIAGE_TOKEN_BUDGET = {
    "think":   200,
    "act":     100,
    "observe": 400,
    "respond": 1000,
}


class TriageAgent:
    def __init__(
        self,
        ctx: UnifiedContext,
        bus: AsyncStreamBus,
        router: DepartmentRouter | None = None,
        on_result=None,  # 回调：建议生成后通知 Orchestrator
    ) -> None:
        self._ctx = ctx
        self._bus = bus
        self._router = router or DepartmentRouter()
        self._client = AsyncOpenAI()
        self._symptom_info = SymptomInfo()
        self._on_result = on_result  # Callable[[str], None]
        # NER 模型懒加载（首次调用时初始化，避免启动慢）
        self._ner = None

    async def handle(self, user_input: str) -> None:
        """处理一轮用户输入，驱动状态机前进"""
        self._symptom_info.raw_descriptions.append(user_input)
        self._ctx.add_user_message(user_input)

        # Safety-First：规则层前置扫描，不等 LLM
        urgency = evaluate_urgency_by_rules(user_input)
        if urgency and urgency.level == UrgencyLevel.EMERGENCY:
            self._ctx.transition(DialogueState.ESCALATING)
            await self._bus.emit(StreamEvent(
                type=EventType.ESCALATION,
                data={"reason": urgency.reason},
                session_id=self._ctx.session_id,
            ))
            await self._bus.emit(StreamEvent(
                type=EventType.RESULT,
                data={"content": EMERGENCY_RESPONSE},
                session_id=self._ctx.session_id,
            ))
            self._ctx.transition(DialogueState.INIT)
            self._ctx.follow_up_count = 0
            self._symptom_info = SymptomInfo()
            return

        # 正常流程：TAOR 四阶段
        await self._think(user_input)

    async def _think(self, user_input: str) -> None:
        """Think 阶段：判断症状信息是否充分"""
        self._ctx.transition(DialogueState.COLLECTING)
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "think"},
            session_id=self._ctx.session_id,
        ))

        # 简单规则判断（Phase 1），后续可换 LLM 判断
        # 从用户输入中提取基本症状信息
        self._extract_symptom_info(user_input)

        if not self._symptom_info.is_sufficient() and self._ctx.can_follow_up():
            # 信息不足，进入追问
            await self._act_follow_up()
        else:
            # 信息足够（或追问次数已满），进入检索
            self._ctx.transition(DialogueState.SUFFICIENT)
            await self._act_search()

    def _extract_symptom_info(self, text: str) -> None:
        """
        用 NER 模型从文本中提取解剖部位和症状实体。
        模型：Adapting/bert-base-chinese-finetuned-NER-biomedical
        """
        if self._ner is None:
            self._ner = hf_pipeline(
                "ner",
                model="Adapting/bert-base-chinese-finetuned-NER-biomedical",
                aggregation_strategy="simple",
            )

        entities = self._ner(text)

        for ent in entities:
            label = ent["entity_group"]
            word = ent["word"].replace(" ", "")  # 去掉 tokenizer 分词空格

            # 解剖部位 → location
            if "部位" in label and not self._symptom_info.location:
                self._symptom_info.location = word

            # 症状 → accompanying
            elif "症状" in label:
                if word not in self._symptom_info.accompanying:
                    self._symptom_info.accompanying.append(word)

        # 时间信息用关键词兜底（NER 模型对时间实体识别较弱）
        if not self._symptom_info.duration:
            for kw in ["天", "小时", "分钟", "周", "个月", "年", "昨", "今", "刚"]:
                if kw in text:
                    self._symptom_info.duration = text
                    break

    async def _act_follow_up(self) -> None:
        """Act 阶段：生成追问"""
        self._ctx.increment_follow_up()
        question = build_follow_up_question(self._symptom_info.missing_fields())

        await self._bus.emit(StreamEvent(
            type=EventType.FOLLOW_UP,
            data={"question": question, "round": self._ctx.follow_up_count},
            session_id=self._ctx.session_id,
        ))

    async def _act_search(self) -> None:
        """Act 阶段：检索症状-科室知识库"""
        self._ctx.transition(DialogueState.SEARCHING)
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "act"},
            session_id=self._ctx.session_id,
        ))

        query = self._symptom_info.to_query_text()
        candidates = await self._router.route(query)

        await self._observe(candidates)

    async def _observe(self, candidates: list) -> None:
        """Observe 阶段：评估检索结果"""
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "observe"},
            session_id=self._ctx.session_id,
        ))

        await self._respond(candidates)

    async def _respond(self, candidates: list) -> None:
        """Respond 阶段：生成最终建议，注入健康档案硬约束"""
        self._ctx.transition(DialogueState.RESPONDING)
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "respond"},
            session_id=self._ctx.session_id,
        ))

        constraint_prompt = self._ctx.build_constraint_prompt()
        dept_list = "\n".join(
            f"- {c.department}（置信度 {c.confidence:.0%}）：{c.reason}"
            for c in candidates
        )

        system = f"""你是一位专业的分诊助手，帮助用户判断应就诊的科室。
回答要简洁、专业、有温度。不做最终诊断，只做科室引导。
结合完整对话历史理解用户的全部症状，不要只看最后一条消息。

{constraint_prompt}"""

        # 在完整对话历史后追加一条系统注入的检索结果
        retrieval_note = (
            f"[知识库检索结果]\n{dept_list}\n\n"
            "请基于以上检索结果和完整对话历史给出：\n"
            "1. 建议就诊科室（优先级排序）\n"
            "2. 紧急程度（紧急/较急/普通/可观察）\n"
            "3. 简要就医建议（1-2 句）"
        )

        messages = (
            [{"role": "system", "content": system}]
            + self._ctx.messages
            + [{"role": "user", "content": retrieval_note}]
        )

        response = await self._client.chat.completions.create(
            model=self._ctx.model_config.smart,
            max_tokens=TRIAGE_TOKEN_BUDGET["respond"],
            messages=messages,
        )

        content = response.choices[0].message.content
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": content},
            session_id=self._ctx.session_id,
        ))
        # 把 assistant 回复追加进对话历史
        self._ctx.add_assistant_message(content)
        # 通知 Orchestrator 记录本次建议，供 followup 使用
        if self._on_result:
            self._on_result(content)
        # 重置状态机和追问计数，但保留 symptom_info 积累（同一会话内）
        self._ctx.transition(DialogueState.INIT)
        self._ctx.follow_up_count = 0
        self._symptom_info = SymptomInfo()
