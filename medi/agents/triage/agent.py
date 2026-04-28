"""
TriageAgent — 智能分诊 Agent

TAOR 四阶段驱动，结合显式对话状态机。
安全第一：规则层前置扫描红旗症状，不依赖 LLM 做紧急判断。
"""

from __future__ import annotations

import anthropic

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
    ) -> None:
        self._ctx = ctx
        self._bus = bus
        self._router = router or DepartmentRouter()
        self._client = anthropic.AsyncAnthropic()
        self._symptom_info = SymptomInfo()

    async def handle(self, user_input: str) -> None:
        """处理一轮用户输入，驱动状态机前进"""
        self._symptom_info.raw_descriptions.append(user_input)

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
            self._ctx.transition(DialogueState.DONE)
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
        从文本中提取症状字段（Phase 1 用简单关键词匹配）。
        Phase 2 可换 NER 模型。
        """
        duration_keywords = ["天", "小时", "分钟", "周", "个月", "年", "小时"]
        location_keywords = ["头", "胸", "腹", "背", "腿", "手", "脚", "喉", "眼", "耳"]

        for kw in duration_keywords:
            if kw in text:
                self._symptom_info.duration = text  # 粗粒度，Phase 2 细化
                break

        for kw in location_keywords:
            if kw in text:
                self._symptom_info.location = kw
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

{constraint_prompt}"""

        user_msg = f"""用户症状描述：{self._symptom_info.to_query_text()}

知识库检索结果：
{dept_list}

请基于以上信息给出：
1. 建议就诊科室（优先级排序）
2. 紧急程度（紧急/较急/普通/可观察）
3. 简要就医建议（1-2 句）"""

        response = await self._client.messages.create(
            model=self._ctx.model_config.smart,
            max_tokens=TRIAGE_TOKEN_BUDGET["respond"],
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )

        content = response.content[0].text
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": content},
            session_id=self._ctx.session_id,
        ))
        self._ctx.transition(DialogueState.DONE)
