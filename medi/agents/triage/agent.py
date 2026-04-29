"""
TriageAgent — 智能分诊 Agent

TAOR 四阶段驱动，结合显式对话状态机。
安全第一：规则层前置扫描红旗症状，不依赖 LLM 做紧急判断。

Phase 3：Act 阶段改为 LLM 驱动的 Tool Use（ReAct 模式）。
LLM 自主决定何时调用 search_symptom_kb，通过 ToolRuntime 执行，
支持未来扩展多工具（药物库、检验指标等）而无需修改 Agent 主流程。
"""

from __future__ import annotations

import json

from transformers import pipeline as hf_pipeline
from openai import AsyncOpenAI

from medi.core.context import DialogueState, UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.tool_runtime import ToolRuntime
from medi.core.llm_client import call_with_fallback
from medi.agents.triage.symptom_collector import SymptomInfo, build_follow_up_question
from medi.agents.triage.urgency_evaluator import (
    UrgencyLevel,
    evaluate_urgency_by_rules,
    EMERGENCY_RESPONSE,
)
from medi.agents.triage.department_router import DepartmentRouter
from medi.agents.triage.tools import make_search_tool, make_urgency_tool, SEARCH_SYMPTOM_KB_SCHEMA
from medi.memory.episodic import EpisodicMemory

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
        self._client = AsyncOpenAI()
        self._symptom_info = SymptomInfo()
        self._episodic = EpisodicMemory(ctx.user_id)
        # ToolRuntime：注册 search_symptom_kb 工具
        self._tool_runtime = ToolRuntime(ctx=ctx, bus=bus)
        self._tool_runtime.register(make_search_tool(self._router))
        self._tool_runtime.register(make_urgency_tool(
            call_with_fallback=call_with_fallback,
            fast_chain=ctx.model_config.fast_chain,
            bus=bus,
            session_id=ctx.session_id,
            obs=ctx.observability,
        ))
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

    def _record_stage(self, stage: str, start: "datetime") -> None:
        """向 ObservabilityStore 写入阶段耗时"""
        if self._ctx.observability is None:
            return
        from datetime import datetime
        from medi.core.observability import StageTrace
        latency_ms = int((datetime.now() - start).total_seconds() * 1000)
        self._ctx.observability.record_stage(StageTrace(
            session_id=self._ctx.session_id,
            timestamp=start,
            stage=stage,
            latency_ms=latency_ms,
        ))

    async def _think(self, user_input: str) -> None:
        """Think 阶段：判断症状信息是否充分"""
        from datetime import datetime
        stage_start = datetime.now()

        self._ctx.transition(DialogueState.COLLECTING)
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "think"},
            session_id=self._ctx.session_id,
        ))

        self._extract_symptom_info(user_input)

        # NER 未能提取到部位时，用 LLM 兜底推断
        if not self._symptom_info.region:
            await self._enrich_region(user_input)

        self._record_stage("think", stage_start)

        if not self._symptom_info.is_sufficient() and self._ctx.can_follow_up():
            # 信息不足，进入追问
            await self._act_follow_up()
        else:
            # 信息足够（或追问次数已满），进入检索
            self._ctx.transition(DialogueState.SUFFICIENT)
            await self._act_search()

    def _extract_symptom_info(self, text: str) -> None:
        """
        从文本中提取 OPQRST 症状字段：
        - NER 模型：解剖部位(R) + 伴随症状
        - 关键词规则：时间(T)、诱因(O)、性质(Q)、程度(S)、加重缓解(P)
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

            # 解剖部位 → R（region）
            if "部位" in label and not self._symptom_info.region:
                self._symptom_info.region = word

            # 症状 → accompanying
            elif "症状" in label:
                if word not in self._symptom_info.accompanying:
                    self._symptom_info.accompanying.append(word)

        # T — 时间/持续时长（NER 对时间实体弱，用关键词兜底）
        if not self._symptom_info.time_pattern:
            for kw in ["天", "小时", "分钟", "周", "个月", "年", "昨", "今", "刚", "最近", "一直"]:
                if kw in text:
                    self._symptom_info.time_pattern = text
                    break

        # O — 发作诱因关键词
        if not self._symptom_info.onset:
            for kw in ["吃", "喝", "运动", "跑", "劳累", "睡", "受凉", "着凉", "摔", "撞", "扭"]:
                if kw in text:
                    self._symptom_info.onset = text
                    break

        # Q — 症状性质关键词
        if not self._symptom_info.quality:
            for kw in ["刺痛", "钝痛", "胀痛", "酸痛", "烧灼", "压迫", "撕裂", "绞痛", "麻", "痒"]:
                if kw in text:
                    self._symptom_info.quality = kw
                    break

        # S — 严重程度（数字评分）
        if not self._symptom_info.severity:
            import re
            m = re.search(r"([0-9]|10)\s*分", text)
            if m:
                self._symptom_info.severity = m.group(1)

        # P — 加重缓解关键词
        if not self._symptom_info.provocation:
            for kw in ["变重", "加重", "变轻", "缓解", "好转", "休息", "活动后", "饭后", "弯腰"]:
                if kw in text:
                    self._symptom_info.provocation = text
                    break

    async def _enrich_region(self, text: str) -> None:
        """
        NER 未提取到解剖部位时，用 LLM 推断症状对应的身体部位。
        例："腹泻" → "腹部"，"头晕" → "头部"，"咳嗽" → "胸/呼吸道"
        输出"未知"时不写入，让后续追问来补。
        """
        response = await call_with_fallback(
            chain=self._ctx.model_config.fast_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="enrich_region",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个医学助手。根据用户描述的症状，推断最可能对应的解剖部位，"
                        "只输出部位名称（1-4个字），无法判断时输出'未知'。"
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip()
        if result and result != "未知" and result != "'未知'":
            self._symptom_info.region = result

    async def _act_follow_up(self) -> None:
        """Act 阶段：用 LLM 生成 OPQRST 追问"""
        self._ctx.increment_follow_up()
        question = await build_follow_up_question(
            missing=self._symptom_info.missing_fields(),
            symptom_info=self._symptom_info,
            client=self._client,
            fast_model=self._ctx.model_config.fast,
        )

        await self._bus.emit(StreamEvent(
            type=EventType.FOLLOW_UP,
            data={"question": question, "round": self._ctx.follow_up_count},
            session_id=self._ctx.session_id,
        ))

    async def _act_search(self) -> None:
        """
        Act 阶段：LLM 驱动的 Tool Use（ReAct 模式）。

        LLM 收到症状摘要后，自主决定调用 search_symptom_kb 工具。
        ToolRuntime 执行工具并返回结构化结果，LLM 拿到结果后进入 Observe。
        最多循环 3 次（防止 LLM 无限 tool call），未调用工具则直接用空结果生成建议。
        """
        from datetime import datetime
        stage_start = datetime.now()

        self._ctx.transition(DialogueState.SEARCHING)
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "act"},
            session_id=self._ctx.session_id,
        ))

        symptom_summary = self._symptom_info.to_summary()
        constraint_prompt = self._ctx.build_constraint_prompt()
        history_prompt = await self._episodic.build_history_prompt()

        system = f"""你是一位专业的分诊助手。请根据用户的症状信息，调用知识库检索工具获取科室建议。

{constraint_prompt}
{history_prompt}"""

        # ReAct 循环的 messages（独立于 ctx.messages，不污染对话历史）
        act_messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"[当前症状摘要]\n{symptom_summary}\n\n"
                    "请调用 search_symptom_kb 工具检索适合的科室。"
                ),
            },
        ]

        tool_result = None
        for _ in range(3):  # 最多 3 次 tool call
            response = await call_with_fallback(
                chain=self._ctx.model_config.fast_chain,
                bus=self._bus,
                session_id=self._ctx.session_id,
                obs=self._ctx.observability,
                call_type="act_search",
                messages=act_messages,
                max_tokens=TRIAGE_TOKEN_BUDGET["act"],
                tools=[SEARCH_SYMPTOM_KB_SCHEMA],
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # LLM 没有调用工具，退出循环
            if not msg.tool_calls:
                break

            # 执行所有 tool call（通常只有一个）
            act_messages.append(msg)  # 把 assistant 消息加入循环 messages
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                tool_result = await self._tool_runtime.call(
                    tc.function.name, **args
                )
                # 把工具结果作为 tool 消息加入循环
                act_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
            break  # 拿到结果后退出，不做多轮检索

        # 从 tool_result 重建 candidates 列表（供 _respond 使用）
        candidates = []
        if tool_result and "candidates" in tool_result:
            from medi.agents.triage.department_router import DepartmentCandidate
            candidates = [
                DepartmentCandidate(
                    department=c["department"],
                    confidence=c["confidence"],
                    reason=c["reason"],
                )
                for c in tool_result["candidates"]
            ]

        self._record_stage("act", stage_start)
        await self._observe(candidates)

    async def _observe(self, candidates: list) -> None:
        """
        Observe 阶段：拿到 Act 检索结果后，调 evaluate_urgency 工具做 LLM 紧急程度评估。

        放在此阶段的原因：
        - Think 阶段症状信息可能还不完整（追问未结束）
        - Act 阶段职责是"调工具查科室"，不应混入紧急评估
        - Observe 语义是"拿到 Act 结果后分析判断"，职责最匹配
        - Respond 阶段只做生成，不做判断

        evaluate_urgency 是规则层（红旗关键词）的第二层兜底，
        处理规则层未覆盖的语义场景（如"拉了很多血"）。
        工具失败不阻断主流程，_respond 用 None 降级处理。
        """
        from datetime import datetime
        stage_start = datetime.now()

        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "observe"},
            session_id=self._ctx.session_id,
        ))

        # 调用 evaluate_urgency 工具做 LLM 紧急程度评估
        urgency_result = None
        try:
            symptom_text = self._symptom_info.to_summary()
            urgency_result = await self._tool_runtime.call(
                "evaluate_urgency",
                symptom_text=symptom_text,
            )
        except Exception:
            pass  # 工具失败不阻断主流程，_respond 用 None 降级处理

        self._record_stage("observe", stage_start)
        await self._respond(candidates, urgency_result)

    async def _respond(self, candidates: list, urgency_result: dict | None = None) -> None:
        """Respond 阶段：生成最终建议，注入健康档案硬约束"""
        from datetime import datetime
        stage_start = datetime.now()

        self._ctx.transition(DialogueState.RESPONDING)
        await self._bus.emit(StreamEvent(
            type=EventType.STAGE_START,
            data={"stage": "respond"},
            session_id=self._ctx.session_id,
        ))

        constraint_prompt = self._ctx.build_constraint_prompt()
        history_prompt = await self._episodic.build_history_prompt()
        dept_list = "\n".join(
            f"- {c.department}（置信度 {c.confidence:.0%}）：{c.reason}"
            for c in candidates
        )
        symptom_summary = self._symptom_info.to_summary()

        system = f"""你是一位专业的分诊助手，帮助用户判断应就诊的科室。
回答要简洁、专业、有温度。不做最终诊断，只做科室引导。
结合完整对话历史理解用户的全部症状，不要只看最后一条消息。

{constraint_prompt}
{history_prompt}"""

        # 注入 LLM 紧急程度评估结果（第二层兜底）
        urgency_hint = ""
        if urgency_result:
            urgency_hint = f"\n[紧急程度评估]\n{urgency_result['reason']}（等级：{urgency_result['level']}）\n"

        # 在完整对话历史后追加结构化症状摘要 + 知识库检索结果
        retrieval_note = (
            f"[OPQRST 症状摘要]\n{symptom_summary}\n\n"
            f"[知识库检索结果]\n{dept_list}\n"
            f"{urgency_hint}\n"
            "请基于以上信息和完整对话历史给出：\n"
            "1. 建议就诊科室（优先级排序）\n"
            "2. 紧急程度（紧急/较急/普通/可观察）\n"
            "3. 简要就医建议（1-2 句，可提及医生会进一步询问哪些信息）"
        )

        messages = (
            [{"role": "system", "content": system}]
            + self._ctx.messages
            + [{"role": "user", "content": retrieval_note}]
        )

        response = await call_with_fallback(
            chain=self._ctx.model_config.smart_chain,
            bus=self._bus,
            session_id=self._ctx.session_id,
            obs=self._ctx.observability,
            call_type="respond",
            messages=messages,
            max_tokens=TRIAGE_TOKEN_BUDGET["respond"],
        )

        self._record_stage("respond", stage_start)
        content = response.choices[0].message.content
        await self._bus.emit(StreamEvent(
            type=EventType.RESULT,
            data={"content": content},
            session_id=self._ctx.session_id,
        ))
        # 把 assistant 回复追加进对话历史
        self._ctx.add_assistant_message(content)
        # 保存分诊记录到 EpisodicMemory
        top_department = candidates[0].department if candidates else "待确认"
        await self._episodic.save(
            symptom_summary=symptom_summary,
            advice=content,
            department=top_department,
        )
        # 重置状态机和追问计数，但保留 symptom_info 积累（同一会话内）
        self._ctx.transition(DialogueState.INIT)
        self._ctx.follow_up_count = 0
        self._symptom_info = SymptomInfo()
