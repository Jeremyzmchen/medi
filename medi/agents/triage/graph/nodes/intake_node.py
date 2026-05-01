"""
IntakeNode — 预诊护士节点

设计说明：
  不使用 interrupt/resume，而是每轮 ainvoke 完整跑一次图。
  IntakeNode 判断：
    - 信息充分 → intake_complete=True → 图继续走到 ClinicalNode
    - 信息不足 → 发出追问 → intake_complete=False → 图到 END（本轮结束）

  Runner 每轮都 invoke（不 resume），checkpointer 保存跨轮累积的 messages
  和 collection_status，下一轮从 checkpoint 恢复后继续追问。

  这样每轮只触发一次 LLM 调用，行为完全可预测。
"""

from __future__ import annotations

import json

from medi.agents.triage.graph.state import (
    TriageGraphState,
    CollectionStatus,
    OPQRSTStatus,
    SymptomData,
    empty_symptom_data,
    empty_collection_status,
    check_minimum_fields,
    count_partial_fields,
    MAX_INTAKE_ROUNDS,
)
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback


_NURSE_SYSTEM_PROMPT = """你是一位经验丰富的预诊护士（Pre-visit Nurse）。

你的职责是在患者见到医生之前，系统地采集完整病史，让医生一进门就有足够信息开始诊断。

【第一步：仔细阅读对话历史，提取已知信息】
在生成任何问题之前，先完整阅读对话历史，提取患者已经告诉你的所有信息。
已经知道的信息绝对不能再问。

【需要采集的信息】
1. 主诉（患者最主要的不适）
2. 部位（如果症状本身已经隐含部位，就标 complete，比如"拉稀"→部位是肠道，"头痛"→部位是头部）
3. 发作时间（什么时候开始）
4. 症状性质（根据症状类型灵活判断）：
   - 疼痛类 → 问刺痛/钝痛/胀痛等
   - 消化道症状（腹泻/呕吐/恶心）→ 问次数、颜色、水样/血样等
   - 发烧 → 问体温数值
   - 皮肤症状 → 问颜色、形状、是否瘙痒
   - 呼吸道症状 → 问干咳/有痰、痰色等
   - 注意：quality 是对症状特征的描述，不要对非疼痛症状问"刺痛还是钝痛"
5. 严重程度（根据症状类型灵活判断）：
   - 疼痛 → 0-10分
   - 发烧 → 体温数值（已有体温就标 complete）
   - 腹泻 → 今天几次、量多少
   - 呕吐 → 吐了几次（用户说"吐了两次"就标 complete）
6. 时间特征（持续性还是间歇性，以及多久了）
7. 诱因和加重/缓解因素
8. 放射痛/扩散（适用于疼痛类症状，消化道症状可标 partial）
9. 伴随症状（患者已提到的都算 complete）
10. 既往类似情况
11. ★ 用药史和过敏史【必须直接询问，不可跳过，不可推测】
12. 症状特异性问题（外伤→活动能力；发烧→有无寒战；腹泻→有无血便）

【medications_allergies 规则】
- 必须直接问患者，不能靠推测
- 患者明确回答"没有"也算 complete
- 对话历史中没有出现患者的直接回答 → 必须是 missing

【对话原则】
- 每次只问一个问题，语气自然温和
- 绝对不重复已知信息
- 不做诊断，不给建议
- 根据症状类型选择合适的问法，不要机械套用疼痛模板

【输出格式】
严格输出 JSON，不要输出任何其他内容：

{
  "next_question": "下一个问题（所有信息都采集完时为空字符串\"\"）",
  "collection_status": {
    "chief_complaint": "complete|partial|missing",
    "opqrst": {
      "onset": "complete|partial|missing",
      "provocation": "complete|partial|missing",
      "quality": "complete|partial|missing",
      "location": "complete|partial|missing",
      "severity": "complete|partial|missing",
      "time_pattern": "complete|partial|missing",
      "radiation": "complete|partial|missing"
    },
    "associated_symptoms": "complete|partial|missing",
    "relevant_history": "complete|partial|missing",
    "medications_allergies": "complete|partial|missing",
    "pattern_specific": {},
    "can_conclude": false,
    "reason": "说明哪些字段还缺，或者为什么可以结束"
  }
}

can_conclude 为 true 的条件：
1. 所有 opqrst 字段均为 complete 或 partial（不适用的字段标 partial）
2. chief_complaint、associated_symptoms、relevant_history 均为 complete 或 partial
3. medications_allergies 必须是 complete
以上三条同时满足。"""


async def intake_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    fast_chain: list,
    fast_model: str,
    obs=None,
) -> dict:
    """
    IntakeNode 执行函数。

    每次 ainvoke 调用一次，判断是否需要继续追问：
    - 需要追问 → 发出问题，intake_complete=False，图走到 END
    - 信息充分 → intake_complete=True，图继续走到 ClinicalNode
    """
    session_id = state["session_id"]
    messages = state.get("messages") or []
    current_status = state.get("collection_status") or empty_collection_status()

    # 用 assistant 消息数量推算追问轮次
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    iteration = state.get("graph_iteration", 0) + 1

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "intake", "round": assistant_count + 1},
        session_id=session_id,
    ))

    # ── 硬性上限兜底 ──
    if assistant_count >= MAX_INTAKE_ROUNDS:
        return {
            "symptom_data": await _extract_symptom_data(messages, fast_chain, bus, session_id, obs),
            "collection_status": current_status,
            "intake_complete": True,
            "next_node": "clinical",
            "graph_iteration": iteration,
        }

    # ── 调用护士 LLM ──
    nurse_result = await _call_nurse_llm(
        messages=messages,
        current_status=current_status,
        fast_chain=fast_chain,
        bus=bus,
        session_id=session_id,
        obs=obs,
    )

    new_status = nurse_result["collection_status"]
    next_question = nurse_result["next_question"]

    # ── 终止判断 ──
    can_conclude = new_status.get("can_conclude", False)
    min_passed, min_missing = check_minimum_fields(new_status, assistant_count)

    import sys
    print(
        f"[intake] round={assistant_count+1} can_conclude={can_conclude} "
        f"min_passed={min_passed} missing={min_missing} "
        f"meds_status={new_status.get('medications_allergies')}",
        file=sys.stderr,
    )

    # min_passed=True：所有必填字段已采集
    # 但如果仍有较多 partial 字段且轮次还不够，让护士继续补充细节
    # partial_threshold: 超过 3 个 partial 且 assistant_count < 5 时继续问
    if min_passed:
        partial_count = count_partial_fields(new_status)
        if partial_count <= 3 or assistant_count >= 5:
            return {
                "symptom_data": await _extract_symptom_data(messages, fast_chain, bus, session_id, obs),
                "collection_status": new_status,
                "intake_complete": True,
                "next_node": "clinical",
                "graph_iteration": iteration,
            }
        # partial 过多，继续让护士问（LLM 的 next_question 自然补充细节）

    # 最低字段未通过但 LLM 说够了：补问缺失字段
    if can_conclude and not min_passed:
        next_question = _minimum_field_question(min_missing)

    # LLM 返回空问题但未结束（防御性兜底）
    if not next_question:
        next_question = _minimum_field_question(min_missing) if min_missing else "请问还有什么其他不舒服需要告诉我吗？"

    # ── 把护士问题写入 messages，发出 FOLLOW_UP，本轮结束 ──
    nurse_message = {"role": "assistant", "content": next_question}

    await bus.emit(StreamEvent(
        type=EventType.FOLLOW_UP,
        data={
            "question": next_question,
            "round": assistant_count + 1,
        },
        session_id=session_id,
    ))

    return {
        "messages": [nurse_message],   # operator.add 追加
        "collection_status": new_status,
        "intake_complete": False,
        "next_node": "intake_wait",     # 特殊标记：本轮到此为止，等待用户
        "graph_iteration": iteration,
    }


async def _call_nurse_llm(
    messages: list[dict],
    current_status: CollectionStatus,
    fast_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    obs=None,
) -> dict:
    # 只传对话历史给 LLM，不传 collection_status
    # 让 LLM 完全从对话内容判断已知信息，避免"状态全 missing → 重复问已知内容"
    conv_messages = [m for m in messages if m.get("role") in ("user", "assistant")]

    llm_messages = (
        [{"role": "system", "content": _NURSE_SYSTEM_PROMPT}]
        + conv_messages
        + [{"role": "user", "content": "请根据以上对话，输出下一步 JSON："}]
    )

    try:
        response = await call_with_fallback(
            chain=fast_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
            call_type="intake_nurse",
            messages=llm_messages,
            max_tokens=400,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return _validate_output(parsed)

    except Exception:
        return {
            "next_question": "您还有什么其他不舒服的地方需要告诉我吗？",
            "collection_status": {**current_status, "can_conclude": False},
        }


def _validate_output(parsed: dict) -> dict:
    question = parsed.get("next_question", "")
    raw_status = parsed.get("collection_status", {})

    opqrst = raw_status.get("opqrst", {})
    for f in ("onset", "provocation", "quality", "location", "severity", "time_pattern", "radiation"):
        opqrst.setdefault(f, "missing")

    status = CollectionStatus(
        chief_complaint=raw_status.get("chief_complaint", "missing"),
        opqrst=OPQRSTStatus(**opqrst),
        associated_symptoms=raw_status.get("associated_symptoms", "missing"),
        relevant_history=raw_status.get("relevant_history", "missing"),
        medications_allergies=raw_status.get("medications_allergies", "missing"),
        pattern_specific=raw_status.get("pattern_specific") or {},
        can_conclude=bool(raw_status.get("can_conclude", False)),
        reason=raw_status.get("reason", ""),
    )
    return {"next_question": question, "collection_status": status}


def _minimum_field_question(missing_fields: list[str]) -> str:
    questions = {
        "chief_complaint":        "您今天主要是哪里不舒服，能描述一下吗？",
        "opqrst.location":        "您能告诉我具体是哪个部位不舒服吗？",
        "opqrst.onset":           "这个症状是什么时候开始的？",
        "opqrst.quality":         "您能描述一下是什么样的感觉吗？比如刺痛、钝痛、胀痛还是其他？",
        "opqrst.severity":        "如果用0到10分来衡量，您觉得疼痛或不适大概是几分？",
        "opqrst.time_pattern":    "这个症状是一直持续还是时好时坏？大概多久了？",
        "medications_allergies":  "您目前在服用什么药物吗？有没有药物或食物过敏？",
    }
    for f in missing_fields:
        if f in questions:
            return questions[f]
    return "请问还有什么重要信息需要告诉我吗？"


_EXTRACT_SYSTEM_PROMPT = """从以下护士-患者对话中提取结构化症状信息。
严格输出 JSON，不要输出任何其他内容：

{
  "onset": "发作时间/诱因，没有则 null",
  "provocation": "加重或缓解因素，没有则 null",
  "quality": "症状性质描述（疼痛类：刺痛/钝痛；消化道：水样/血样；发烧：体温数值），没有则 null",
  "region": "解剖部位（如：腹部、右膝、耳道），没有则 null",
  "severity": "严重程度（疼痛 0-10 分或体温或次数），没有则 null",
  "time_pattern": "持续性/间歇性/急性发作，以及持续时长，没有则 null",
  "radiation": "放射痛或扩散部位，没有则 null",
  "accompanying": ["伴随症状列表，没有则空数组"],
  "relevant_history": "既往相关病史，没有则 null",
  "medications": ["当前用药列表，没有则空数组"],
  "allergies": ["过敏史列表，没有则空数组"]
}"""


async def _extract_symptom_data(
    messages: list[dict],
    fast_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    obs=None,
) -> SymptomData:
    """用 LLM 从对话历史中提取结构化 SymptomData，供向量检索和 ClinicalNode 使用。"""
    raw = [m["content"] for m in messages if m.get("role") == "user"]
    conv = [m for m in messages if m.get("role") in ("user", "assistant")]

    try:
        response = await call_with_fallback(
            chain=fast_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
            call_type="symptom_extract",
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            ] + conv + [
                {"role": "user", "content": "请从以上对话中提取症状信息，输出 JSON："},
            ],
            max_tokens=300,
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        return SymptomData(
            raw_descriptions=raw,
            onset=parsed.get("onset"),
            provocation=parsed.get("provocation"),
            quality=parsed.get("quality"),
            region=parsed.get("region"),
            severity=parsed.get("severity"),
            time_pattern=parsed.get("time_pattern"),
            radiation=parsed.get("radiation"),
            accompanying=list(parsed.get("accompanying") or []),
            relevant_history=parsed.get("relevant_history"),
            medications=list(parsed.get("medications") or []),
            allergies=list(parsed.get("allergies") or []),
        )
    except Exception:
        # 提取失败退回原始文本，RAG 检索词会差一些，但不影响主流程
        return SymptomData(
            raw_descriptions=raw,
            onset=None, provocation=None, quality=None,
            region=None, severity=None, time_pattern=None,
            radiation=None, accompanying=[],
            relevant_history=None, medications=[], allergies=[],
        )
