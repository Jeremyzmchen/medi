"""
IntakeControllerNode — 预诊采集调度节点。

对应论文（arXiv:2511.01445）4.2 节 Controller + 4.3 节 Prompter：
  - Controller：读取 MonitorResult 分数，选择下一个最高价值子任务槽（确定性）
  - Prompter：LLM 根据当前已知信息生成上下文感知问题（非确定性）

设计原则：
  "选什么槽"是确定性的（Monitor 分数 + 优先级规则），
  "怎么问"是 LLM 的事——把当前已知状态、目标槽位和对话历史传给 LLM，
  让它像真正的护士一样生成自然、有共情的追问，而不是枚举字典。

路由：
  can_finish_intake=True  → next_node="clinical"  → 图路由到 ClinicalNode
  can_finish_intake=False → next_node="intake_wait" → emit FOLLOW_UP，图到 END
"""

from __future__ import annotations

import sys

from medi.agents.triage.graph.state import (
    ControllerDecision,
    MAX_INTAKE_ROUNDS,
    TriageGraphState,
)
from medi.agents.triage.intake_facts import (
    BASE_SLOT_SPECS,
    FactStore,
    collection_status_from_facts,
    slot_label,
)
from medi.agents.triage.intake_protocols import (
    ResolvedIntakePlan,
    resolve_intake_plan,
)
from medi.agents.triage.task_tree import slot_task_priority
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


PREFERRED_MAX_QUESTIONS = 6

# 槽位优先级（数字越小越优先）
SLOT_PRIORITY: dict[str, int] = {
    "hpi.chief_complaint": 5,
    "hpi.onset": 20,
    "hpi.timing": 20,
    "hpi.severity": 20,
    "hpi.location": 35,
    "hpi.character": 35,
    "hpi.associated_symptoms": 35,
    "safety.current_medications": 45,
    "safety.allergies": 45,
    "hpi.relevant_history": 70,
}

# Fallback 固定问题（LLM 失败时使用）
_FALLBACK_QUESTIONS: dict[str, str] = {
    "hpi.chief_complaint": "您今天主要是哪里不舒服，能描述一下吗？",
    "hpi.onset": "这个症状是什么时候开始的？",
    "hpi.timing": "这个症状是一直持续还是时好时坏？大概持续多久了？",
    "hpi.severity": "这个不适大概有多严重？",
    "hpi.location": "具体是哪个部位不舒服？",
    "hpi.character": "能描述一下是什么样的感觉吗？",
    "hpi.associated_symptoms": "除了这个主要不适，还有其他伴随症状吗？",
    "safety.current_medications": "您目前在用什么药吗？另外有没有药物或食物过敏？",
    "safety.allergies": "有没有药物或食物过敏？",
    "hpi.relevant_history": "以前有过类似情况，或有什么相关疾病史吗？",
}

_PROMPTER_SYSTEM = """你是一位专业的预诊护士，正在通过对话为患者做就诊前信息采集。

你的任务：根据当前已知的患者信息和对话历史，为指定的目标槽位生成**一个**自然、专业的追问。

要求：
- 只问一个问题，不要一次问多个
- 语气温和、有共情，像真正的护士说话
- 如果患者上一轮的回答与目标槽位相关但不完整，先承认他说的，再追问缺失部分
- 不要机械重复槽位名称，不要出现"请问您的xxx是什么"这样的格式
- 不要给出诊断、建议或安慰
- 用中文回答，直接输出问题文本，不要有任何前缀或解释
"""


async def intake_controller_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    fast_chain: list,
    health_profile=None,
    max_rounds: int = MAX_INTAKE_ROUNDS,
    obs=None,
) -> dict:
    """
    读取 MonitorResult，选定下一槽，用 LLM 生成问题，发出追问或放行到 clinical。
    """
    session_id = state["session_id"]
    messages = state.get("messages") or []
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

    fixed_protocol_id = _locked_protocol_id(state)
    plan = resolve_intake_plan(messages, health_profile, fixed_protocol_id=fixed_protocol_id)
    store = FactStore.from_state(state.get("intake_facts"))
    monitor = state.get("monitor_result") or {}

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "controller", "round": assistant_count + 1},
        session_id=session_id,
    ))

    score = monitor.get("score", 0)
    red_flags_checked = monitor.get("red_flags_checked", False)
    safety_covered = monitor.get("safety_slots_covered", False)
    doctor_summary_ready = monitor.get("doctor_summary_ready", False)
    high_value_missing = monitor.get("high_value_missing_slots") or []
    requested_slots = state.get("requested_slots") or []
    clinical_missing = [
        s for s in (state.get("clinical_missing_slots") or [])
        if not store.is_answered(s)
    ]

    # ── Controller：判断是否可以结束采集（确定性）────────────────────
    preferred_limit = min(max_rounds, PREFERRED_MAX_QUESTIONS)
    can_finish, finish_reason = _decide_finish(
        score=score,
        red_flags_checked=red_flags_checked,
        safety_covered=safety_covered,
        doctor_summary_ready=doctor_summary_ready,
        assistant_count=assistant_count,
        max_rounds=max_rounds,
        preferred_limit=preferred_limit,
        high_value_missing=high_value_missing,
        requested_slots=requested_slots,
        clinical_missing=clinical_missing,
        plan=plan,
    )

    # ── Controller：选下一个最高价值槽（确定性）─────────────────────
    askable = [s for s in high_value_missing if requested_slots.count(s) < 2]
    next_slot = _pick_next_slot(askable, plan, clinical_missing) if not can_finish else None

    # ── Prompter：LLM 生成问题（非确定性）───────────────────────────
    next_question = None
    if next_slot:
        next_question = await _llm_generate_question(
            slot=next_slot,
            plan=plan,
            store=store,
            messages=messages,
            fast_chain=fast_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
        )

    collection_status = collection_status_from_facts(
        store=store,
        plan=plan,
        complete=can_finish,
        reason=finish_reason,
    )

    decision = ControllerDecision(
        can_finish_intake=can_finish,
        next_best_slot=next_slot,
        next_best_question=next_question,
        reason=finish_reason,
    )

    print(
        f"[controller] can_finish={can_finish} next_slot={next_slot} "
        f"score={score} round={assistant_count + 1}",
        file=sys.stderr,
    )

    if can_finish:
        return {
            "collection_status": collection_status,
            "controller_decision": decision,
            "intake_complete": True,
            "next_node": "clinical",
        }

    question = next_question or "为了让医生更快了解情况，请再补充一个最重要的信息。"
    nurse_message = {"role": "assistant", "content": question}

    await bus.emit(StreamEvent(
        type=EventType.FOLLOW_UP,
        data={"question": question, "round": assistant_count + 1},
        session_id=session_id,
    ))

    result = {
        "messages": [nurse_message],
        "collection_status": collection_status,
        "controller_decision": decision,
        "intake_complete": False,
        "next_node": "intake_wait",
    }
    if next_slot:
        result["requested_slots"] = [next_slot]
    return result


# ─────────────────────────────────────────────
# Prompter：LLM 问题生成
# ─────────────────────────────────────────────

async def _llm_generate_question(
    slot: str,
    plan: ResolvedIntakePlan,
    store: FactStore,
    messages: list[dict],
    fast_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    obs=None,
) -> str:
    """
    把当前已知信息、对话历史、目标槽位打包给 LLM，
    让它像真正的护士一样生成自然追问。
    LLM 失败时退回固定问题。
    """
    user_prompt = _build_prompter_input(slot, plan, store, messages)
    try:
        response = await call_with_fallback(
            chain=fast_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
            call_type="intake_prompter",
            messages=[
                {"role": "system", "content": _PROMPTER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=120,
            temperature=0.3,
        )
        question = response.choices[0].message.content.strip()
        # 基础合法性校验：不接受空回答或超长回答
        if question and len(question) < 200:
            return question
    except Exception as e:
        print(f"[controller] prompter LLM failed: {e}", file=sys.stderr)

    return _fallback_question(slot, plan, store)


def _build_prompter_input(
    slot: str,
    plan: ResolvedIntakePlan,
    store: FactStore,
    messages: list[dict],
) -> str:
    """
    构建 Prompter 的用户侧 prompt，包含：
    - 当前已知患者信息（FactStore 摘要）
    - 最近几轮对话
    - 目标槽位说明
    - 该槽是否已有 partial 值（告知 LLM 需要承接）
    """
    known_facts = store.prompt_context()

    # 最近 6 条消息（3 轮对话），避免 prompt 过长
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    conv_lines = []
    for m in recent_messages:
        role = "患者" if m.get("role") == "user" else "护士"
        content = str(m.get("content", "")).strip()
        if content:
            conv_lines.append(f"{role}：{content}")
    conv_text = "\n".join(conv_lines) if conv_lines else "（对话刚开始）"

    target_label = slot_label(slot, plan)
    target_desc = _slot_description(slot, plan)

    # 如果槽位已有 partial 值，告知 LLM
    partial_note = ""
    fact = store.get(slot)
    if fact and fact.status == "partial" and fact.value:
        partial_note = (
            "\n注意：患者之前提到了\"" + fact.value + "\"，"
            "但信息不完整，请承认这个回答再追问缺失部分。"
        )

    return f"""【当前已知患者信息】
{known_facts}

【最近对话记录】
{conv_text}

【目标】
需要采集的信息：{target_label}
说明：{target_desc}
协议：{plan.protocol_label}{partial_note}

请生成一个自然的追问。"""


def _slot_description(slot: str, plan: ResolvedIntakePlan) -> str:
    """返回槽位的临床含义描述，优先用协议里的标签。"""
    # specific.* 槽从协议 pattern_required 里找描述
    if slot.startswith("specific."):
        key = slot.removeprefix("specific.")
        desc = dict(plan.pattern_required).get(key)
        if desc:
            return desc
    # hpi.* 和 safety.* 用 BASE_SLOT_SPECS 的问题作描述
    spec = BASE_SLOT_SPECS.get(slot)
    if spec:
        return spec.label
    return slot_label(slot, plan)


def _fallback_question(slot: str, plan: ResolvedIntakePlan, store: FactStore) -> str:
    """LLM 失败时的最后兜底，保留少量关键槽的固定问题。"""
    # 安全槽位合并询问
    if slot == "safety.current_medications" and not store.is_answered("safety.allergies"):
        return "您目前在用什么药吗？另外有没有药物或食物过敏？"
    if slot == "safety.allergies" and store.is_answered("safety.current_medications"):
        meds = store.value("safety.current_medications") or "您提到的药物"
        return f"您提到{meds}，那有没有药物或食物过敏？"

    if slot in _FALLBACK_QUESTIONS:
        return _FALLBACK_QUESTIONS[slot]
    spec = BASE_SLOT_SPECS.get(slot)
    if spec:
        return spec.question
    return f"关于{plan.protocol_label}，还需要了解{slot_label(slot, plan)}，方便的话请补充一下？"


# ─────────────────────────────────────────────
# Controller 调度逻辑（确定性）
# ─────────────────────────────────────────────

def _decide_finish(
    score: int,
    red_flags_checked: bool,
    safety_covered: bool,
    doctor_summary_ready: bool,
    assistant_count: int,
    max_rounds: int,
    preferred_limit: int,
    high_value_missing: list[str],
    requested_slots: list[str],
    clinical_missing: list[str],
    plan: ResolvedIntakePlan,
) -> tuple[bool, str]:
    askable = [s for s in high_value_missing if requested_slots.count(s) < 2]
    next_slot = _pick_next_slot(askable, plan, clinical_missing)

    if assistant_count >= max_rounds:
        return True, "达到最大追问轮数，已将未采集信息标记给医生补问"
    if doctor_summary_ready and safety_covered:
        return True, "主诉、病程、严重程度、危险信号和安全信息已具备医生预诊价值"
    if assistant_count >= preferred_limit and score >= 65 and red_flags_checked:
        return True, "核心预诊档案已具备价值，低收益信息不再继续打扰患者"
    if not next_slot and score >= 55:
        return True, "剩余缺失项已多次追问或价值较低，进入医生侧总结"
    return False, f"医生预诊档案还缺少：{slot_label(next_slot, plan) if next_slot else '必要信息'}"


def _pick_next_slot(
    slots: list[str],
    plan: ResolvedIntakePlan,
    clinical_missing: list[str],
) -> str | None:
    if not slots:
        return None
    clinical_set = set(clinical_missing)
    return sorted(
        slots,
        key=lambda s: (0 if s in clinical_set else 1, _slot_priority(s, plan)),
    )[0]


def _slot_priority(slot: str, plan: ResolvedIntakePlan) -> int:
    tree_priority = slot_task_priority(slot, plan)
    if tree_priority < 999:
        return tree_priority
    if slot in SLOT_PRIORITY:
        return SLOT_PRIORITY[slot]
    if slot.startswith("specific."):
        key = slot.removeprefix("specific.")
        critical = {k for k, _ in plan.pattern_required}
        return 10 if key in critical else 30
    return 99


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _locked_protocol_id(state: TriageGraphState) -> str | None:
    protocol_id = state.get("intake_protocol_id")
    if protocol_id and protocol_id != "generic_opqrst":
        return protocol_id
    return None
