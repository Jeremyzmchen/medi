"""
IntakePrompterNode - task-focused follow-up question generation.

This node reads the task selected by Controller and turns it into one natural
patient-facing question. Scheduling remains task-driven; extraction fields are
not part of the Controller contract.
"""

from __future__ import annotations

import json
import sys

from medi.agents.triage.graph.state import TriageGraphState
from medi.agents.triage.intake_facts import (
    BASE_SLOT_SPECS,
    FactStore,
    slot_label,
)
from medi.agents.triage.intake_protocols import (
    ResolvedIntakePlan,
    resolve_intake_plan,
)
from medi.agents.triage.task_definitions import TASK_BY_ID
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


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

_TASK_FALLBACK_QUESTIONS: dict[str, str] = {
    "T1_PRIMARY_DEPARTMENT": "您今天主要是哪里不舒服，能描述一下最困扰您的症状吗？",
    "T1_SECONDARY_DEPARTMENT": "这个不适最明显的部位和表现是什么？",
    "T2_ONSET": "这个症状是什么时候开始的？",
    "T2_MAIN_SYMPTOM_CHARACTERISTICS": "能描述一下这个不适的部位、性质和严重程度吗？",
    "T2_DISEASE_PROGRESSION": "从开始到现在，这个不适是在加重、减轻、反复，还是基本没变化？",
    "T2_ACCOMPANYING_SYMPTOMS": "除了这个主要不适，还有其他伴随症状吗？",
    "T2_DIAGNOSTIC_THERAPEUTIC_HISTORY": "发病后有没有做过检查、用过药或做过处理？效果怎么样？",
    "T2_GENERAL_CONDITION": "发病以后精神、睡眠、食欲、大小便或体重有没有明显变化？",
    "T3_DISEASE_HISTORY": "以前有过重要疾病、慢性病，或类似的不适吗？",
    "T3_IMMUNIZATION_HISTORY": "疫苗接种情况大致正常吗，近期有没有接种过疫苗？",
    "T3_SURGICAL_TRAUMA_HISTORY": "以前做过手术，或有过比较重要的外伤吗？",
    "T3_BLOOD_TRANSFUSION_HISTORY": "以前输过血或有过输血不良反应吗？",
    "T3_ALLERGY_HISTORY": "有没有药物、食物或其他东西过敏？",
    "T3_CURRENT_MEDICATIONS": "孩子这两天有没有用过退烧药、止咳药或其他药？如果完全没用药，也可以直接说没有。",
}

_PROMPTER_SYSTEM = """你是一位专业的预诊护士，正在通过对话为患者做就诊前信息采集。

你的任务：根据当前选中的采集任务、已知患者信息和对话历史，生成**一个**自然、专业的追问。

要求：
- 只问一个问题，不要一次问多个
- 围绕当前采集任务提问，不要向患者暴露任务编号或技术字段名
- 语气温和、有共情，像真正的护士说话
- 如果患者上一轮的回答相关但不完整，先承认他说的，再追问缺失部分
- 不要给出诊断、建议或安慰
- 用中文回答，直接输出问题文本，不要有任何前缀或解释
"""


async def intake_prompter_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    fast_chain: list,
    health_profile=None,
    obs=None,
) -> dict:
    """Generate the follow-up question for the task chosen by Controller."""
    session_id = state["session_id"]
    messages = state.get("messages") or []
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "prompter", "round": assistant_count + 1},
        session_id=session_id,
    ))

    fixed_protocol_id = _locked_protocol_id(state)
    plan = resolve_intake_plan(messages, health_profile, fixed_protocol_id=fixed_protocol_id)
    store = FactStore.from_state(state.get("intake_facts"))
    decision = dict(state.get("controller_decision") or {})

    task_id = decision.get("next_best_task") or state.get("current_task")
    task_instruction = decision.get("task_instruction") or _task_instruction(task_id)

    question = await _llm_generate_question(
        task_id=task_id,
        task_instruction=task_instruction,
        plan=plan,
        store=store,
        medical_record=state.get("medical_record") or {},
        task_progress=state.get("task_progress") or {},
        messages=messages,
        fast_chain=fast_chain,
        bus=bus,
        session_id=session_id,
        obs=obs,
    )

    decision["next_best_question"] = question

    print(
        f"[prompter] task={task_id}",
        file=sys.stderr,
    )

    return {
        "controller_decision": decision,
        "next_node": "inquirer",
    }


async def _llm_generate_question(
    *,
    task_id: str | None,
    task_instruction: str | None,
    plan: ResolvedIntakePlan,
    store: FactStore,
    medical_record: dict,
    task_progress: dict[str, dict],
    messages: list[dict],
    fast_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    obs=None,
) -> str:
    user_prompt = _build_prompter_input(
        task_id=task_id,
        task_instruction=task_instruction,
        plan=plan,
        store=store,
        medical_record=medical_record,
        task_progress=task_progress,
        messages=messages,
    )
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
        if question and len(question) < 200:
            return question
    except Exception as e:
        print(f"[prompter] LLM failed: {e}", file=sys.stderr)

    return _fallback_question(None, plan, store, task_id=task_id)


def _build_prompter_input(
    *,
    task_id: str | None,
    task_instruction: str | None,
    plan: ResolvedIntakePlan,
    store: FactStore,
    medical_record: dict,
    task_progress: dict[str, dict],
    messages: list[dict],
) -> str:
    known_facts = store.prompt_context()
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    conv_lines = []
    for m in recent_messages:
        role = "患者" if m.get("role") == "user" else "护士"
        content = str(m.get("content", "")).strip()
        if content:
            conv_lines.append(f"{role}：{content}")
    conv_text = "\n".join(conv_lines) if conv_lines else "（对话刚开始）"

    task = TASK_BY_ID.get(task_id or "")
    task_label = task.label if task else (task_id or "当前采集任务")
    task_desc = task.description if task else ""
    progress = task_progress.get(task_id or "", {})
    missing = [
        detail.get("description") or detail.get("id")
        for detail in progress.get("requirement_details") or []
        if not detail.get("completed")
    ]

    record_text = json.dumps(medical_record, ensure_ascii=False, indent=2)
    if len(record_text) > 1800:
        record_text = record_text[:1800] + "\n..."

    return f"""【当前采集任务】
任务：{task_label}
任务目标：{task_desc or task_instruction or "补齐医生预诊所需信息"}
调度说明：{task_instruction or "根据当前任务完成度继续追问"}
任务完成度：{progress.get("score", 0)}，状态：{progress.get("status", "pending")}
当前缺口：{"；".join(missing) if missing else "由当前任务目标决定"}

【当前已知患者信息】
{known_facts}

【结构化病历草稿】
{record_text}

【最近对话记录】
{conv_text}

请生成一个围绕当前采集任务的自然追问。"""
def _fallback_question(
    slot: str | None,
    plan: ResolvedIntakePlan,
    store: FactStore,
    *,
    task_id: str | None = None,
) -> str:
    if slot == "safety.current_medications" and not store.is_answered("safety.allergies"):
        return "您目前在用什么药吗？另外有没有药物或食物过敏？"
    if slot == "safety.allergies" and store.is_answered("safety.current_medications"):
        meds = store.value("safety.current_medications") or "您提到的药物"
        return f"您提到{meds}，那有没有药物或食物过敏？"

    if slot in _FALLBACK_QUESTIONS:
        return _FALLBACK_QUESTIONS[slot]
    if slot in BASE_SLOT_SPECS:
        return BASE_SLOT_SPECS[slot].question

    if task_id in _TASK_FALLBACK_QUESTIONS:
        return _TASK_FALLBACK_QUESTIONS[task_id]
    task = TASK_BY_ID.get(task_id or "")
    if task:
        return f"关于{task.label}，方便的话请再补充一点相关信息吗？"
    return f"关于{plan.protocol_label}，方便的话请再补充一个对医生判断最重要的信息吗？"


def _task_instruction(task_id: str | None) -> str | None:
    task = TASK_BY_ID.get(task_id or "")
    if not task:
        return None
    return f"推进任务：{task.label}。目标：{task.description}"


def _locked_protocol_id(state: TriageGraphState) -> str | None:
    protocol_id = state.get("intake_protocol_id")
    if protocol_id and protocol_id != "generic_opqrst":
        return protocol_id
    return None
