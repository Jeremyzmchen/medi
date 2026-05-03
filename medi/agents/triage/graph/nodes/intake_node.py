"""
IntakeNode - 预诊护士节点。

核心思路：
1. LLM 只负责从对话里抽取临床事实。
2. FactStore 负责合并“已知/否认/未知”的事实。
3. IntakeReviewNode 负责判断预诊档案是否足够、下一步是否追问。
"""

from __future__ import annotations

import json
import sys

from medi.agents.triage.graph.state import (
    CollectionStatus,
    TriageGraphState,
)
from medi.agents.triage.intake_facts import (
    FactStore,
    collection_status_from_facts,
    extraction_slot_prompt,
)
from medi.agents.triage.intake_protocols import (
    ResolvedIntakePlan,
    question_for_missing_field,
    resolve_intake_plan,
)
from medi.agents.triage.intake_rules import extract_deterministic_facts
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


_FACT_EXTRACT_SYSTEM_PROMPT = """你是预诊分诊系统里的临床事实抽取器，不是对话生成器。

你的任务是从护士-患者对话中抽取事实，严格输出 JSON，不要生成追问、建议或诊断。

抽取规则：
- 只记录患者明确说过的信息，不能把护士问题当作患者回答。
- 患者说”没有””否认””不是”某症状时，status 用 absent。
- 患者提到”布洛芬””退烧药””降压药”等，用 safety.current_medications。
- 过敏史必须单独抽取到 safety.allergies；不要因为患者提到用药，就推断过敏史已回答。
- 如果患者没有回答某个槽位，不要输出这个槽位，除非患者明确表示不知道。
- value 尽量保留患者原话里的关键短语，例如”三天””最高39度””咽痛，没有腹泻”。
- confidence 使用 0 到 1 的小数。
- hpi.onset 只记录症状真正开始的时间，不要把潜水、飞行、游泳、外伤等暴露事件时间当作起病时间。
- 症状出现前的相关经历抽取到 hpi.exposure_event，例如“上周菲律宾潜水”。
- 如果患者明确说暴露当时或暴露后没有某症状，抽取到 hpi.exposure_symptoms，status 用 absent，例如“潜水当时及之后无耳痛”。

一般情况（gc.*）抽取规则：
- 患者描述精神差/嗜睡/烦躁/意识模糊时，抽取到 gc.mental_status。
- 患者描述睡眠不好/失眠/嗜睡时，抽取到 gc.sleep。
- 患者描述不想吃/没胃口/能吃时，抽取到 gc.appetite。
- 患者描述大便异常（便秘/腹泻/黑便/血便）时，抽取到 gc.bowel。
- 患者描述小便异常（尿少/尿频/尿痛/无尿）时，抽取到 gc.urination。
- 患者描述体重明显变化时，抽取到 gc.weight_change。

允许的 status：
- present: 明确存在或明确回答了肯定信息
- absent: 明确否认或明确回答没有
- partial: 有信息但不完整
- unknown: 患者明确表示不知道/不清楚
- not_applicable: 对当前症状明显不适用且患者已表达

输出格式：
{
  “facts”: [
    {
      “slot”: “hpi.onset”,
      “status”: “present”,
      “value”: “昨晚开始”,
      “evidence”: “我从昨晚开始发烧”,
      “confidence”: 0.9
    }
  ]
}
"""


async def intake_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    fast_chain: list,
    fast_model: str,
    health_profile=None,
    obs=None,
) -> dict:
    """每轮进入节点时只抽取和合并事实，是否继续追问交给 IntakeReviewNode。"""
    session_id = state["session_id"]
    messages = state.get("messages") or []
    fixed_protocol_id = _locked_protocol_id(state)
    intake_plan = resolve_intake_plan(
        messages,
        health_profile,
        fixed_protocol_id=fixed_protocol_id,
    )

    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    iteration = state.get("graph_iteration", 0) + 1

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "intake", "round": assistant_count + 1},
        session_id=session_id,
    ))

    store = FactStore.from_state(state.get("intake_facts"))
    extracted_facts = await _call_fact_extractor(
        messages=messages,
        store=store,
        fast_chain=fast_chain,
        bus=bus,
        session_id=session_id,
        intake_plan=intake_plan,
        obs=obs,
    )
    store.merge_items(extracted_facts, source_turn=assistant_count + 1)
    store.merge_items(
        extract_deterministic_facts(messages, protocol_id=intake_plan.protocol_id),
        source_turn=assistant_count + 1,
    )

    collection_status = collection_status_from_facts(
        store=store,
        plan=intake_plan,
        complete=False,
        reason="等待预诊质量门评估",
    )

    print(
        f"[intake] round={assistant_count + 1} "
        f"protocol={intake_plan.protocol_id} overlays={intake_plan.overlay_ids} "
        f"meds_status={collection_status.get('medications_allergies')}",
        file=sys.stderr,
    )

    raw_user_descriptions = [
        m["content"]
        for m in messages
        if m.get("role") == "user" and m.get("content")
    ]

    return {
        "symptom_data": store.to_symptom_data(raw_user_descriptions),
        "collection_status": collection_status,
        "intake_facts": store.to_state(),
        "intake_protocol_id": intake_plan.protocol_id,
        "intake_overlays": intake_plan.overlay_ids,
        "intake_complete": False,
        "next_node": "intake_review",
        "graph_iteration": iteration,
    }


async def _call_fact_extractor(
    messages: list[dict],
    store: FactStore,
    fast_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    intake_plan: ResolvedIntakePlan,
    obs=None,
) -> list[dict]:
    conv_messages = [
        m for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    prompt = _build_fact_extract_prompt(intake_plan, store)

    try:
        response = await call_with_fallback(
            chain=fast_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
            call_type="intake_fact_extract",
            messages=[
                {"role": "system", "content": prompt},
                *conv_messages,
                {"role": "user", "content": "请根据以上完整对话抽取临床事实，严格输出 JSON。"},
            ],
            max_tokens=600,
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        return _validate_fact_output(parsed)
    except Exception as exc:
        print(f"[intake] fact extraction failed: {exc}", file=sys.stderr)
        return []


def _build_fact_extract_prompt(
    intake_plan: ResolvedIntakePlan,
    store: FactStore,
) -> str:
    plan_section = intake_plan.prompt_section(
        completed_fields=_completed_protocol_fields(intake_plan, store),
        completed_pattern_keys=_completed_pattern_keys(intake_plan, store),
    )
    return f"""{_FACT_EXTRACT_SYSTEM_PROMPT}

当前主诉采集重点：
{plan_section}

允许抽取的槽位：
{extraction_slot_prompt(intake_plan)}

当前已知事实：
{store.prompt_context()}
"""


def _validate_fact_output(parsed: dict) -> list[dict]:
    facts = parsed.get("facts") or []
    if not isinstance(facts, list):
        return []

    valid: list[dict] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        if not slot:
            continue
        valid.append(item)
    return valid


def _completed_protocol_fields(
    intake_plan: ResolvedIntakePlan,
    store: FactStore,
) -> list[str]:
    completed: list[str] = []
    field_slots = {
        "chief_complaint": ("hpi.chief_complaint",),
        "opqrst.location": ("hpi.location",),
        "opqrst.onset": ("hpi.onset",),
        "opqrst.quality": ("hpi.character",),
        "opqrst.severity": ("hpi.severity", "specific.max_temperature", "specific.frequency"),
        "opqrst.time_pattern": ("hpi.timing",),
        "opqrst.provocation": ("hpi.aggravating_alleviating",),
        "opqrst.radiation": ("hpi.radiation",),
        "associated_symptoms": ("hpi.associated_symptoms", "specific.associated_fever_symptoms"),
        "relevant_history": ("hpi.relevant_history",),
        "medications_allergies": ("safety.current_medications", "safety.allergies"),
    }
    for field in intake_plan.required_fields:
        slots = field_slots.get(field, ())
        if field == "medications_allergies":
            answered = all(store.is_answered(slot) for slot in slots)
        else:
            answered = any(store.is_answered(slot) for slot in slots)
        if slots and answered:
            completed.append(field)
    return completed


def _completed_pattern_keys(
    intake_plan: ResolvedIntakePlan,
    store: FactStore,
) -> list[str]:
    return [
        key for key, _ in intake_plan.pattern_required
        if store.is_answered(f"specific.{key}")
    ]


def _locked_protocol_id(state: TriageGraphState) -> str | None:
    protocol_id = state.get("intake_protocol_id")
    if protocol_id and protocol_id != "generic_opqrst":
        return protocol_id
    return None


def _question_for_missing_fields(
    missing_fields: list[str],
    intake_plan: ResolvedIntakePlan,
    status: CollectionStatus,
) -> str:
    """
    Backward-compatible helper used by existing tests.

    The new intake flow asks via SlotPlanner, but this helper preserves the
    older medication/allergy edge-case behavior.
    """
    if (
        "medications_allergies" in missing_fields
        and status.get("medications_allergies") == "partial"
    ):
        return "您刚才已经提到用药了，那有没有药物或食物过敏？"
    return question_for_missing_field(missing_fields, intake_plan)
