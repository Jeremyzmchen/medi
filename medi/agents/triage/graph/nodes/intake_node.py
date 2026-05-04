"""
IntakeNode - 预诊护士节点。

核心思路：
1. LLM 只负责从对话里抽取临床事实。
2. ClinicalFactStore 负责合并“已知/否认/未知”的事实。
3. IntakeReviewNode 负责判断预诊档案是否足够、下一步是否追问。
"""

from __future__ import annotations

import json
import sys

from medi.agents.triage.graph.state import (
    IntakePlanState,
    TriageGraphState,
)
from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import (
    ResolvedIntakePlan,
    resolve_intake_plan,
)
from medi.agents.triage.intake_rules import extract_deterministic_facts
from medi.agents.triage.prompts.fact_extraction import build_fact_extract_prompt
from medi.agents.triage.preconsultation_record import update_preconsultation_record
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


async def intake_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    fast_chain: list,
    profile_snapshot=None,
    obs=None,
) -> dict:
    """每轮进入节点时只抽取和合并事实，是否继续追问交给 IntakeReviewNode。"""
    session_id = state["session_id"]
    messages = state.get("messages") or []
    # 锁定主述症状
    fixed_protocol_id = _locked_protocol_id(state)
    # 信息采集计划
    intake_plan = resolve_intake_plan(
        messages,
        profile_snapshot,
        fixed_protocol_id=fixed_protocol_id,
    )
    # 系统已提问次数
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    # 图过程循环次数（用于clinicalNode节点到intake在判断信息不足时的return)
    workflow_control = state.get("workflow_control") or {}
    iteration = int(workflow_control.get("graph_iteration") or 0) + 1

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "intake", "round": assistant_count + 1},
        session_id=session_id,
    ))

    # 信息证据抽取
    store = ClinicalFactStore.from_state(state.get("clinical_facts"))
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

    preconsultation_record = update_preconsultation_record(
        state.get("preconsultation_record"),
        store=store,
        plan=intake_plan,
    )

    print(
        f"[intake] round={assistant_count + 1} "
        f"protocol={intake_plan.protocol_id} overlays={intake_plan.overlay_ids}",
        file=sys.stderr,
    )

    return {
        "intake_plan": _intake_plan_state(intake_plan),
        "preconsultation_record": preconsultation_record,
        "clinical_facts": store.to_state(),
        "workflow_control": {
            "next_node": "monitor",
            "intake_complete": False,
            "graph_iteration": iteration,
        },
    }


async def _call_fact_extractor(
    messages: list[dict],
    store: ClinicalFactStore,
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
    prompt = build_fact_extract_prompt(intake_plan, store)

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
            max_tokens=1200,
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        return _validate_fact_output(parsed)
    except Exception as exc:
        print(f"[intake] fact extraction failed: {exc}", file=sys.stderr)
        return []


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


def _locked_protocol_id(state: TriageGraphState) -> str | None:
    intake_plan = state.get("intake_plan") or {}
    protocol_id = intake_plan.get("protocol_id")
    if protocol_id and protocol_id != "generic_opqrst":
        return protocol_id
    return None


def _intake_plan_state(plan: ResolvedIntakePlan) -> IntakePlanState:
    return IntakePlanState(
        protocol_id=plan.protocol_id,
        protocol_label=plan.protocol_label,
        overlay_ids=plan.overlay_ids,
    )
