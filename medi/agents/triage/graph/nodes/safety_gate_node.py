"""
SafetyGateNode - graph-native red-flag guardrail.

This node keeps the deterministic emergency check inside the LangGraph flow.
It checks the latest user message, then either routes to intake or stops the
graph after emitting escalation/result events.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command

from medi.agents.triage.graph.state import TriageGraphState
from medi.agents.triage.urgency_evaluator import (
    EMERGENCY_RESPONSE,
    _RED_FLAG_KEYWORDS,
    UrgencyLevel,
    evaluate_urgency_by_rules,
)
from medi.core.llm_client import call_with_fallback
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent


SEMANTIC_SAFETY_ANCHORS = (
    "呼吸", "喘", "氧", "胸", "心", "冷汗", "出汗", "左臂", "胳膊",
    "意识", "昏", "晕", "抽搐", "说话", "口角", "嘴歪", "肢体",
    "手脚", "头痛", "头疼", "出血", "咳血", "呕血", "便血", "过敏", "肿",
)

SAFETY_KEY_POINTS = (
    "respiratory_distress: current breathing difficulty, cannot catch breath, "
    "air hunger, rapid breathing, or cannot get oxygen.",
    "cardiac_emergency: chest pain/tightness or heart discomfort with sweating, "
    "breathing difficulty, left arm/jaw radiation, fainting, or cardiac-arrest wording.",
    "altered_consciousness: fainting, coma, confusion, seizure, or loss of consciousness.",
    "stroke_signs: facial droop, slurred speech, unilateral weakness/numbness.",
    "severe_bleeding: heavy bleeding, coughing blood, vomiting blood, or large bloody stool.",
    "thunderclap_headache: sudden worst headache with vomiting, neck stiffness, or neurologic signs.",
    "anaphylaxis: allergy symptoms with throat tightness, face/lip/tongue swelling, wheeze, or dyspnea.",
)


async def safety_gate_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    fast_chain: list | None = None,
    obs=None,
) -> Command[Literal["intake", "__end__"]]:
    """Run the deterministic red-flag check before intake processing."""
    session_id = state["session_id"]
    text = _latest_user_text(state.get("messages") or [])

    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "safety_gate"},
        session_id=session_id,
    ))

    urgency = evaluate_urgency_by_rules(text)
    if (
        urgency
        and urgency.level == UrgencyLevel.EMERGENCY
        and not _rule_hit_is_negated(text)
    ):
        return await _block(
            state=state,
            bus=bus,
            session_id=session_id,
            reason=urgency.reason,
            method="rule",
            risk_concept="explicit_red_flag",
            confidence=1.0,
            triggered_by_rule=True,
        )

    semantic = await _semantic_safety_check(
        text=text,
        state=state,
        bus=bus,
        session_id=session_id,
        fast_chain=fast_chain or [],
        obs=obs,
    )
    if _semantic_should_block(semantic):
        return await _block(
            state=state,
            bus=bus,
            session_id=session_id,
            reason=str(semantic.get("reason") or "Semantic safety classifier detected emergency risk."),
            method="llm",
            risk_concept=str(semantic.get("risk_concept") or ""),
            confidence=float(semantic.get("confidence") or 0.0),
            triggered_by_rule=False,
        )

    return Command(
        update={
            "safety_gate": {
                "status": "passed",
                "urgency_level": semantic.get("urgency_level"),
                "reason": semantic.get("reason") or "No emergency red-flag rule matched.",
                "triggered_by_rule": False,
                "method": semantic.get("method") or "none",
                "risk_concept": semantic.get("risk_concept"),
                "confidence": semantic.get("confidence"),
            },
            "workflow_control": {
                "next_node": "intake",
                "intake_complete": False,
                "graph_iteration": (state.get("workflow_control") or {}).get("graph_iteration", 0),
            },
        },
        goto="intake",
    )


async def _block(
    *,
    state: TriageGraphState,
    bus: AsyncStreamBus,
    session_id: str,
    reason: str,
    method: str,
    risk_concept: str,
    confidence: float,
    triggered_by_rule: bool,
) -> Command[Literal["__end__"]]:
    await bus.emit(StreamEvent(
        type=EventType.ESCALATION,
        data={"reason": reason},
        session_id=session_id,
    ))
    await bus.emit(StreamEvent(
        type=EventType.RESULT,
        data={"content": EMERGENCY_RESPONSE},
        session_id=session_id,
    ))
    return Command(
        update={
            "safety_gate": {
                "status": "blocked",
                "urgency_level": UrgencyLevel.EMERGENCY.value,
                "reason": reason,
                "triggered_by_rule": triggered_by_rule,
                "method": method,
                "risk_concept": risk_concept,
                "confidence": confidence,
            },
            "triage_output": {
                "meta": {
                    "schema_version": "safety_gate.v1",
                    "source": "safety_gate",
                    "fallback_used": True,
                    "fallback_reason": reason,
                },
                "patient": {},
                "doctor_report": {},
            },
            "workflow_control": {
                "next_node": END,
                "intake_complete": False,
                "graph_iteration": (state.get("workflow_control") or {}).get("graph_iteration", 0),
            },
        },
        goto=END,
    )


async def _semantic_safety_check(
    *,
    text: str,
    state: TriageGraphState,
    bus: AsyncStreamBus,
    session_id: str,
    fast_chain: list,
    obs=None,
) -> dict:
    if not text.strip() or not fast_chain or not _may_need_semantic_check(text):
        return {
            "method": "none",
            "urgency_level": None,
            "reason": "No emergency red-flag rule matched.",
            "risk_concept": None,
            "confidence": None,
        }

    try:
        response = await asyncio.wait_for(
            call_with_fallback(
                chain=fast_chain,
                bus=bus,
                session_id=session_id,
                obs=obs,
                call_type="safety_gate_semantic",
                messages=_semantic_prompt(text, state),
                max_tokens=220,
                temperature=0,
            ),
            timeout=4.0,
        )
    except Exception:
        return {
            "method": "llm_unavailable",
            "urgency_level": None,
            "reason": "Semantic safety classifier unavailable; deterministic rules did not match.",
            "risk_concept": None,
            "confidence": None,
        }

    content = response.choices[0].message.content.strip()
    parsed = _parse_json_object(content)
    if not parsed:
        return {
            "method": "llm_parse_failed",
            "urgency_level": None,
            "reason": "Semantic safety classifier returned an unreadable decision.",
            "risk_concept": None,
            "confidence": None,
        }

    return {
        "method": "llm",
        "decision": str(parsed.get("decision") or "pass").lower(),
        "urgency_level": parsed.get("urgency_level"),
        "risk_concept": parsed.get("risk_concept"),
        "confidence": _safe_float(parsed.get("confidence"), default=0.0),
        "reason": str(parsed.get("reason") or ""),
    }


def _semantic_prompt(text: str, state: TriageGraphState) -> list[dict]:
    recent_messages = (state.get("messages") or [])[-6:]
    key_points = "\n".join(f"- {item}" for item in SAFETY_KEY_POINTS)
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative medical safety classifier for a triage chatbot. "
                "Decide whether the latest user message describes a current emergency. "
                "Use only these key emergency points:\n"
                f"{key_points}\n\n"
                "Do not block for negated symptoms, hypothetical worry, past history, "
                "or symptoms of another person unless the current patient is at risk. "
                "Return only JSON with keys: decision, urgency_level, risk_concept, "
                "confidence, reason. decision must be block or pass. urgency_level "
                "must be emergency, urgent, or normal."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "latest_user_message": text,
                    "recent_messages": recent_messages,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _semantic_should_block(result: dict) -> bool:
    if result.get("method") != "llm":
        return False
    confidence = _safe_float(result.get("confidence"), default=0.0)
    return (
        result.get("decision") == "block"
        and result.get("urgency_level") == UrgencyLevel.EMERGENCY.value
        and confidence >= 0.75
    )


def _may_need_semantic_check(text: str) -> bool:
    return any(anchor in text for anchor in SEMANTIC_SAFETY_ANCHORS)


def _rule_hit_is_negated(text: str) -> bool:
    negations = ("没有", "无", "不", "不是", "否认", "未")
    found = False
    for keyword in _RED_FLAG_KEYWORDS:
        index = text.find(keyword)
        if index < 0:
            continue
        found = True
        before = text[max(0, index - 6):index]
        if not any(negation in before for negation in negations):
            return False
    return found


def _parse_json_object(content: str) -> dict:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""
