from types import SimpleNamespace

import pytest

import medi.agents.triage.graph.nodes.clinical_node as clinical_module
from medi.agents.triage.graph.nodes.clinical_node import _missing_for_diagnosis
from medi.agents.triage.urgency_evaluator import UrgencyLevel, UrgencyResult
from medi.core.stream_bus import AsyncStreamBus


def test_clinical_missing_for_diagnosis_returns_explicit_slots() -> None:
    state = {
        "intake_plan": {"protocol_id": "chest_pain"},
        "clinical_facts": [
            {"slot": "specific.exertional_related", "status": "present", "value": "活动后加重"},
        ],
        "preconsultation_record": {
            "t2_hpi": {
                "onset": {"value": "今天"},
                "timing": {"value": "持续"},
                "associated_symptoms": {"value": "出汗"},
            },
        },
    }

    missing = _missing_for_diagnosis(state)

    assert "hpi.location" in missing
    assert "hpi.radiation" in missing
    assert "specific.dyspnea_sweating" in missing


def test_clinical_missing_for_diagnosis_empty_when_no_actionable_gap() -> None:
    state = {
        "intake_plan": {"protocol_id": "fever"},
        "preconsultation_record": {
            "t2_hpi": {
                "onset": {"value": "昨晚"},
                "timing": {"value": "反复发热"},
                "associated_symptoms": {"value": "咽痛"},
                "specific": {
                    "max_temperature": {"value": "39度"},
                },
            },
        },
    }

    assert _missing_for_diagnosis(state) == []


def test_clinical_missing_for_diagnosis_does_not_repeat_unknown_answer() -> None:
    state = {
        "intake_plan": {"protocol_id": "chest_pain"},
        "clinical_facts": [
            {"slot": "hpi.onset", "status": "unknown", "value": None},
            {"slot": "hpi.associated_symptoms", "status": "absent", "value": "无"},
            {"slot": "hpi.location", "status": "present", "value": "胸口"},
            {"slot": "hpi.radiation", "status": "unknown", "value": None},
            {"slot": "specific.dyspnea_sweating", "status": "unknown", "value": None},
            {"slot": "specific.exertional_related", "status": "present", "value": "活动后无明显变化"},
        ],
    }
    missing = _missing_for_diagnosis(state)

    assert "hpi.onset" not in missing
    assert "hpi.associated_symptoms" not in missing
    assert "hpi.radiation" not in missing
    assert "specific.dyspnea_sweating" not in missing


class _RouterStub:
    async def route(self, query_text: str, top_k: int = 3):
        return [
            SimpleNamespace(department="神经内科", confidence=0.82, reason="头痛相关"),
        ]


@pytest.mark.asyncio
async def test_clinical_node_routes_with_command_to_output(monkeypatch) -> None:
    async def fake_urgency(**kwargs):
        return UrgencyResult(
            level=UrgencyLevel.NORMAL,
            reason="普通就医",
            triggered_by_rule=False,
        )

    async def fake_differential(**kwargs):
        return [
            {
                "condition": "偏头痛",
                "likelihood": "high",
                "reasoning": "症状匹配",
                "supporting_symptoms": ["头痛"],
                "risk_factors": [],
            }
        ]

    monkeypatch.setattr(clinical_module, "evaluate_urgency_by_llm", fake_urgency)
    monkeypatch.setattr(clinical_module, "_generate_differential", fake_differential)
    monkeypatch.setattr(
        clinical_module,
        "evaluate_risk_factors",
        lambda symptom_summary, profile_snapshot: {
            "risk_factors": [],
            "risk_summary": "",
            "elevated_urgency": False,
        },
    )

    result = await clinical_module.clinical_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "头痛"}],
            "preconsultation_record": {},
            "clinical_facts": [],
            "workflow_control": {"graph_iteration": 1},
        },
        bus=AsyncStreamBus(),
        router=_RouterStub(),
        smart_chain=[],
        fast_chain=[],
        profile_snapshot=None,
        constraint_prompt="",
        session_id="s1",
    )

    assert result.goto == "output"
    assert result.update["workflow_control"]["next_node"] == "output"
    assert result.update["clinical_assessment"]["status"] == "complete"


@pytest.mark.asyncio
async def test_clinical_node_routes_with_command_to_intake(monkeypatch) -> None:
    async def fake_urgency(**kwargs):
        return UrgencyResult(
            level=UrgencyLevel.NORMAL,
            reason="普通就医",
            triggered_by_rule=False,
        )

    async def fake_differential(**kwargs):
        return [
            {
                "condition": "待评估",
                "likelihood": "medium",
                "reasoning": "信息不足",
                "supporting_symptoms": [],
                "risk_factors": [],
            }
        ]

    monkeypatch.setattr(clinical_module, "evaluate_urgency_by_llm", fake_urgency)
    monkeypatch.setattr(clinical_module, "_generate_differential", fake_differential)
    monkeypatch.setattr(
        clinical_module,
        "evaluate_risk_factors",
        lambda symptom_summary, profile_snapshot: {
            "risk_factors": [],
            "risk_summary": "",
            "elevated_urgency": False,
        },
    )

    result = await clinical_module.clinical_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "胸痛"}],
            "intake_plan": {"protocol_id": "chest_pain"},
            "preconsultation_record": {},
            "clinical_facts": [],
            "workflow_control": {"graph_iteration": 1},
        },
        bus=AsyncStreamBus(),
        router=_RouterStub(),
        smart_chain=[],
        fast_chain=[],
        profile_snapshot=None,
        constraint_prompt="",
        session_id="s1",
    )

    assert result.goto == "intake"
    assert result.update["workflow_control"]["next_node"] == "intake"
    assert result.update["clinical_assessment"]["status"] == "needs_more_info"
