from medi.agents.triage.graph.nodes.intake_controller_node import (
    intake_controller_node,
    _decide_finish,
    _select_next_task,
    _task_instruction,
)
from medi.core.stream_bus import AsyncStreamBus
import pytest


def test_controller_selects_highest_value_pending_task() -> None:
    progress = {
        "T1_PRIMARY_DEPARTMENT": {
            "task_id": "T1_PRIMARY_DEPARTMENT",
            "base_priority": 100,
            "critical": True,
            "score": 0.5,
            "requirement_details": [],
        },
        "T2_ONSET": {
            "task_id": "T2_ONSET",
            "base_priority": 85,
            "critical": False,
            "score": 0.0,
            "requirement_details": [],
        },
    }

    task_id, priority_score = _select_next_task(
        pending_tasks=["T1_PRIMARY_DEPARTMENT", "T2_ONSET"],
        task_progress=progress,
        task_rounds={},
    )

    assert task_id == "T2_ONSET"
    assert priority_score == 85.0


def test_controller_skips_chief_complaint_generation_during_inquiry() -> None:
    progress = {
        "T4_CHIEF_COMPLAINT_GENERATION": {
            "task_id": "T4_CHIEF_COMPLAINT_GENERATION",
            "base_priority": 10,
            "critical": False,
            "score": 0.0,
            "requirement_details": [],
        },
        "T3_ALLERGY_HISTORY": {
            "task_id": "T3_ALLERGY_HISTORY",
            "base_priority": 65,
            "critical": True,
            "score": 0.0,
            "requirement_details": [],
        },
    }

    task_id, _ = _select_next_task(
        pending_tasks=["T4_CHIEF_COMPLAINT_GENERATION", "T3_ALLERGY_HISTORY"],
        task_progress=progress,
        task_rounds={},
    )

    assert task_id == "T3_ALLERGY_HISTORY"


def test_task_instruction_uses_missing_requirement_descriptions() -> None:
    progress = {
        "T2_ONSET": {
            "task_id": "T2_ONSET",
            "score": 0.33,
            "requirement_details": [
                {"id": "onset_time", "description": "Time is documented.", "completed": True},
                {"id": "onset_trigger", "description": "Trigger is documented.", "completed": False},
            ],
        },
    }

    instruction = _task_instruction("T2_ONSET", progress)

    assert "Onset" in instruction
    assert "Trigger is documented." in instruction


def test_controller_does_not_use_preferred_exit_when_safety_or_red_flags_missing() -> None:
    progress = {
        "T3_CURRENT_MEDICATIONS": {
            "task_id": "T3_CURRENT_MEDICATIONS",
            "critical": True,
            "score": 0.0,
        },
        "T2_GENERAL_CONDITION": {
            "task_id": "T2_GENERAL_CONDITION",
            "critical": False,
            "score": 0.33,
        },
    }

    can_finish, reason = _decide_finish(
        selected_task="T3_CURRENT_MEDICATIONS",
        pending_tasks=["T3_CURRENT_MEDICATIONS", "T2_GENERAL_CONDITION"],
        task_progress=progress,
        assistant_count=8,
        max_rounds=10,
        preferred_limit=8,
        monitor_score=68,
        red_flags_checked=False,
        safety_slots_covered=False,
        doctor_summary_ready=False,
    )

    assert can_finish is False
    assert "Current Medications" in reason


@pytest.mark.asyncio
async def test_controller_routes_with_command_to_prompter() -> None:
    result = await intake_controller_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "headache"}],
            "task_board": {
                "monitor": {
                    "score": 20,
                    "red_flags_checked": False,
                    "safety_slots_covered": False,
                    "doctor_summary_ready": False,
                },
                "progress": {
                    "T2_ONSET": {
                        "task_id": "T2_ONSET",
                        "base_priority": 85,
                        "critical": False,
                        "score": 0.0,
                        "requirement_details": [],
                    },
                },
                "pending_tasks": ["T2_ONSET"],
                "task_rounds": {},
            },
            "workflow_control": {"graph_iteration": 0},
        },
        bus=AsyncStreamBus(),
    )

    assert result.goto == "prompter"
    assert result.update["workflow_control"]["next_node"] == "prompter"
    assert result.update["task_board"]["current_task"] == "T2_ONSET"


@pytest.mark.asyncio
async def test_controller_routes_with_command_to_clinical() -> None:
    result = await intake_controller_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "headache"}],
            "task_board": {
                "monitor": {
                    "score": 95,
                    "red_flags_checked": True,
                    "safety_slots_covered": True,
                    "doctor_summary_ready": True,
                },
                "progress": {
                    "T2_ONSET": {
                        "task_id": "T2_ONSET",
                        "base_priority": 85,
                        "critical": False,
                        "score": 1.0,
                        "requirement_details": [],
                    },
                },
                "pending_tasks": [],
                "task_rounds": {},
            },
            "workflow_control": {"graph_iteration": 0},
        },
        bus=AsyncStreamBus(),
    )

    assert result.goto == "clinical"
    assert result.update["workflow_control"]["next_node"] == "clinical"
    assert result.update["task_board"]["current_task"] is None
