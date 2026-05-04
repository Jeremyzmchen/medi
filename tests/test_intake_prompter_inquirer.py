import pytest

from medi.agents.triage.graph.nodes.intake_inquirer_node import intake_inquirer_node
from medi.agents.triage.graph.nodes.intake_prompter_node import (
    intake_prompter_node,
)
from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.prompts.intake_prompter import build_prompter_input
from medi.core.stream_bus import AsyncStreamBus


def test_prompter_input_is_task_focused() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "头痛"}])
    prompt = build_prompter_input(
        task_id="T2_ONSET",
        task_instruction="推进任务：Onset。",
        plan=plan,
        store=ClinicalFactStore(),
        preconsultation_record={},
        task_progress={
            "T2_ONSET": {
                "score": 0.0,
                "status": "pending",
                "requirement_details": [
                    {"id": "onset_time", "description": "Time is documented.", "completed": False}
                ],
            }
        },
        messages=[{"role": "user", "content": "我头痛"}],
    )

    assert "当前采集任务" in prompt
    assert "Onset" in prompt
    assert "当前缺口" in prompt
    assert "Time is documented." in prompt
    assert "hpi.onset" not in prompt


@pytest.mark.asyncio
async def test_prompter_node_falls_back_without_llm_chain() -> None:
    result = await intake_prompter_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "我头痛"}],
            "task_board": {
                "controller": {
                    "next_best_task": "T2_ONSET",
                    "task_instruction": "推进任务：Onset。",
                },
                "progress": {"T2_ONSET": {"score": 0.0, "status": "pending"}},
            },
            "preconsultation_record": {},
        },
        bus=AsyncStreamBus(),
        fast_chain=[],
    )

    decision = result["task_board"]["controller"]
    assert decision["next_best_task"] == "T2_ONSET"
    assert decision["next_best_question"] == "这个症状是什么时候开始的？"
    assert result["workflow_control"]["next_node"] == "inquirer"


@pytest.mark.asyncio
async def test_inquirer_node_delivers_question_and_records_attempt() -> None:
    result = await intake_inquirer_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "我头痛"}],
            "task_board": {
                "controller": {
                    "next_best_task": "T2_ONSET",
                    "next_best_question": "这个症状是什么时候开始的？",
                },
                "task_rounds": {},
            },
        },
        bus=AsyncStreamBus(),
    )

    assert result["messages"] == [
        {"role": "assistant", "content": "这个症状是什么时候开始的？"}
    ]
    assert result["task_board"]["task_rounds"] == {"T2_ONSET": 1}
    assert result["workflow_control"]["next_node"] == "intake_wait"

