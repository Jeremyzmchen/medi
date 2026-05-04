from functools import partial
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from medi.agents.triage.graph.nodes.intake_inquirer_node import intake_inquirer_node
from medi.agents.triage.graph.nodes.intake_prompter_node import (
    intake_prompter_node,
)
from medi.agents.triage.graph.state import TriageGraphState
from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.prompts.intake_prompter import build_prompter_input
from medi.agents.triage.runner import TriageGraphRunner
from medi.core.context import DialogueState, UnifiedContext
from medi.core.stream_bus import AsyncStreamBus, EventType


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
async def test_inquirer_node_interrupts_then_records_answer_on_resume() -> None:
    bus = AsyncStreamBus()
    builder = StateGraph(TriageGraphState)
    builder.add_node("inquirer", partial(intake_inquirer_node, bus=bus))
    builder.set_entry_point("inquirer")
    builder.add_edge("inquirer", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s1"}}

    first = await graph.ainvoke(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "headache"}],
            "task_board": {
                "controller": {
                    "next_best_task": "T2_ONSET",
                    "next_best_question": "When did this symptom start?",
                },
                "task_rounds": {},
            },
        },
        config=config,
    )

    payload = first["__interrupt__"][0].value
    assert payload["question"] == "When did this symptom start?"
    assert payload["task"] == "T2_ONSET"
    assert first["messages"] == [
        {"role": "user", "content": "headache"},
    ]

    resumed = await graph.ainvoke(Command(resume="since yesterday"), config=config)

    assert resumed["messages"] == [
        {"role": "user", "content": "headache"},
        {"role": "assistant", "content": "When did this symptom start?"},
        {"role": "user", "content": "since yesterday"},
    ]
    assert resumed["task_board"]["task_rounds"] == {"T2_ONSET": 1}
    assert resumed["workflow_control"]["next_node"] == "intake"


@pytest.mark.asyncio
async def test_runner_emits_follow_up_from_interrupt_payload() -> None:
    ctx = UnifiedContext(user_id="u1", session_id="s1")
    bus = AsyncStreamBus()
    queue = bus._make_queue()
    runner = TriageGraphRunner(ctx=ctx, bus=bus)

    handled = await runner._handle_interrupt({
        "__interrupt__": [
            SimpleNamespace(value={
                "question": "When did this symptom start?",
                "task": "T2_ONSET",
                "round": 1,
            })
        ]
    })

    stage_event = await queue.get()
    follow_up_event = await queue.get()

    assert handled is True
    assert runner._pending_interrupt is True
    assert ctx.dialogue_state == DialogueState.INTAKE_WAITING
    assert ctx.messages[-1] == {
        "role": "assistant",
        "content": "When did this symptom start?",
    }
    assert stage_event.type == EventType.STAGE_START
    assert follow_up_event.type == EventType.FOLLOW_UP
    assert follow_up_event.data == {
        "question": "When did this symptom start?",
        "round": 1,
        "task": "T2_ONSET",
    }

