from functools import partial

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from medi.agents.triage.graph.nodes.intake_inquirer_node import intake_inquirer_node
from medi.agents.triage.graph.state import TriageGraphState
from medi.agents.triage.runner import GraphThreadStatus, TriageGraphRunner
from medi.core.context import UnifiedContext
from medi.core.stream_bus import AsyncStreamBus


def _runner(session_id: str, checkpointer: MemorySaver) -> TriageGraphRunner:
    return TriageGraphRunner(
        ctx=UnifiedContext(user_id="u1", session_id=session_id),
        bus=AsyncStreamBus(),
        checkpointer=checkpointer,
    )


@pytest.mark.asyncio
async def test_runner_recovers_interrupted_thread_from_checkpoint() -> None:
    checkpointer = MemorySaver()
    bus = AsyncStreamBus()
    builder = StateGraph(TriageGraphState)
    builder.add_node("inquirer", partial(intake_inquirer_node, bus=bus))
    builder.set_entry_point("inquirer")
    builder.add_edge("inquirer", END)
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "s-interrupt"}}

    await graph.ainvoke(
        {
            "session_id": "s-interrupt",
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

    restarted_runner = _runner("s-interrupt", checkpointer)
    status = await restarted_runner.inspect_thread_state()
    context = await restarted_runner.resume_context()

    assert status == GraphThreadStatus.INTERRUPTED
    assert context["status"] == "interrupted"
    assert context["pending_question"] == "When did this symptom start?"
    assert context["pending_task"] == "T2_ONSET"
    assert context["recommended_action"] == "answer"
    assert "continue" in context["actions"]


@pytest.mark.asyncio
async def test_runner_recovers_active_thread_from_checkpoint() -> None:
    checkpointer = MemorySaver()

    def seed_active(state: TriageGraphState) -> dict:
        return {
            "messages": [{"role": "user", "content": "I still feel nauseous"}],
            "clinical_facts": [{"slot": "hpi.associated_symptoms", "value": "nausea"}],
        }

    builder = StateGraph(TriageGraphState)
    builder.add_node("seed", seed_active)
    builder.set_entry_point("seed")
    builder.add_edge("seed", END)
    graph = builder.compile(checkpointer=checkpointer)
    await graph.ainvoke(
        {"session_id": "s-active", "user_id": "u1"},
        config={"configurable": {"thread_id": "s-active"}},
    )

    restarted_runner = _runner("s-active", checkpointer)

    assert await restarted_runner.inspect_thread_state() == GraphThreadStatus.ACTIVE
    assert restarted_runner.symptom_summary()


@pytest.mark.asyncio
async def test_runner_recognizes_completed_and_reset_deletes_checkpoint() -> None:
    checkpointer = MemorySaver()

    def seed_completed(state: TriageGraphState) -> dict:
        return {
            "triage_output": {
                "meta": {},
                "patient": {},
                "doctor_report": {},
            },
        }

    builder = StateGraph(TriageGraphState)
    builder.add_node("seed", seed_completed)
    builder.set_entry_point("seed")
    builder.add_edge("seed", END)
    graph = builder.compile(checkpointer=checkpointer)
    await graph.ainvoke(
        {"session_id": "s-complete", "user_id": "u1"},
        config={"configurable": {"thread_id": "s-complete"}},
    )

    restarted_runner = _runner("s-complete", checkpointer)
    assert await restarted_runner.inspect_thread_state() == GraphThreadStatus.COMPLETED

    await restarted_runner.reset_graph_state()

    assert await restarted_runner.inspect_thread_state() == GraphThreadStatus.EMPTY
