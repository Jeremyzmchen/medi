"""
Graph Builder

每轮 ainvoke 完整跑一次图，无 interrupt/resume。
IntakeNode 返回 next_node="intake_wait" 时图走到 END（本轮结束，等待用户）。
IntakeNode 返回 next_node="clinical" 时继续到 ClinicalNode → OutputNode。
"""

from __future__ import annotations

import functools

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from medi.agents.triage.graph.state import TriageGraphState
from medi.agents.triage.graph.nodes.intake_node import intake_node
from medi.agents.triage.graph.nodes.clinical_node import clinical_node
from medi.agents.triage.graph.nodes.output_node import output_node
from medi.agents.triage.department_router import DepartmentRouter
from medi.core.stream_bus import AsyncStreamBus
from medi.memory.episodic import EpisodicMemory


def build_triage_graph(
    bus: AsyncStreamBus,
    router: DepartmentRouter,
    smart_chain: list,
    fast_chain: list,
    fast_model: str,
    health_profile,
    constraint_prompt: str,
    history_prompt: str,
    episodic: EpisodicMemory,
    session_id: str,
    obs=None,
):
    bound_intake = functools.partial(
        intake_node,
        bus=bus,
        fast_chain=fast_chain,
        fast_model=fast_model,
        obs=obs,
    )
    bound_clinical = functools.partial(
        clinical_node,
        bus=bus,
        router=router,
        smart_chain=smart_chain,
        fast_chain=fast_chain,
        health_profile=health_profile,
        constraint_prompt=constraint_prompt,
        session_id=session_id,
        obs=obs,
    )
    bound_output = functools.partial(
        output_node,
        bus=bus,
        smart_chain=smart_chain,
        constraint_prompt=constraint_prompt,
        history_prompt=history_prompt,
        health_profile=health_profile,
        episodic=episodic,
        session_id=session_id,
        obs=obs,
    )

    builder = StateGraph(TriageGraphState)
    builder.add_node("intake", bound_intake)
    builder.add_node("clinical", bound_clinical)
    builder.add_node("output", bound_output)

    builder.set_entry_point("intake")

    builder.add_conditional_edges(
        "intake",
        _route_from_intake,
        {
            "clinical": "clinical",
            "end": END,             # 本轮追问完毕，等待用户
        },
    )

    builder.add_conditional_edges(
        "clinical",
        _route_from_clinical,
        {
            "output": "output",
            "intake": "intake",
        },
    )

    builder.add_edge("output", END)

    return builder.compile()


def _route_from_intake(state: TriageGraphState) -> str:
    if state.get("intake_complete"):
        return "clinical"
    return "end"


def _route_from_clinical(state: TriageGraphState) -> str:
    next_node = state.get("next_node", "output")
    iteration = state.get("graph_iteration", 1)
    if next_node == "intake" and iteration < 2:
        return "intake"
    return "output"
