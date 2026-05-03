"""
Graph Builder

每轮 graph.ainvoke 都从 intake 入口跑一次图；跨轮状态由 runner 注入的
MemorySaver checkpointer 按 session_id/thread_id 保存和恢复，不使用 LangGraph
interrupt/resume。

流程：
- intake：抽取并合并预诊事实
- monitor：评估预诊档案完整度和缺失槽位
- controller：决定是否继续追问；若需要追问则选择下一任务
- prompter：根据当前任务生成自然追问
- inquirer：emit FOLLOW_UP，并让本轮图走到 END
- clinical：完成科室路由、紧急程度和鉴别诊断；必要时最多 back-loop 一次回到 intake
- output：生成患者侧建议和医生侧 HPI，随后结束图
"""

from __future__ import annotations

import functools

from langgraph.graph import StateGraph, END

from medi.agents.triage.graph.state import TriageGraphState
from medi.agents.triage.graph.nodes.intake_node import intake_node
from medi.agents.triage.graph.nodes.intake_monitor_node import intake_monitor_node
from medi.agents.triage.graph.nodes.intake_controller_node import intake_controller_node
from medi.agents.triage.graph.nodes.intake_prompter_node import intake_prompter_node
from medi.agents.triage.graph.nodes.intake_inquirer_node import intake_inquirer_node
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
    health_profile,
    constraint_prompt: str,
    history_prompt: str,
    episodic: EpisodicMemory,
    session_id: str,
    obs=None,
):
    # 不同节点绑定不同依赖

    bound_intake = functools.partial(
        intake_node,
        bus=bus,
        fast_chain=fast_chain,
        health_profile=health_profile,
        obs=obs,
    )

    bound_intake_monitor = functools.partial(
        intake_monitor_node,
        bus=bus,
        health_profile=health_profile,
    )

    bound_intake_controller = functools.partial(
        intake_controller_node,
        bus=bus,
        health_profile=health_profile,
    )

    bound_intake_prompter = functools.partial(
        intake_prompter_node,
        bus=bus,
        fast_chain=fast_chain,
        health_profile=health_profile,
        obs=obs,
    )

    bound_intake_inquirer = functools.partial(
        intake_inquirer_node,
        bus=bus,
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
    builder.add_node("monitor", bound_intake_monitor)
    builder.add_node("controller", bound_intake_controller)
    builder.add_node("prompter", bound_intake_prompter)
    builder.add_node("inquirer", bound_intake_inquirer)
    builder.add_node("clinical", bound_clinical)
    builder.add_node("output", bound_output)

    # 图节点入口
    builder.set_entry_point("intake")

    # intake 永远路由到 monitor
    builder.add_edge("intake", "monitor")

    # monitor 永远路由到 controller
    builder.add_edge("monitor", "controller")

    # controller 条件判断边
    builder.add_conditional_edges(
        "controller",
        _route_from_controller,
        {
            "clinical": "clinical",
            "prompter": "prompter",
        },
    )

    builder.add_edge("prompter", "inquirer")
    builder.add_edge("inquirer", END)
    
    # 判断科室前最多再走一轮询问
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


def _route_from_controller(state: TriageGraphState) -> str:
    if state.get("intake_complete"):
        return "clinical"
    return "prompter"


def _route_from_clinical(state: TriageGraphState) -> str:
    next_node = state.get("next_node", "output")
    iteration = state.get("graph_iteration", 1)
    # 最多再回去问一次
    if next_node == "intake" and iteration < 2:
        return "intake"
    return "output"
