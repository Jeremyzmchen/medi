"""
Graph Builder

首轮 graph.ainvoke 从 intake 入口进入；追问时 inquirer 使用 LangGraph
interrupt 暂停图。下一轮用户回答由 runner 通过 Command(resume=...) 交还
给 inquirer，随后沿 inquirer -> intake 重新抽取事实。

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
from medi.agents.triage.graph.nodes.safety_gate_node import safety_gate_node
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
    profile_snapshot,
    constraint_prompt: str,
    history_prompt: str,
    episodic: EpisodicMemory,
    session_id: str,
    obs=None,
    checkpointer=None,
):
    # 不同节点绑定不同依赖

    bound_safety_gate = functools.partial(
        safety_gate_node,
        bus=bus,
        fast_chain=fast_chain,
        obs=obs,
    )

    bound_intake = functools.partial(
        intake_node,
        bus=bus,
        fast_chain=fast_chain,
        profile_snapshot=profile_snapshot,
        obs=obs,
    )

    bound_intake_monitor = functools.partial(
        intake_monitor_node,
        bus=bus,
        profile_snapshot=profile_snapshot,
    )

    bound_intake_controller = functools.partial(
        intake_controller_node,
        bus=bus,
        profile_snapshot=profile_snapshot,
    )

    bound_intake_prompter = functools.partial(
        intake_prompter_node,
        bus=bus,
        fast_chain=fast_chain,
        profile_snapshot=profile_snapshot,
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
        profile_snapshot=profile_snapshot,
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
        profile_snapshot=profile_snapshot,
        episodic=episodic,
        session_id=session_id,
        obs=obs,
    )

    builder = StateGraph(TriageGraphState)
    builder.add_node("safety_gate", bound_safety_gate, destinations=("intake", END))
    builder.add_node("intake", bound_intake)
    builder.add_node("monitor", bound_intake_monitor)
    builder.add_node("controller", bound_intake_controller)
    builder.add_node("prompter", bound_intake_prompter)
    builder.add_node("inquirer", bound_intake_inquirer)
    builder.add_node("clinical", bound_clinical)
    builder.add_node("output", bound_output)

    # 图节点入口
    builder.set_entry_point("safety_gate")

    # intake 永远路由到 monitor
    # safety_gate uses Command(goto=...) to continue to intake or stop at END.

    builder.add_edge("intake", "monitor")

    # monitor 永远路由到 controller
    builder.add_edge("monitor", "controller")

    # controller 条件判断边
    # controller returns Command(goto=...) to choose clinical or prompter.

    builder.add_edge("prompter", "inquirer")
    # inquirer pauses with interrupt(); after Command(resume=...) it appends the
    # patient's answer and routes back through safety_gate before fact extraction.
    builder.add_edge("inquirer", "safety_gate")
    
    # 判断科室前最多再走一轮询问
    # clinical returns Command(goto=...) to choose output or back-loop to intake.

    builder.add_edge("output", END)

    return builder.compile(checkpointer=checkpointer)
