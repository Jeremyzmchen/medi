"""
会话存储 — 内存级 Session 状态管理

每个 session 持有：
  - UnifiedContext（对话历史、状态机、健康档案）
  - TriageAgent / MedicationAgent / OrchestratorAgent（已初始化，可复用）

生命周期：服务进程内存，重启后丢失（后期可换 Redis 序列化）

并发安全：asyncio 单线程模型，不需要锁。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from medi.core.context import UnifiedContext, ModelConfig
from medi.core.observability import ObservabilityStore
from medi.core.stream_bus import AsyncStreamBus
from medi.agents.triage.runner import TriageGraphRunner
from medi.agents.triage.department_router import DepartmentRouter
from medi.agents.orchestrator import OrchestratorAgent
from medi.agents.medication.agent import MedicationAgent
from medi.agents.health_report.agent import HealthReportAgent
from medi.memory.health_profile import HealthProfile, load_profile


@dataclass
class Session:
    session_id: str
    ctx: UnifiedContext
    agent: TriageGraphRunner
    orchestrator: OrchestratorAgent
    medication_agent: MedicationAgent
    health_report_agent: HealthReportAgent
    obs: ObservabilityStore


# 全局 session 字典：session_id → Session
_sessions: dict[str, Session] = {}

_router = DepartmentRouter()


async def get_or_create_session(session_id: str | None, user_id: str) -> Session:
    """
    获取已有 session 或新建一个。

    首轮：session_id=None，生成新 ID，加载健康档案，初始化所有 Agent。
    后续轮：session_id 已存在，直接返回（对话历史在 ctx.messages 里）。
    """
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    # 新 session
    new_id = session_id or str(uuid.uuid4())[:8]
    profile = await load_profile(user_id)

    obs = ObservabilityStore()
    ctx = UnifiedContext(
        user_id=user_id,
        session_id=new_id,
        model_config=ModelConfig(),
        enabled_tools={"search_symptom_kb", "evaluate_urgency", "get_department_info"},
        health_profile=profile,
        observability=obs,
    )

    # Bus 在每轮请求时重建，这里只做占位
    bus = AsyncStreamBus()

    orchestrator = OrchestratorAgent(ctx=ctx, bus=bus)
    agent = TriageGraphRunner(
        ctx=ctx,
        bus=bus,
        router=_router,
    )
    medication_agent = MedicationAgent(ctx=ctx, bus=bus)
    health_report_agent = HealthReportAgent(ctx=ctx, bus=bus)

    session = Session(
        session_id=new_id,
        ctx=ctx,
        agent=agent,
        orchestrator=orchestrator,
        medication_agent=medication_agent,
        health_report_agent=health_report_agent,
        obs=obs,
    )
    _sessions[new_id] = session
    return session


def rebind_bus(session: Session, bus: AsyncStreamBus) -> None:
    """每轮请求开始时，给所有 Agent 换新的 Bus（旧 Bus 已关闭）"""
    session.agent._bus = bus  # TriageGraphRunner._bus
    session.orchestrator._bus = bus
    session.medication_agent._bus = bus
    session.health_report_agent._bus = bus
    session.health_report_agent._diet_agent._bus = bus
    session.health_report_agent._schedule_agent._bus = bus
    session.ctx.observability = session.obs  # obs 跨轮复用
