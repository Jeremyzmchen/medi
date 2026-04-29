"""
Medi CLI 入口

用法：
  medi                    # 开始分诊对话（guest 用户）
  medi --user-id u1       # 指定用户（加载健康档案）
  medi observe            # 查看最近请求的可观测性数据
  medi observe --session <id>  # 查看单个 session 的完整链路
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from medi.core.context import UnifiedContext, ModelConfig
from medi.core.stream_bus import AsyncStreamBus, EventType
from medi.core.observability import ObservabilityStore, query_recent_sessions, query_session_detail
from medi.agents.triage.agent import TriageAgent
from medi.agents.triage.department_router import DepartmentRouter
from medi.agents.orchestrator import OrchestratorAgent, Intent
from medi.agents.triage.symptom_collector import SymptomInfo
from medi.agents.medication.agent import MedicationAgent
from medi.agents.health_report.agent import HealthReportAgent
from medi.memory.health_profile import HealthProfile, load_profile, save_profile
from medi.memory.episodic import EpisodicMemory

app = typer.Typer(name="medi", help="Medi 智能健康 Agent", invoke_without_command=True)
console = Console()


async def _collect_profile(user_id: str) -> HealthProfile:
    """首次使用时引导用户填写健康档案"""
    console.print("\n[yellow]您是第一次使用 Medi，请先完善您的健康档案（更准确的分诊建议）[/yellow]")
    console.print("[dim]直接回车可跳过某项[/dim]\n")

    profile = HealthProfile(user_id=user_id)

    age_input = console.input("年龄：").strip()
    if age_input.isdigit():
        profile.age = int(age_input)

    gender_input = console.input("性别（男/女）：").strip()
    if gender_input in ("男", "女"):
        profile.gender = gender_input

    allergies_input = console.input("过敏史（如：青霉素、磺胺，多个用逗号分隔）：").strip()
    if allergies_input:
        profile.allergies = [a.strip() for a in allergies_input.replace("，", ",").split(",") if a.strip()]

    chronic_input = console.input("慢性病史（如：高血压、糖尿病，多个用逗号分隔）：").strip()
    if chronic_input:
        profile.chronic_conditions = [c.strip() for c in chronic_input.replace("，", ",").split(",") if c.strip()]

    meds_input = console.input("当前用药（如：二甲双胍，多个用逗号分隔）：").strip()
    if meds_input:
        profile.current_medications = [m.strip() for m in meds_input.replace("，", ",").split(",") if m.strip()]

    await save_profile(profile)
    console.print("\n[green]档案已保存[/green]")
    return profile


async def _chat_loop(user_id: str) -> None:
    session_id = str(uuid.uuid4())[:8]

    # 加载健康档案，首次使用引导填写
    profile = await load_profile(user_id)
    if not profile.is_complete() and user_id != "guest":
        profile = await _collect_profile(user_id)
    elif profile.is_complete():
        console.print(f"\n[dim]已加载健康档案：{profile.gender}，{profile.age}岁[/dim]")

    obs = ObservabilityStore()
    ctx = UnifiedContext(
        user_id=user_id,
        session_id=session_id,
        model_config=ModelConfig(),
        enabled_tools={"search_symptom_kb", "evaluate_urgency", "get_department_info"},
        health_profile=profile,
        observability=obs,
    )
    router = DepartmentRouter()
    bus = AsyncStreamBus()
    orchestrator = OrchestratorAgent(ctx=ctx, bus=bus)
    agent = TriageAgent(
        ctx=ctx,
        bus=bus,
        router=router,
        on_result=orchestrator.update_last_response,
    )
    medication_agent = MedicationAgent(ctx=ctx, bus=bus)
    health_report_agent = HealthReportAgent(ctx=ctx, bus=bus)

    console.print(f"\n[bold green]Medi 分诊助手[/bold green] (会话 {session_id})")
    console.print("请描述您的症状，输入 [bold]quit[/bold] 退出")

    # 展示最近就诊记录
    if user_id != "guest":
        episodic = EpisodicMemory(user_id)
        recent = await episodic.recent(limit=3)
        if recent:
            console.print("\n[dim]最近就诊记录：[/dim]")
            for r in recent:
                date_str = r.visit_date.strftime("%m-%d")
                console.print(f"[dim]  {date_str} | {r.department}[/dim]")
                if r.chief_complaint:
                    fields = [line.strip() for line in r.chief_complaint.splitlines() if line.strip()]
                    detail = " | ".join(fields)
                    console.print(f"[dim]       {detail}[/dim]")
    console.print()

    async def handle_turn(user_input: str) -> None:
        nonlocal bus
        bus = AsyncStreamBus()
        agent._bus = bus
        orchestrator._bus = bus
        medication_agent._bus = bus
        health_report_agent._bus = bus
        health_report_agent._diet_agent._bus = bus
        health_report_agent._schedule_agent._bus = bus

        async def consume() -> None:
            async for event in bus.stream():
                if event.type == EventType.FOLLOW_UP:
                    console.print(f"\n[cyan]Medi:[/cyan] {event.data['question']}")
                elif event.type == EventType.RESULT:
                    console.print("\n[cyan]Medi:[/cyan]")
                    console.print(Markdown(event.data["content"]))
                elif event.type == EventType.ESCALATION:
                    console.print(f"\n[bold red]警告:[/bold red] {event.data['reason']}")

        async def produce() -> None:
            symptom_summary = agent._symptom_info.to_summary()
            intent = await orchestrator.classify_intent(user_input, symptom_summary)

            if intent == Intent.OUT_OF_SCOPE:
                await orchestrator.handle_out_of_scope()
            elif intent == Intent.FOLLOWUP:
                await orchestrator.handle_followup(user_input)
            elif intent == Intent.NEW_SYMPTOM:
                ctx.messages.clear()
                agent._symptom_info = SymptomInfo()
                await agent.handle(user_input)
            elif intent == Intent.MEDICATION:
                await medication_agent.handle(user_input)
            elif intent == Intent.HEALTH_REPORT:
                await health_report_agent.handle(user_input)
            else:
                await agent.handle(user_input)

            await bus.close()

        await asyncio.gather(consume(), produce())

        # 每轮结束后 flush trace 到 SQLite
        await obs.flush()

    while True:
        try:
            user_input = console.input("\n[bold]您:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        await handle_turn(user_input)

    console.print("\n[dim]会话结束[/dim]")


async def _show_observe(session_id: Optional[str]) -> None:
    if session_id:
        detail = await query_session_detail(session_id)
        _print_session_detail(session_id, detail)
    else:
        sessions = await query_recent_sessions(limit=10)
        _print_sessions_summary(sessions)


def _print_sessions_summary(sessions: list[dict]) -> None:
    if not sessions:
        console.print("[dim]暂无可观测性数据（先运行 medi 对话）[/dim]")
        return

    table = Table(title="最近 10 个会话")
    table.add_column("Session", style="cyan")
    table.add_column("时间", style="dim")
    table.add_column("LLM 调用", justify="right")
    table.add_column("总 Token", justify="right")
    table.add_column("总耗时(ms)", justify="right")
    table.add_column("降级次数", justify="right", style="yellow")
    table.add_column("错误次数", justify="right", style="red")

    for s in sessions:
        table.add_row(
            s["session_id"],
            s["start_time"][:16],
            str(s["llm_calls"]),
            str(s["total_tokens"] or 0),
            str(s["total_llm_ms"] or 0),
            str(s["fallback_count"] or 0),
            str(s["error_count"] or 0),
        )

    console.print(table)
    console.print("[dim]用 --session <id> 查看单个会话详情[/dim]")


def _print_session_detail(session_id: str, detail: dict) -> None:
    console.print(f"\n[bold]Session {session_id} 链路详情[/bold]\n")

    if detail["stages"]:
        table = Table(title="TAOR 阶段耗时")
        table.add_column("阶段")
        table.add_column("耗时(ms)", justify="right")
        for s in detail["stages"]:
            table.add_row(s["stage"], str(s["latency_ms"]))
        console.print(table)

    if detail["llm_calls"]:
        table = Table(title="LLM 调用记录")
        table.add_column("调用类型")
        table.add_column("供应商")
        table.add_column("降级", justify="center")
        table.add_column("Prompt Token", justify="right")
        table.add_column("Completion Token", justify="right")
        table.add_column("耗时(ms)", justify="right")
        table.add_column("成功", justify="center")
        for c in detail["llm_calls"]:
            table.add_row(
                c.get("call_type", "-"),
                c["provider"],
                "Y" if c["is_fallback"] else "-",
                str(c["prompt_tokens"]),
                str(c["completion_tokens"]),
                str(c["latency_ms"]),
                "Y" if c["success"] else "[red]N[/red]",
            )
        console.print(table)

    if detail["tool_calls"]:
        table = Table(title="工具调用记录")
        table.add_column("工具名")
        table.add_column("耗时(ms)", justify="right")
        table.add_column("成功", justify="center")
        table.add_column("错误")
        for t in detail["tool_calls"]:
            table.add_row(
                t["tool_name"],
                str(t["latency_ms"]),
                "Y" if t["success"] else "[red]N[/red]",
                t["error_msg"] or "-",
            )
        console.print(table)


@app.command()
def chat(
    user_id: str = typer.Option("guest", "--user-id", "-u", help="用户 ID"),
) -> None:
    """开始分诊对话"""
    asyncio.run(_chat_loop(user_id))


@app.command()
def observe(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="查看指定 session 的详情"),
) -> None:
    """查看可观测性数据（LLM 调用、工具调用、阶段耗时）"""
    asyncio.run(_show_observe(session))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    reload: bool = typer.Option(False, "--reload", help="开发模式：代码变更自动重载"),
) -> None:
    """启动 FastAPI HTTP 服务（支持 SSE 流式对话）"""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]缺少 uvicorn，请运行：pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Medi API[/bold green] 启动于 http://{host}:{port}")
    console.print(f"[dim]文档：http://{host}:{port}/docs[/dim]")
    uvicorn.run(
        "medi.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
