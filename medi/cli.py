"""
Medi CLI 入口

用法：
  medi                    # 开始分诊对话（guest 用户）
  medi --user-id u1       # 指定用户（加载健康档案）
"""

from __future__ import annotations

import asyncio
import uuid

import typer
from rich.console import Console
from rich.markdown import Markdown

from medi.core.context import UnifiedContext, ModelConfig
from medi.core.stream_bus import AsyncStreamBus, EventType
from medi.agents.triage.agent import TriageAgent
from medi.agents.triage.department_router import DepartmentRouter
from medi.agents.orchestrator import OrchestratorAgent, Intent
from medi.agents.triage.symptom_collector import SymptomInfo
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
        profile.allergies = [a.strip() for a in allergies_input.split("，") if a.strip()]
        # 兼容英文逗号
        if not profile.allergies:
            profile.allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]

    chronic_input = console.input("慢性病史（如：高血压、糖尿病，多个用逗号分隔）：").strip()
    if chronic_input:
        profile.chronic_conditions = [c.strip() for c in chronic_input.split("，") if c.strip()]
        if not profile.chronic_conditions:
            profile.chronic_conditions = [c.strip() for c in chronic_input.split(",") if c.strip()]

    meds_input = console.input("当前用药（如：二甲双胍，多个用逗号分隔）：").strip()
    if meds_input:
        profile.current_medications = [m.strip() for m in meds_input.split("，") if m.strip()]
        if not profile.current_medications:
            profile.current_medications = [m.strip() for m in meds_input.split(",") if m.strip()]

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

    ctx = UnifiedContext(
        user_id=user_id,
        session_id=session_id,
        model_config=ModelConfig(),
        enabled_tools={"search_symptom_kb", "evaluate_urgency", "get_department_info"},
        health_profile=profile,
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
                complaint = r.chief_complaint[:30].replace("\n", " ")
                console.print(f"[dim]  {date_str} | {r.department} | {complaint}[/dim]")
    console.print()

    async def handle_turn(user_input: str) -> None:
        nonlocal bus
        bus = AsyncStreamBus()
        agent._bus = bus
        orchestrator._bus = bus

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
                # 新主诉：清空对话历史和症状信息，重新开始分诊
                ctx.messages.clear()
                agent._symptom_info = SymptomInfo()
                await agent.handle(user_input)
            else:
                # SYMPTOM / MEDICATION → 路由到对应 Agent
                # Phase 2: MEDICATION → MedicationAgent，目前统一走 TriageAgent
                await agent.handle(user_input)

            await bus.close()

        await asyncio.gather(consume(), produce())

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


@app.command()
def chat(
    user_id: str = typer.Option("guest", "--user-id", "-u", help="用户 ID"),
) -> None:
    """开始分诊对话"""
    asyncio.run(_chat_loop(user_id))


if __name__ == "__main__":
    app()
