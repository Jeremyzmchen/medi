"""
Medi CLI 入口

用法：
  medi chat              # 开始分诊对话
  medi chat --user-id u1 # 指定用户（加载健康档案）
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

app = typer.Typer(name="medi", help="Medi 智能健康 Agent")
console = Console()


async def _chat_loop(user_id: str) -> None:
    session_id = str(uuid.uuid4())[:8]

    ctx = UnifiedContext(
        user_id=user_id,
        session_id=session_id,
        model_config=ModelConfig(),
        enabled_tools={"search_symptom_kb", "evaluate_urgency", "get_department_info"},
    )
    bus = AsyncStreamBus()
    agent = TriageAgent(ctx=ctx, bus=bus)

    console.print(f"\n[bold green]Medi 分诊助手[/bold green] (会话 {session_id})")
    console.print("请描述您的症状，输入 [bold]quit[/bold] 退出\n")

    async def consume_events() -> None:
        async for event in bus.stream():
            if event.type == EventType.FOLLOW_UP:
                console.print(f"\n[cyan]Medi:[/cyan] {event.data['question']}")
            elif event.type == EventType.RESULT:
                console.print("\n[cyan]Medi:[/cyan]")
                console.print(Markdown(event.data["content"]))
            elif event.type == EventType.ESCALATION:
                console.print(f"\n[bold red]警告:[/bold red] {event.data['reason']}")
            elif event.type == EventType.DONE:
                break

    while True:
        try:
            user_input = console.input("\n[bold]您:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        event_task = asyncio.create_task(consume_events())
        await agent.handle(user_input)
        await bus.close()
        await event_task

        if ctx.dialogue_state.value == "done":
            break

    console.print("\n[dim]会话结束[/dim]")


@app.command()
def chat(
    user_id: str = typer.Option("guest", "--user-id", "-u", help="用户 ID"),
) -> None:
    """开始分诊对话"""
    asyncio.run(_chat_loop(user_id))


if __name__ == "__main__":
    app()
