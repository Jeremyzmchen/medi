import asyncio

from medi.agents.orchestrator import Intent, OrchestratorAgent
from medi.core.context import DialogueState, UnifiedContext
from medi.core.stream_bus import AsyncStreamBus


def test_intake_waiting_state_routes_short_reply_to_symptom() -> None:
    ctx = UnifiedContext(user_id="u1", session_id="s1")
    ctx.transition(DialogueState.INTAKE_WAITING)
    orchestrator = OrchestratorAgent(ctx=ctx, bus=AsyncStreamBus())

    intent = asyncio.run(orchestrator.classify_intent("？？"))

    assert intent == Intent.SYMPTOM
