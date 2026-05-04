from types import SimpleNamespace

import pytest
from langgraph.graph import END

from medi.agents.triage.graph.nodes.safety_gate_node import safety_gate_node
from medi.core.stream_bus import AsyncStreamBus, EventType


class _FakeProvider:
    def __init__(self, content: str) -> None:
        self._content = content

    @property
    def name(self) -> str:
        return "fake"

    async def create(self, messages: list[dict], max_tokens: int, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                )
            ],
            usage=None,
        )


@pytest.mark.asyncio
async def test_safety_gate_passes_non_emergency_message_to_intake() -> None:
    bus = AsyncStreamBus()
    queue = bus._make_queue()

    result = await safety_gate_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "headache"}],
            "workflow_control": {"graph_iteration": 0},
        },
        bus=bus,
    )

    stage_event = await queue.get()

    assert result.goto == "intake"
    assert result.update["safety_gate"]["status"] == "passed"
    assert result.update["workflow_control"]["next_node"] == "intake"
    assert stage_event.type == EventType.STAGE_START
    assert stage_event.data == {"stage": "safety_gate"}


@pytest.mark.asyncio
async def test_safety_gate_blocks_red_flag_message() -> None:
    bus = AsyncStreamBus()
    queue = bus._make_queue()

    result = await safety_gate_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "\u6211\u73b0\u5728\u80f8\u75db"}],
            "workflow_control": {"graph_iteration": 0},
        },
        bus=bus,
    )

    stage_event = await queue.get()
    escalation_event = await queue.get()
    result_event = await queue.get()

    assert result.goto == END
    assert result.update["safety_gate"]["status"] == "blocked"
    assert result.update["safety_gate"]["urgency_level"] == "emergency"
    assert result.update["triage_output"]["meta"]["source"] == "safety_gate"
    assert stage_event.type == EventType.STAGE_START
    assert escalation_event.type == EventType.ESCALATION
    assert result_event.type == EventType.RESULT
    assert "\u6025\u8bca\u79d1\u5ba4\u6302\u53f7" in result_event.data["content"]


@pytest.mark.asyncio
async def test_safety_gate_blocks_semantic_emergency_message() -> None:
    bus = AsyncStreamBus()
    queue = bus._make_queue()

    result = await safety_gate_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "\u547c\u5438\u82e6\u96be\u4e86"}],
            "workflow_control": {"graph_iteration": 0},
        },
        bus=bus,
        fast_chain=[
            _FakeProvider(
                '{"decision":"block","urgency_level":"emergency",'
                '"risk_concept":"respiratory_distress","confidence":0.91,'
                '"reason":"User appears to describe current breathing difficulty."}'
            )
        ],
    )

    await queue.get()
    escalation_event = await queue.get()
    result_event = await queue.get()

    assert result.goto == END
    assert result.update["safety_gate"]["status"] == "blocked"
    assert result.update["safety_gate"]["method"] == "llm"
    assert result.update["safety_gate"]["risk_concept"] == "respiratory_distress"
    assert escalation_event.type == EventType.ESCALATION
    assert result_event.type == EventType.RESULT
    assert "\u6b63\u5728\u4e3a\u60a8\u8fde\u63a5\u6025\u8bca\u79d1\u5ba4\u6302\u53f7" in result_event.data["content"]


@pytest.mark.asyncio
async def test_safety_gate_does_not_block_negated_semantic_message() -> None:
    bus = AsyncStreamBus()

    result = await safety_gate_node(
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "\u6211\u6ca1\u6709\u547c\u5438\u56f0\u96be"}],
            "workflow_control": {"graph_iteration": 0},
        },
        bus=bus,
        fast_chain=[
            _FakeProvider(
                '{"decision":"pass","urgency_level":"normal",'
                '"risk_concept":null,"confidence":0.88,'
                '"reason":"The symptom is explicitly negated."}'
            )
        ],
    )

    assert result.goto == "intake"
    assert result.update["safety_gate"]["status"] == "passed"
    assert result.update["safety_gate"]["method"] == "llm"
