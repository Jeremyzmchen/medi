"""
TriageAgent 工具定义

把 DepartmentRouter 包装成标准 ToolDefinition，供 ToolRuntime 注册和执行。
同时提供 OpenAI function calling schema，注入 LLM 让其自主决定何时调用。
"""

from __future__ import annotations

from medi.core.tool_runtime import ToolDefinition, ToolPriority
from medi.agents.triage.department_router import DepartmentRouter
from medi.agents.triage.urgency_evaluator import evaluate_urgency_by_llm


# OpenAI function calling schema — 注入 LLM messages，让 LLM 知道有这个工具
SEARCH_SYMPTOM_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_symptom_kb",
        "description": (
            "在症状-科室知识库中检索，根据症状描述返回最匹配的科室候选列表。"
            "当用户的症状信息足够时调用此工具获取科室建议。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "症状描述文本，包含部位、性质、时间等信息",
                }
            },
            "required": ["query"],
        },
    },
}


def make_urgency_tool(call_with_fallback, fast_chain, bus, session_id: str, obs=None) -> ToolDefinition:
    """把 evaluate_urgency_by_llm 包装为 ToolDefinition"""

    async def evaluate_urgency(symptom_text: str) -> dict:
        result = await evaluate_urgency_by_llm(
            symptom_text=symptom_text,
            call_with_fallback=call_with_fallback,
            fast_chain=fast_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
        )
        return {
            "level": result.level.value,
            "reason": result.reason,
            "triggered_by_rule": result.triggered_by_rule,
        }

    return ToolDefinition(
        name="evaluate_urgency",
        priority=ToolPriority.STANDARD,
        fn=evaluate_urgency,
        description="LLM 紧急程度评估，规则层未命中时调用",
    )


def make_search_tool(router: DepartmentRouter) -> ToolDefinition:
    """
    把 DepartmentRouter.route() 包装为 ToolDefinition。
    返回值序列化为 dict，供 LLM Observe 阶段读取。
    """
    async def search_symptom_kb(query: str) -> dict:
        candidates = await router.route(query)
        return {
            "candidates": [
                {
                    "department": c.department,
                    "confidence": c.confidence,
                    "reason": c.reason,
                }
                for c in candidates
            ]
        }

    return ToolDefinition(
        name="search_symptom_kb",
        priority=ToolPriority.STANDARD,
        fn=search_symptom_kb,
        description="症状-科室向量知识库检索",
    )
