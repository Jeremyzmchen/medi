"""
TriageAgent 工具定义

把 DepartmentRouter 包装成标准 ToolDefinition，供 ToolRuntime 注册和执行。
同时提供 OpenAI function calling schema，注入 LLM 让其自主决定何时调用。
"""

from __future__ import annotations

from medi.core.tool_runtime import ToolDefinition, ToolPriority
from medi.agents.triage.department_router import DepartmentRouter


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
