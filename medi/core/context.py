"""
UnifiedContext — 贯穿所有模块的共享上下文

与 Weave 的 UnifiedContext 的区别：
- 增加 HealthProfile 作为硬约束字段
- 增加 DialogueState 跟踪对话状态机
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medi.memory.health_profile import HealthProfile
    from medi.core.providers import LLMProvider


class DialogueState(Enum):
    """分诊对话状态机"""
    INIT = "init"
    COLLECTING = "collecting"       # 追问中，信息不足
    SUFFICIENT = "sufficient"       # 信息足够，准备检索
    SEARCHING = "searching"         # 检索知识库中
    RESPONDING = "responding"       # 生成建议中
    ESCALATING = "escalating"       # 发现红旗症状，立即升级
    DONE = "done"


@dataclass
class ModelConfig:
    # 模型名（供日志/显示用，实际调用走 provider chain）
    fast: str = "gpt-4o-mini"
    smart: str = "gpt-4o"
    # 跨供应商降级链（运行时由 build_smart/fast_chain() 填充）
    smart_chain: list[LLMProvider] = field(default_factory=list)
    fast_chain: list[LLMProvider] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 延迟导入避免循环依赖
        from medi.core.providers import build_smart_chain, build_fast_chain
        if not self.smart_chain:
            self.smart_chain = build_smart_chain()
        if not self.fast_chain:
            self.fast_chain = build_fast_chain()


@dataclass
class UnifiedContext:
    """
    所有模块共享的上下文，不逐层传参。

    health_profile 作为硬约束注入每次 LLM 调用，
    区别于 Weave 中记忆的软约束（洞察）语义。
    """
    user_id: str
    session_id: str

    # 对话状态机（Medi 特有）
    dialogue_state: DialogueState = DialogueState.INIT
    follow_up_count: int = 0          # 已追问轮数，上限 3

    # 用户健康档案（Medi 特有，硬约束）
    health_profile: HealthProfile | None = None

    # 模型配置
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # 工具权限
    enabled_tools: set[str] = field(default_factory=set)

    # 会话内对话历史（传给 LLM，保留完整上下文）
    messages: list[dict] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def transition(self, new_state: DialogueState) -> None:
        self.dialogue_state = new_state

    def can_follow_up(self) -> bool:
        return self.follow_up_count < 3

    def increment_follow_up(self) -> None:
        self.follow_up_count += 1

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.enabled_tools

    def build_constraint_prompt(self) -> str:
        """
        将健康档案转为硬约束 prompt 片段，注入 system prompt。
        Weave 的记忆是软约束（供参考），这里是强制约束（必须遵守）。
        """
        if self.health_profile is None:
            return ""

        p = self.health_profile
        lines = ["[用户健康约束 - 必须严格遵守，不得忽略]"]

        if p.allergies:
            lines.append(f"- 过敏史：{', '.join(p.allergies)}（涉及药物建议时必须过滤）")
        if p.chronic_conditions:
            lines.append(f"- 慢性病：{', '.join(p.chronic_conditions)}（影响科室优先级判断）")
        if p.current_medications:
            lines.append(f"- 当前用药：{', '.join(p.current_medications)}（避免药物冲突建议）")
        if p.age:
            lines.append(f"- 年龄：{p.age}岁，性别：{p.gender or '未知'}")

        return "\n".join(lines)
