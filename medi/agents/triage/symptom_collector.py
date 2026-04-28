"""
SymptomCollector — 症状收集器

负责判断症状信息是否充分，并生成追问。
最多追问 3 轮，避免用户体验差。

充分的症状信息需包含：
- 症状部位
- 持续时间
- 伴随症状（恶心、发烧等）
- 症状性质（持续/间歇、加重/减轻）
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SymptomInfo:
    """当前会话收集到的症状信息"""
    raw_descriptions: list[str] = field(default_factory=list)  # 用户原始描述
    location: str | None = None          # 症状部位
    duration: str | None = None          # 持续时间
    accompanying: list[str] = field(default_factory=list)       # 伴随症状
    nature: str | None = None            # 症状性质

    def is_sufficient(self) -> bool:
        """判断信息是否足够进行科室路由"""
        return bool(self.location and self.duration)

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.location:
            missing.append("症状部位")
        if not self.duration:
            missing.append("持续时间")
        return missing

    def to_query_text(self) -> str:
        """拼接为知识库检索用的文本"""
        parts = self.raw_descriptions.copy()
        if self.location:
            parts.append(self.location)
        if self.duration:
            parts.append(self.duration)
        parts.extend(self.accompanying)
        return " ".join(parts)


def build_follow_up_question(missing: list[str]) -> str:
    """根据缺失字段生成追问"""
    if "症状部位" in missing and "持续时间" in missing:
        return "请问您的不适主要在哪个部位？大概持续多久了？"
    if "症状部位" in missing:
        return "请问您的不适主要在哪个部位？"
    if "持续时间" in missing:
        return "这个症状大概持续多久了？"
    return "还有其他伴随症状吗？比如发烧、恶心等？"
