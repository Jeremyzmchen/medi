"""
SymptomCollector — 症状收集器（OPQRST 标准）

OPQRST 是急诊医学和分诊中广泛使用的症状采集框架：
  O - Onset         发作时间与诱因（什么时候开始？做什么时发生的？）
  P - Provocation   加重/缓解因素（什么使症状变重或变轻？）
  Q - Quality       症状性质（刺痛、钝痛、压迫感、烧灼感？）
  R - Region/Radiation 部位与放射（哪里不舒服？有没有放射痛？）
  S - Severity      严重程度（0-10 分评分）
  T - Time          时间特征（持续多久？持续性还是间歇性？）

充分标准（最低要求）：
  - R（部位）必须有
  - T（时间）或 O（发作诱因）至少有一个
  三轮追问结束后无论信息是否充分都继续（避免过度追问）

LLM 追问：不使用硬编码模板，让 LLM 根据已有信息和缺失字段生成自然的追问。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI


@dataclass
class SymptomInfo:
    """当前会话收集到的症状信息（OPQRST）"""
    raw_descriptions: list[str] = field(default_factory=list)  # 用户原始描述

    # OPQRST 字段
    onset: str | None = None           # O - 发作时间与诱因
    provocation: str | None = None     # P - 加重/缓解因素
    quality: str | None = None         # Q - 症状性质（刺痛/钝痛/烧灼等）
    region: str | None = None          # R - 部位与放射痛
    severity: str | None = None        # S - 严重程度（0-10）
    time_pattern: str | None = None    # T - 时间特征（持续/间歇，多久）

    accompanying: list[str] = field(default_factory=list)   # 伴随症状（NER 提取）

    def is_sufficient(self) -> bool:
        """
        充分标准：
          - R（部位）必须有
          - T（时间）或 O（发作诱因）至少有一个
        """
        has_region = bool(self.region)
        has_time_or_onset = bool(self.time_pattern or self.onset)
        return has_region and has_time_or_onset

    def missing_fields(self) -> list[str]:
        """返回缺失的 OPQRST 字段名（优先级排序）"""
        missing = []
        if not self.region:
            missing.append("R")      # 部位 — 最关键
        if not self.time_pattern and not self.onset:
            missing.append("T/O")    # 时间/诱因 — 第二优先
        if not self.quality:
            missing.append("Q")      # 性质
        if not self.severity:
            missing.append("S")      # 严重程度
        if not self.provocation:
            missing.append("P")      # 加重缓解
        return missing

    def to_query_text(self) -> str:
        """拼接为知识库检索用的文本"""
        parts = self.raw_descriptions.copy()
        if self.region:
            parts.append(self.region)
        if self.time_pattern:
            parts.append(self.time_pattern)
        if self.onset:
            parts.append(self.onset)
        if self.quality:
            parts.append(self.quality)
        parts.extend(self.accompanying)
        return " ".join(parts)

    def to_summary(self) -> str:
        """生成结构化症状摘要，供 _respond 阶段注入 prompt"""
        lines = []
        if self.region:
            lines.append(f"部位：{self.region}")
        if self.time_pattern:
            lines.append(f"时间：{self.time_pattern}")
        if self.onset:
            lines.append(f"诱因：{self.onset}")
        if self.quality:
            lines.append(f"性质：{self.quality}")
        if self.severity:
            lines.append(f"程度：{self.severity}/10")
        if self.provocation:
            lines.append(f"加重/缓解：{self.provocation}")
        if self.accompanying:
            lines.append(f"伴随症状：{', '.join(self.accompanying)}")
        return "\n".join(lines) if lines else "（无结构化信息）"


_FIELD_LABELS = {
    "R":   "不适部位（哪里不舒服）",
    "T/O": "发作时间与诱因（什么时候开始，做什么时发生的）",
    "Q":   "症状性质（刺痛、钝痛、压迫感、烧灼感等）",
    "S":   "严重程度（0-10分，10分最严重）",
    "P":   "加重或缓解因素（什么使症状变重或变轻）",
}


async def build_follow_up_question(
    missing: list[str],
    symptom_info: SymptomInfo,
    client: AsyncOpenAI,
    fast_model: str = "gpt-4o-mini",
) -> str:
    """
    用 LLM 根据已有症状信息和缺失字段生成自然的追问。
    missing 只传最高优先级的 1-2 个字段，避免一次追问太多。
    """
    # 最多追问 2 个字段，保持对话自然
    top_missing = missing[:2]
    missing_desc = "、".join(_FIELD_LABELS[f] for f in top_missing if f in _FIELD_LABELS)

    known_info = symptom_info.to_summary()

    response = await client.chat.completions.create(
        model=fast_model,
        max_tokens=80,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位分诊护士，正在和患者对话收集症状信息。\n"
                    "根据已知信息和需要补充的字段，用温和、简洁的语气生成一句追问。\n"
                    "只问最重要的问题，不要列清单，不要问超过两件事。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[已知症状信息]\n{known_info}\n\n"
                    f"[需要补充的信息]\n{missing_desc}\n\n"
                    "请生成一句自然的追问："
                ),
            },
        ],
    )

    return response.choices[0].message.content.strip()
