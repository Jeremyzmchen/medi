"""
症状采集工具

IntakeNode 使用的工具：
  - check_symptom_pattern：识别症状所属的临床模式，返回针对性追问优先级
  - symptom_data_to_info：将 SymptomData TypedDict 转为 SymptomInfo dataclass（供现有逻辑复用）
  - symptom_info_to_data：反向转换
"""

from __future__ import annotations

from medi.agents.triage.symptom_collector import SymptomInfo
from medi.agents.triage.graph.state import SymptomData


# 症状模式 → 优先追问字段映射
# 不同模式的临床信息需求不同：
#   cardiac：放射痛 + 诱因比性质更重要
#   gi：饮食关联比时间特征更重要
#   neuro：发作模式（持续/间歇）最关键
_PATTERN_PRIORITIES: dict[str, list[str]] = {
    "cardiac":        ["P", "R", "T/O", "Q", "S"],
    "gi":             ["T/O", "P", "Q", "R", "S"],
    "neuro":          ["T", "Q", "R", "P", "S"],
    "musculoskeletal": ["T/O", "P", "R", "Q", "S"],
    "respiratory":    ["T", "Q", "P", "S", "R"],
    "default":        ["R", "T/O", "Q", "S", "P"],
}

# 症状关键词 → 临床模式
_KEYWORD_TO_PATTERN: dict[str, str] = {
    # 心血管
    "胸痛": "cardiac", "胸闷": "cardiac", "心悸": "cardiac",
    "心跳": "cardiac", "胸部": "cardiac",
    # 消化
    "腹痛": "gi", "腹泻": "gi", "恶心": "gi", "呕吐": "gi",
    "消化": "gi", "胃": "gi", "肠": "gi", "腹部": "gi",
    # 神经
    "头痛": "neuro", "头晕": "neuro", "眩晕": "neuro",
    "麻木": "neuro", "无力": "neuro", "抽搐": "neuro",
    # 肌肉骨骼
    "关节": "musculoskeletal", "腰痛": "musculoskeletal",
    "背痛": "musculoskeletal", "颈痛": "musculoskeletal",
    "扭伤": "musculoskeletal", "骨折": "musculoskeletal",
    # 呼吸
    "咳嗽": "respiratory", "咳痰": "respiratory", "气短": "respiratory",
    "哮喘": "respiratory", "呼吸": "respiratory",
}


def check_symptom_pattern(symptom_data: SymptomData) -> dict:
    """
    识别症状的临床模式，返回针对性的追问优先级。

    Returns:
        {
            "pattern": "cardiac" | "gi" | "neuro" | ...,
            "priority_fields": ["P", "R", ...],  # 该模式下最重要的缺失字段
            "pattern_hint": "心血管症状模式"      # 供调试/日志
        }
    """
    descriptions = symptom_data.get("raw_descriptions", [])
    region = symptom_data.get("region") or ""
    full_text = " ".join(descriptions) + " " + region

    # 关键词匹配确定模式
    detected_pattern = "default"
    for keyword, pattern in _KEYWORD_TO_PATTERN.items():
        if keyword in full_text:
            detected_pattern = pattern
            break

    priority_fields = _PATTERN_PRIORITIES[detected_pattern]

    pattern_labels = {
        "cardiac": "心血管症状模式",
        "gi": "消化系统症状模式",
        "neuro": "神经系统症状模式",
        "musculoskeletal": "肌肉骨骼症状模式",
        "respiratory": "呼吸系统症状模式",
        "default": "通用症状模式",
    }

    return {
        "pattern": detected_pattern,
        "priority_fields": priority_fields,
        "pattern_hint": pattern_labels[detected_pattern],
    }


def symptom_info_to_data(info: SymptomInfo) -> SymptomData:
    """SymptomInfo dataclass → SymptomData TypedDict（用于写入 graph state）"""
    return SymptomData(
        raw_descriptions=list(info.raw_descriptions),
        onset=info.onset,
        provocation=info.provocation,
        quality=info.quality,
        region=info.region,
        severity=info.severity,
        time_pattern=info.time_pattern,
        accompanying=list(info.accompanying),
    )


def symptom_data_to_info(data: SymptomData) -> SymptomInfo:
    """SymptomData TypedDict → SymptomInfo dataclass（供现有提取逻辑复用）"""
    info = SymptomInfo()
    info.raw_descriptions = list(data.get("raw_descriptions") or [])
    info.onset = data.get("onset")
    info.provocation = data.get("provocation")
    info.quality = data.get("quality")
    info.region = data.get("region")
    info.severity = data.get("severity")
    info.time_pattern = data.get("time_pattern")
    info.accompanying = list(data.get("accompanying") or [])
    return info
