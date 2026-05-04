"""
临床推理工具

ClinicalNode 使用的工具：
  - evaluate_risk_factors：结合 ProfileSnapshot 评估患者特异性风险因子
"""

from __future__ import annotations

def evaluate_risk_factors(
    symptom_summary: str,
    profile_snapshot,
) -> dict:
    """
    将 ProfileSnapshot 中的慢性病史、用药史与当前症状进行交叉分析，
    返回患者特异性风险因子摘要。

    这比单纯注入 constraint_prompt 更精准：
    只返回与当前症状直接相关的风险因子，而非所有慢性病。

    Returns:
        {
            "risk_factors": ["高血压史（心血管症状需排除高血压急症）", ...],
            "risk_summary": "患者合并高血压，当前胸闷症状需优先排除心源性原因",
            "elevated_urgency": bool  # 是否因风险因子应提升紧急程度
        }
    """
    if profile_snapshot is None:
        return {
            "risk_factors": [],
            "risk_summary": "",
            "elevated_urgency": False,
        }

    full_text = symptom_summary or ""

    risk_factors: list[str] = []
    elevated_urgency = False

    conditions = getattr(profile_snapshot, "chronic_conditions", []) or []
    medications = getattr(profile_snapshot, "current_medications", []) or []
    age = getattr(profile_snapshot, "age", None)
    gender = getattr(profile_snapshot, "gender", None)

    # 心血管风险
    cardiac_keywords = ["胸", "心", "气短", "呼吸"]
    is_cardiac_symptom = any(kw in full_text for kw in cardiac_keywords)
    if is_cardiac_symptom:
        for cond in conditions:
            if any(kw in cond for kw in ["高血压", "心脏", "冠心病", "心律失常"]):
                risk_factors.append(f"{cond}史（心血管症状需排除相关急症）")
                elevated_urgency = True
        if age and age >= 60:
            risk_factors.append(f"年龄 {age} 岁（老年患者心血管风险较高）")

    # 消化系统风险
    gi_keywords = ["腹", "胃", "肠", "恶心", "呕"]
    is_gi_symptom = any(kw in full_text for kw in gi_keywords)
    if is_gi_symptom:
        for cond in conditions:
            if any(kw in cond for kw in ["糖尿病", "胃溃疡", "肝", "胆"]):
                risk_factors.append(f"{cond}史（消化症状可能与基础疾病相关）")
        for med in medications:
            if any(kw in med for kw in ["阿司匹林", "布洛芬", "消炎药", "NSAIDs"]):
                risk_factors.append(f"正在使用 {med}（可能引起消化道不适）")

    # 神经系统风险
    neuro_keywords = ["头", "眩晕", "麻木", "无力", "视力"]
    is_neuro_symptom = any(kw in full_text for kw in neuro_keywords)
    if is_neuro_symptom:
        for cond in conditions:
            if any(kw in cond for kw in ["高血压", "糖尿病", "房颤", "脑"]):
                risk_factors.append(f"{cond}史（神经症状需排除脑血管事件）")
                elevated_urgency = True

    # 药物过敏相关
    allergies = getattr(profile_snapshot, "allergies", []) or []
    if allergies:
        risk_factors.append(f"已知过敏史：{', '.join(allergies)}（用药建议需规避）")

    # 构建摘要
    if risk_factors:
        risk_summary = f"患者存在以下风险因子，建议医生重点关注：{'；'.join(risk_factors[:3])}"
    else:
        risk_summary = ""

    return {
        "risk_factors": risk_factors,
        "risk_summary": risk_summary,
        "elevated_urgency": elevated_urgency,
    }

