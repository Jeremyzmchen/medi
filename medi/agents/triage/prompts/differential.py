"""Prompt builders for triage differential diagnosis generation."""

from __future__ import annotations


def build_differential_prompt(
    symptom_summary: str,
    department_candidates: list[dict],
    risk_factors_summary: str,
    constraint_prompt: str,
) -> str:
    """
    构建鉴别诊断 LLM prompt。

    要求 LLM 以 JSON 格式输出 differential_diagnoses 列表，
    每项包含 condition / likelihood / reasoning / supporting_symptoms / risk_factors。
    """
    dept_list = "\n".join(
        f"- {candidate['department']}（置信度 {candidate['confidence']:.0%}）"
        for candidate in department_candidates[:3]
    )

    risk_section = f"\n[患者风险因子]\n{risk_factors_summary}" if risk_factors_summary else ""

    return f"""你是一位经验丰富的临床医生，正在进行初步鉴别诊断分析。

{constraint_prompt}

[症状摘要]
{symptom_summary}

[科室检索结果]
{dept_list}
{risk_section}

请基于以上信息，给出 2-4 个鉴别诊断，严格按以下 JSON 格式输出（不要输出其他内容）：

{{
  "differential_diagnoses": [
    {{
      "condition": "疑似诊断名称",
      "likelihood": "high|medium|low",
      "reasoning": "推理依据（1-2句）",
      "supporting_symptoms": ["支持该诊断的症状1", "症状2"],
      "risk_factors": ["患者特异性风险因子（如有）"]
    }}
  ]
}}

注意：
- likelihood 只能是 high / medium / low 三个值之一
- 按可能性从高到低排列
- 不做最终诊断，只做初步鉴别分析供医生参考"""

