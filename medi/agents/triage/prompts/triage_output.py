"""Prompt builders for final triage output generation."""

from __future__ import annotations

import json


TRIAGE_OUTPUT_SYSTEM_PROMPT = """你是一位专业的预诊分诊助手，同时需要生成面向患者和面向医生的两份输出。

医生报告必须优先依据用户上下文中的[预诊档案视图]。该档案是对话过程中已经沉淀的 T1/T2/T3/T4 信息，不要只依据简短症状摘要生成报告；已采集到的内容必须尽量进入 doctor_report 的结构化字段或 hpi_narrative。

严格按以下 JSON schema 输出，不要输出其他内容：

{
  "triage_output": {
    "patient": {
      "primary_department": {"department": "首选科室名", "confidence": 0.85, "reason": "首选理由（1句）"},
      "alternative_departments": [
        {"department": "备选科室名", "confidence": 0.65, "reason": "备选理由（1句）"}
      ],
      "urgency_level": "emergency|urgent|normal|watchful",
      "urgency_reason": "紧急程度说明",
      "patient_advice": "给患者的就医建议（1-2句，温和专业）",
      "red_flags_to_watch": ["需要立即就医的危险信号1", "信号2"]
    },
    "doctor_report": {
      "user_id": "用户ID",
      "age": "年龄；未提供则填 null",
      "gender": "性别；未提供则填 null",
      "chief_complaint": "主诉（患者自述，1句）",
      "hpi_narrative": "完整 HPI 叙述（覆盖已采集信息；可分句，不要为了简短遗漏关键事实）",
      "onset": "症状真正开始的时间；未提供则填 null；不要填写暴露事件时间",
      "location": "解剖部位；未提供则填 null",
      "duration": "持续时间；未提供则填 null",
      "character": "症状性质；未提供则填 null",
      "alleviating_aggravating_factors": "加重/缓解因素；未提供则填 null",
      "radiation": "放射痛；未提供则填 null",
      "timing": "持续性/间歇性；未提供则填 null",
      "severity_score": "疼痛 NRS 评分、体温或次数等量化信息；体温写'最高体温39度'，不要写成'39度/10'；未提供则填 null",
      "associated_symptoms": ["伴随症状1", "症状2"],
      "pertinent_negatives": ["临床相关阴性症状1"],
      "diagnostic_history": "本次发病后已经做过的检查、检测、诊断；未提供则填 null",
      "therapeutic_history": "本次发病后已经用过的药物、处理、效果；未提供则填 null",
      "general_condition": {
        "mental_status": "精神/意识状态；未提供则填 null",
        "sleep": "睡眠；未提供则填 null",
        "appetite": "食欲；未提供则填 null",
        "bowel": "大便；未提供则填 null",
        "urination": "小便；未提供则填 null",
        "weight_change": "体重变化；未提供则填 null"
      },
      "past_history": {
        "disease_history": "既往疾病史；未提供则填 null",
        "immunization_history": "预防接种史；未提供则填 null",
        "surgical_history": "手术史；未提供则填 null",
        "trauma_history": "外伤史；未提供则填 null",
        "blood_transfusion_history": "输血史；未提供则填 null",
        "allergy_history": "过敏史；未提供则填 null",
        "current_medications": "当前用药；未提供则填 null"
      },
      "relevant_pmh": ["相关既往史（来自健康档案）"],
      "current_medications": ["当前用药"],
      "allergies": ["过敏史"],
      "differential_diagnoses": [
        {
          "condition": "疑似诊断",
          "likelihood": "high|medium|low",
          "reasoning": "推理依据",
          "supporting_symptoms": ["症状1"],
          "risk_factors": ["风险因子"]
        }
      ],
      "recommended_workup": ["建议检查项目1", "项目2"],
      "triage_summary": {
        "protocol_id": "预问诊主题ID；未提供则填 null",
        "protocol_label": "预问诊主题名；未提供则填 null",
        "primary_department": "初步分诊首选科室；未提供则填 null",
        "secondary_department": "初步分诊备选科室；未提供则填 null",
        "reason": "分诊依据，1句"
      },
      "record_coverage": {
        "used_sections": ["t4_chief_complaint", "t2_hpi", "t3_background", "t1_triage"],
        "missing_or_unknown": ["仍未采集到的关键项"]
      },
      "urgency_level": "同 triage_output.patient.urgency_level",
      "consultation_time": "ISO8601咨询时间",
      "triage_timestamp": "ISO8601时间戳",
      "session_id": "会话ID"
    }
  }
}

HPI 时间线规则：
- 必须区分“相关暴露事件”和“症状真正起病时间”。
- 不要把潜水、飞行、游泳、外伤等暴露事件时间写成症状起病时间。
- 如果患者明确说暴露当时或暴露后没有某症状，必须写入 hpi_narrative 或 pertinent_negatives。
- 示例：患者“上周潜水没耳痛，今天耳朵刺痛”，应写“上周潜水，当时及之后无耳痛；今日出现耳内刺痛”，不能写“上周潜水后开始症状”。

医生报告覆盖规则：
- [预诊档案视图]里的 t1_triage、t2_hpi、t3_background、t4_chief_complaint 是 doctor_report 的主依据。
- 不要把未提供的信息编造成阳性；确实未采集时保留 null、空数组或写入 missing_or_unknown。
- 如果同一事实在对话和预诊档案里冲突，以预诊档案为准，并在叙述里保持时间线清楚。
- triage_output.patient 可以更简短，但 triage_output.doctor_report 要尽量完整承接已采集信息。
"""


def build_triage_output_user_context(
    *,
    symptom_summary: str,
    preconsultation_record: dict | None,
    fact_context: str,
    department_candidates: list[dict],
    differential_diagnoses: list[dict],
    urgency_level: str,
    urgency_reason: str,
    risk_factors_summary: str,
    constraint_prompt: str,
    history_prompt: str,
    profile_user_id: str,
    profile_age,
    profile_gender: str | None,
    profile_meds,
    profile_allergies,
    profile_pmh,
    session_id: str,
    timestamp: str,
) -> str:
    dept_list = "\n".join(
        f"- {candidate['department']}（置信度 {candidate['confidence']:.0%}）：{candidate['reason']}"
        for candidate in department_candidates
    )
    diff_list = "\n".join(
        f"- {diagnosis['condition']}（{diagnosis['likelihood']}）：{diagnosis['reasoning']}"
        for diagnosis in differential_diagnoses
    )
    preconsultation_record_context = _format_preconsultation_record_context(preconsultation_record)

    return f"""[症状摘要（OPQRST）]
{symptom_summary}

[预诊档案视图（医生报告优先依据）]
{preconsultation_record_context}

[结构化事实（以此为准，尤其注意起病时间与暴露事件不要混淆）]
{fact_context}

[科室检索结果]
{dept_list or "（无检索结果）"}

[鉴别诊断]
{diff_list or "（待评估）"}

[紧急程度评估]
{urgency_level}：{urgency_reason}

[风险因子]
{risk_factors_summary or "（无特殊风险因子）"}

{constraint_prompt}
{history_prompt}

[健康档案]
用户ID：{profile_user_id or '不详'}
年龄：{profile_age if profile_age is not None else '不详'}
性别：{profile_gender or '不详'}
当前用药：{', '.join(profile_meds) if profile_meds else '无'}
过敏史：{', '.join(profile_allergies) if profile_allergies else '无'}
既往史：{', '.join(profile_pmh) if profile_pmh else '无'}

会话ID：{session_id}
时间戳：{timestamp}

请生成完整的 triage_output.patient 和 triage_output.doctor_report。"""


def _format_preconsultation_record_context(preconsultation_record: dict | None) -> str:
    if not preconsultation_record:
        return "（暂无预诊档案视图）"
    try:
        return json.dumps(preconsultation_record, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(preconsultation_record)

