"""
OutputNode — 双向输出节点

职责：
  1. 生成患者侧输出：科室建议、紧急程度、就医指引（PatientOutput）
  2. 生成医生侧输出：结构化 HPI 摘要（DoctorHPI JSON）
  3. 通过 AsyncStreamBus 发布 RESULT 事件（含两个结构化字段）
  4. 持久化到 EpisodicMemory（visit_records）

双向输出通过单次 LLM 调用（JSON 模式）生成，保证一致性。
结果以结构化 JSON 返回，patient_output / doctor_hpi 分别挂在 RESULT 事件的 data 上。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from medi.agents.triage.graph.state import (
    TriageGraphState,
    PatientOutput,
    DoctorHPI,
)
from medi.agents.triage.graph.nodes.clinical_node import _build_symptom_summary
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback
from medi.memory.episodic import EpisodicMemory


_OUTPUT_SYSTEM_PROMPT = """你是一位专业的预诊分诊助手，同时需要生成面向患者和面向医生的两份输出。

严格按以下 JSON schema 输出，不要输出其他内容：

{
  "patient_output": {
    "recommended_departments": [
      {"department": "科室名", "confidence": 0.85, "reason": "推荐理由（1句）"}
    ],
    "urgency_level": "emergency|urgent|normal|watchful",
    "urgency_reason": "紧急程度说明",
    "patient_advice": "给患者的就医建议（1-2句，温和专业）",
    "red_flags_to_watch": ["需要立即就医的危险信号1", "信号2"]
  },
  "doctor_hpi": {
    "chief_complaint": "主诉（患者自述，1句）",
    "hpi_narrative": "完整 HPI 叙述（OLDCARTS 格式，100字以内）",
    "onset": "发作时间/诱因或null",
    "location": "解剖部位或null",
    "duration": "持续时间或null",
    "character": "症状性质或null",
    "alleviating_aggravating_factors": "加重/缓解因素或null",
    "radiation": "放射痛或null",
    "timing": "持续性/间歇性或null",
    "severity_score": "疼痛 NRS 评分、体温或次数等量化信息；体温写'最高体温39度'，不要写成'39度/10'；没有则null",
    "associated_symptoms": ["伴随症状1", "症状2"],
    "pertinent_negatives": ["临床相关阴性症状1"],
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
    "urgency_level": "同 patient_output.urgency_level",
    "triage_timestamp": "ISO8601时间戳",
    "session_id": "会话ID"
  }
}"""


async def output_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    smart_chain: list,
    constraint_prompt: str,
    history_prompt: str,
    health_profile,       # HealthProfile | None
    episodic: EpisodicMemory,
    session_id: str,
    obs=None,
) -> dict:
    """OutputNode 执行函数"""
    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "output"},
        session_id=session_id,
    ))

    symptom_data = state.get("symptom_data") or {}
    symptom_summary = _build_symptom_summary(symptom_data)

    department_candidates = state.get("department_candidates") or []
    urgency_level = state.get("urgency_level") or "normal"
    urgency_reason = state.get("urgency_reason") or ""
    differential_diagnoses = state.get("differential_diagnoses") or []
    risk_factors_summary = state.get("risk_factors_summary") or ""

    # ── 构建 context ──
    dept_list = "\n".join(
        f"- {c['department']}（置信度 {c['confidence']:.0%}）：{c['reason']}"
        for c in department_candidates
    )
    diff_list = "\n".join(
        f"- {d['condition']}（{d['likelihood']}）：{d['reasoning']}"
        for d in differential_diagnoses
    )
    profile_meds = getattr(health_profile, "current_medications", []) or []
    profile_allergies = getattr(health_profile, "allergies", []) or []
    profile_pmh = getattr(health_profile, "chronic_conditions", []) or []

    timestamp = datetime.now(timezone.utc).isoformat()

    user_context = f"""[症状摘要（OPQRST）]
{symptom_summary}

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
当前用药：{', '.join(profile_meds) if profile_meds else '无'}
过敏史：{', '.join(profile_allergies) if profile_allergies else '无'}
既往史：{', '.join(profile_pmh) if profile_pmh else '无'}

会话ID：{session_id}
时间戳：{timestamp}

请生成完整的 patient_output 和 doctor_hpi。"""

    messages = state.get("messages") or []

    # ── LLM 单次调用，生成双向输出 ──
    patient_output, doctor_hpi = await _generate_dual_output(
        system_prompt=_OUTPUT_SYSTEM_PROMPT,
        conversation_messages=messages,
        user_context=user_context,
        smart_chain=smart_chain,
        bus=bus,
        session_id=session_id,
        obs=obs,
        # 降级数据（LLM 失败时使用）
        fallback_departments=department_candidates,
        fallback_urgency=urgency_level,
        fallback_urgency_reason=urgency_reason,
        fallback_differentials=differential_diagnoses,
        profile_meds=profile_meds,
        profile_allergies=profile_allergies,
        profile_pmh=profile_pmh,
        timestamp=timestamp,
        session_id_val=session_id,
    )

    # ── 发布 RESULT 事件（含结构化字段） ──
    await bus.emit(StreamEvent(
        type=EventType.RESULT,
        data={
            "content": patient_output["patient_advice"],
            "patient_output": patient_output,
            "doctor_hpi": doctor_hpi,
        },
        session_id=session_id,
    ))

    # ── 持久化到 EpisodicMemory ──
    # 优先用 LLM 生成的 patient_output 里的科室（经过推理修正），
    # 而非直接用向量检索的 department_candidates（可能匹配偏差）
    llm_depts = patient_output.get("recommended_departments") or []
    top_department = (
        llm_depts[0]["department"] if llm_depts
        else (department_candidates[0]["department"] if department_candidates else "待确认")
    )
    await episodic.save(
        symptom_summary=symptom_summary,
        advice=patient_output["patient_advice"],
        department=top_department,
    )

    return {
        "patient_output": patient_output,
        "doctor_hpi": doctor_hpi,
        "next_node": "done",
    }


async def _generate_dual_output(
    system_prompt: str,
    conversation_messages: list[dict],
    user_context: str,
    smart_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    obs=None,
    # fallback params
    fallback_departments=None,
    fallback_urgency="normal",
    fallback_urgency_reason="",
    fallback_differentials=None,
    profile_meds=None,
    profile_allergies=None,
    profile_pmh=None,
    timestamp="",
    session_id_val="",
) -> tuple[PatientOutput, DoctorHPI]:
    """单次 LLM 调用生成 patient_output + doctor_hpi"""
    try:
        llm_messages = (
            [{"role": "system", "content": system_prompt}]
            + [m for m in conversation_messages if m.get("role") in ("user", "assistant")]
            + [{"role": "user", "content": user_context}]
        )

        response = await call_with_fallback(
            chain=smart_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
            call_type="output_dual",
            messages=llm_messages,
            max_tokens=1500,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        patient_output = _parse_patient_output(parsed.get("patient_output", {}))
        doctor_hpi = _parse_doctor_hpi(parsed.get("doctor_hpi", {}))
        return patient_output, doctor_hpi

    except Exception:
        # LLM 或解析失败：用已有结构化数据构建降级输出
        return (
            _fallback_patient_output(
                fallback_departments or [],
                fallback_urgency,
                fallback_urgency_reason,
            ),
            _fallback_doctor_hpi(
                fallback_departments or [],
                fallback_differentials or [],
                profile_meds or [],
                profile_allergies or [],
                profile_pmh or [],
                fallback_urgency,
                timestamp,
                session_id_val,
            ),
        )


def _parse_patient_output(data: dict) -> PatientOutput:
    depts = [
        {"department": d.get("department", ""), "confidence": float(d.get("confidence", 0)), "reason": d.get("reason", "")}
        for d in data.get("recommended_departments", [])
    ]
    return PatientOutput(
        recommended_departments=depts,
        urgency_level=data.get("urgency_level", "normal"),
        urgency_reason=data.get("urgency_reason", ""),
        patient_advice=data.get("patient_advice", "建议尽快就诊。"),
        red_flags_to_watch=list(data.get("red_flags_to_watch") or []),
    )


def _parse_doctor_hpi(data: dict) -> DoctorHPI:
    diffs = []
    for d in data.get("differential_diagnoses") or []:
        diffs.append({
            "condition": d.get("condition", ""),
            "likelihood": d.get("likelihood", "medium"),
            "reasoning": d.get("reasoning", ""),
            "supporting_symptoms": list(d.get("supporting_symptoms") or []),
            "risk_factors": list(d.get("risk_factors") or []),
        })
    return DoctorHPI(
        chief_complaint=data.get("chief_complaint", ""),
        hpi_narrative=data.get("hpi_narrative", ""),
        onset=data.get("onset"),
        location=data.get("location"),
        duration=data.get("duration"),
        character=data.get("character"),
        alleviating_aggravating_factors=data.get("alleviating_aggravating_factors"),
        radiation=data.get("radiation"),
        timing=data.get("timing"),
        severity_score=_clean_severity_score(data.get("severity_score")),
        associated_symptoms=list(data.get("associated_symptoms") or []),
        pertinent_negatives=list(data.get("pertinent_negatives") or []),
        relevant_pmh=list(data.get("relevant_pmh") or []),
        current_medications=list(data.get("current_medications") or []),
        allergies=list(data.get("allergies") or []),
        differential_diagnoses=diffs,
        recommended_workup=list(data.get("recommended_workup") or []),
        urgency_level=data.get("urgency_level", "normal"),
        triage_timestamp=data.get("triage_timestamp", ""),
        session_id=data.get("session_id", ""),
    )


def _fallback_patient_output(departments, urgency, urgency_reason) -> PatientOutput:
    return PatientOutput(
        recommended_departments=departments[:3],
        urgency_level=urgency,
        urgency_reason=urgency_reason or "已完成初步评估",
        patient_advice="建议就诊上述科室，请携带相关病历资料。",
        red_flags_to_watch=["症状突然加重", "出现新症状"],
    )


def _fallback_doctor_hpi(
    departments, differentials, meds, allergies, pmh,
    urgency, timestamp, session_id
) -> DoctorHPI:
    return DoctorHPI(
        chief_complaint="见症状摘要",
        hpi_narrative="患者经预诊系统采集，详见结构化症状数据。",
        onset=None, location=None, duration=None, character=None,
        alleviating_aggravating_factors=None, radiation=None,
        timing=None, severity_score=None,
        associated_symptoms=[],
        pertinent_negatives=[],
        relevant_pmh=pmh,
        current_medications=meds,
        allergies=allergies,
        differential_diagnoses=differentials,
        recommended_workup=[],
        urgency_level=urgency,
        triage_timestamp=timestamp,
        session_id=session_id,
    )


def _clean_severity_score(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:℃|度)\s*/\s*10", r"\1度", text)
    return text
