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
    DepartmentResult,
)
from medi.agents.triage.intake_facts import FactStore
from medi.agents.triage.symptom_utils import build_symptom_summary
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback
from medi.memory.episodic import EpisodicMemory


_OUTPUT_SYSTEM_PROMPT = """你是一位专业的预诊分诊助手，同时需要生成面向患者和面向医生的两份输出。

医生报告必须优先依据用户上下文中的[结构化病历草稿]。该草稿是对话过程中已经沉淀的 CC/HPI/PH/Triage 信息，不要只依据简短症状摘要生成报告；已采集到的内容必须尽量进入 doctor_hpi 的结构化字段或 hpi_narrative。

严格按以下 JSON schema 输出，不要输出其他内容：

{
  "patient_output": {
    "primary_department": {"department": "首选科室名", "confidence": 0.85, "reason": "首选理由（1句）"},
    "alternative_departments": [
      {"department": "备选科室名", "confidence": 0.65, "reason": "备选理由（1句）"}
    ],
    "urgency_level": "emergency|urgent|normal|watchful",
    "urgency_reason": "紧急程度说明",
    "patient_advice": "给患者的就医建议（1-2句，温和专业）",
    "red_flags_to_watch": ["需要立即就医的危险信号1", "信号2"]
  },
  "doctor_hpi": {
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
      "used_sections": ["cc", "hpi", "ph", "triage"],
      "missing_or_unknown": ["仍未采集到的关键项"]
    },
    "urgency_level": "同 patient_output.urgency_level",
    "consultation_time": "ISO8601咨询时间",
    "triage_timestamp": "ISO8601时间戳",
    "session_id": "会话ID"
  }
}

HPI 时间线规则：
- 必须区分“相关暴露事件”和“症状真正起病时间”。
- 不要把潜水、飞行、游泳、外伤等暴露事件时间写成症状起病时间。
- 如果患者明确说暴露当时或暴露后没有某症状，必须写入 hpi_narrative 或 pertinent_negatives。
- 示例：患者“上周潜水没耳痛，今天耳朵刺痛”，应写“上周潜水，当时及之后无耳痛；今日出现耳内刺痛”，不能写“上周潜水后开始症状”。

医生报告覆盖规则：
- [结构化病历草稿]里的 cc、hpi、ph、triage 是 doctor_hpi 的主依据。
- 不要把未提供的信息编造成阳性；确实未采集时保留 null、空数组或写入 missing_or_unknown。
- 如果同一事实在对话和结构化草稿里冲突，以结构化草稿为准，并在叙述里保持时间线清楚。
- patient_output 可以更简短，但 doctor_hpi 要尽量完整承接已采集信息。
"""


class PatientOutputBuilder:
    """构建患者侧输出，封装 LLM JSON 解析和降级输出。"""

    def build_from_llm(self, data: dict) -> PatientOutput:
        legacy_depts = self._department_list(data.get("recommended_departments", []))
        primary = self._department(data.get("primary_department")) or (
            legacy_depts[0] if legacy_depts else self._unknown_department()
        )
        alternatives = self._department_list(data.get("alternative_departments", []))
        if not alternatives and legacy_depts:
            alternatives = legacy_depts[1:]

        return PatientOutput(
            primary_department=primary,
            alternative_departments=alternatives,
            urgency_level=data.get("urgency_level", "normal"),
            urgency_reason=data.get("urgency_reason", ""),
            patient_advice=data.get("patient_advice", "建议尽快就诊。"),
            red_flags_to_watch=list(data.get("red_flags_to_watch") or []),
        )

    def build_fallback(
        self,
        departments,
        urgency: str,
        urgency_reason: str,
    ) -> PatientOutput:
        depts = self._department_list(departments)
        primary = depts[0] if depts else self._unknown_department()
        return PatientOutput(
            primary_department=primary,
            alternative_departments=depts[1:3],
            urgency_level=urgency,
            urgency_reason=urgency_reason or "已完成初步评估",
            patient_advice="建议就诊首选科室，请携带相关病历资料。",
            red_flags_to_watch=["症状突然加重", "出现新症状"],
        )

    def _department_list(self, raw_items) -> list[DepartmentResult]:
        return [
            dept
            for dept in (self._department(item) for item in (raw_items or []))
            if dept is not None
        ]

    def _department(self, raw) -> DepartmentResult | None:
        if not isinstance(raw, dict):
            return None
        return DepartmentResult(
            department=str(raw.get("department", "") or ""),
            confidence=float(raw.get("confidence", 0) or 0),
            reason=str(raw.get("reason", "") or ""),
        )

    def _unknown_department(self) -> DepartmentResult:
        return DepartmentResult(
            department="待确认",
            confidence=0.0,
            reason="当前信息不足以确定首选科室",
        )


class DoctorHpiBuilder:
    """构建医生侧 HPI，封装结构化字段清洗和降级输出。"""

    def __init__(
        self,
        user_id: str = "",
        age: int | None = None,
        gender: str | None = None,
        consultation_time: str = "",
        session_id: str = "",
        profile_meds: list[str] | None = None,
        profile_allergies: list[str] | None = None,
        profile_pmh: list[str] | None = None,
        medical_record: dict | None = None,
    ) -> None:
        self._user_id = user_id
        self._age = age
        self._gender = gender
        self._consultation_time = consultation_time
        self._session_id = session_id
        self._profile_meds = list(profile_meds or [])
        self._profile_allergies = list(profile_allergies or [])
        self._profile_pmh = list(profile_pmh or [])
        self._medical_record = medical_record or {}

    def build_from_llm(self, data: dict) -> DoctorHPI:
        data = data or {}
        diffs = self._normalize_differentials(data.get("differential_diagnoses"))
        consultation_time = (
            self._consultation_time
            or data.get("consultation_time")
            or data.get("triage_timestamp", "")
        )
        general_condition = self._general_condition_from(data.get("general_condition"))
        past_history = self._past_history_from(data.get("past_history"))
        triage_summary = self._triage_summary_from(data.get("triage_summary"))
        severity_score = self._severity_from(data.get("severity_score"))
        result = DoctorHPI(
            user_id=self._user_id or data.get("user_id", ""),
            age=self._age if self._age is not None else data.get("age"),
            gender=self._gender if self._gender is not None else data.get("gender"),
            chief_complaint=self._first_text(
                data.get("chief_complaint"),
                self._record_value("cc.generated"),
                self._record_value("cc.draft"),
                self._record_value("hpi.chief_complaint"),
            ),
            hpi_narrative=self._merge_narrative_with_record(data.get("hpi_narrative")),
            onset=self._first_text(data.get("onset"), self._record_value("hpi.onset")) or None,
            location=self._first_text(data.get("location"), self._record_value("hpi.location")) or None,
            duration=self._first_text(data.get("duration"), self._record_value("hpi.duration")) or None,
            character=self._first_text(data.get("character"), self._record_value("hpi.character")) or None,
            alleviating_aggravating_factors=self._first_text(
                data.get("alleviating_aggravating_factors"),
                self._record_value("hpi.aggravating_alleviating"),
            ) or None,
            radiation=self._first_text(data.get("radiation"), self._record_value("hpi.radiation")) or None,
            timing=self._first_text(
                data.get("timing"),
                self._record_value("hpi.timing"),
                self._record_value("hpi.progression"),
            ) or None,
            severity_score=severity_score,
            associated_symptoms=self._merge_lists(
                data.get("associated_symptoms"),
                "hpi.associated_symptoms",
                "hpi.specific.associated_fever_symptoms",
            ),
            pertinent_negatives=self._merge_lists(
                data.get("pertinent_negatives"),
                "hpi.exposure_symptoms",
            ),
            diagnostic_history=self._first_text(
                data.get("diagnostic_history"),
                self._record_value("hpi.diagnostic_history"),
            ) or None,
            therapeutic_history=self._first_text(
                data.get("therapeutic_history"),
                self._record_value("hpi.therapeutic_history"),
                self._record_value("hpi.specific.antipyretics"),
            ) or None,
            general_condition=general_condition,
            past_history=past_history,
            relevant_pmh=self._merge_lists(
                data.get("relevant_pmh"),
                "hpi.relevant_history",
                "ph.disease_history",
                extra=self._profile_pmh,
            ),
            current_medications=self._merge_lists(
                data.get("current_medications"),
                "ph.current_medications",
                extra=self._profile_meds,
            ),
            allergies=self._merge_lists(
                data.get("allergies"),
                "ph.allergy_history",
                extra=self._profile_allergies,
            ),
            differential_diagnoses=diffs,
            recommended_workup=self._merge_lists(data.get("recommended_workup")),
            triage_summary=triage_summary,
            record_coverage={},
            urgency_level=data.get("urgency_level", "normal"),
            consultation_time=consultation_time,
            triage_timestamp=consultation_time,
            session_id=self._session_id or data.get("session_id", ""),
        )
        result["record_coverage"] = self._record_coverage_from(data.get("record_coverage"), result)
        return result

    def build_fallback(
        self,
        differentials,
        meds,
        allergies,
        pmh,
        urgency: str,
        timestamp: str,
        session_id: str,
    ) -> DoctorHPI:
        consultation_time = self._consultation_time or timestamp
        general_condition = self._general_condition_from({})
        past_history = self._past_history_from({})
        triage_summary = self._triage_summary_from({})
        result = DoctorHPI(
            user_id=self._user_id,
            age=self._age,
            gender=self._gender,
            chief_complaint=self._first_text(
                self._record_value("cc.generated"),
                self._record_value("cc.draft"),
                self._record_value("hpi.chief_complaint"),
                "见症状摘要",
            ),
            hpi_narrative=(
                self._merge_narrative_with_record("")
                or "患者经预诊系统采集，详见结构化症状数据。"
            ),
            onset=self._record_value("hpi.onset") or None,
            location=self._record_value("hpi.location") or None,
            duration=self._record_value("hpi.duration") or None,
            character=self._record_value("hpi.character") or None,
            alleviating_aggravating_factors=self._record_value("hpi.aggravating_alleviating") or None,
            radiation=self._record_value("hpi.radiation") or None,
            timing=self._first_text(
                self._record_value("hpi.timing"),
                self._record_value("hpi.progression"),
            ) or None,
            severity_score=self._severity_from(None),
            associated_symptoms=self._merge_lists(
                None,
                "hpi.associated_symptoms",
                "hpi.specific.associated_fever_symptoms",
            ),
            pertinent_negatives=self._merge_lists(None, "hpi.exposure_symptoms"),
            diagnostic_history=self._record_value("hpi.diagnostic_history") or None,
            therapeutic_history=self._first_text(
                self._record_value("hpi.therapeutic_history"),
                self._record_value("hpi.specific.antipyretics"),
            ) or None,
            general_condition=general_condition,
            past_history=past_history,
            relevant_pmh=self._merge_lists(
                None,
                "hpi.relevant_history",
                "ph.disease_history",
                extra=pmh,
            ),
            current_medications=self._merge_lists(None, "ph.current_medications", extra=meds),
            allergies=self._merge_lists(None, "ph.allergy_history", extra=allergies),
            differential_diagnoses=self._normalize_differentials(differentials),
            recommended_workup=[],
            triage_summary=triage_summary,
            record_coverage={},
            urgency_level=urgency,
            consultation_time=consultation_time,
            triage_timestamp=consultation_time,
            session_id=self._session_id or session_id,
        )
        result["record_coverage"] = self._record_coverage_from({}, result)
        return result

    def _normalize_differentials(self, raw_items) -> list[dict]:
        diffs = []
        for item in raw_items or []:
            if not isinstance(item, dict):
                continue
            diffs.append({
                "condition": item.get("condition", ""),
                "likelihood": item.get("likelihood", "medium"),
                "reasoning": item.get("reasoning", ""),
                "supporting_symptoms": _text_list(item.get("supporting_symptoms")),
                "risk_factors": _text_list(item.get("risk_factors")),
            })
        return diffs

    def _record_value(self, path: str) -> str:
        current = self._medical_record
        for part in path.split("."):
            if not isinstance(current, dict):
                return ""
            current = current.get(part)
        return _payload_text(current)

    def _record_raw(self, path: str):
        current = self._medical_record
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _first_text(self, *values) -> str:
        for value in values:
            text = _payload_text(value)
            if text:
                return text
        return ""

    def _merge_lists(self, raw, *record_paths: str, extra=None) -> list[str]:
        items = _text_list(raw)
        for path in record_paths:
            items.extend(_text_list(self._record_value(path)))
        items.extend(_text_list(extra))
        return _dedupe_text(items)

    def _severity_from(self, raw) -> str | None:
        max_temperature = self._record_value("hpi.specific.max_temperature")
        severity = self._first_text(
            raw,
            self._record_value("hpi.severity"),
            max_temperature,
        )
        severity = _clean_severity_score(severity)
        if severity and max_temperature and severity == max_temperature and "体温" not in severity:
            return f"最高体温{severity}"
        return severity

    def _general_condition_from(self, raw) -> dict:
        data = raw if isinstance(raw, dict) else {}
        fields = ("mental_status", "sleep", "appetite", "bowel", "urination", "weight_change")
        return {
            field: self._first_text(data.get(field), self._record_value(f"hpi.general_condition.{field}")) or None
            for field in fields
        }

    def _past_history_from(self, raw) -> dict:
        data = raw if isinstance(raw, dict) else {}
        fields = (
            "disease_history",
            "immunization_history",
            "surgical_history",
            "trauma_history",
            "blood_transfusion_history",
            "allergy_history",
            "current_medications",
        )
        history = {
            field: self._first_text(data.get(field), self._record_value(f"ph.{field}")) or None
            for field in fields
        }
        if history["disease_history"] is None and self._profile_pmh:
            history["disease_history"] = "、".join(self._profile_pmh)
        if history["allergy_history"] is None and self._profile_allergies:
            history["allergy_history"] = "、".join(self._profile_allergies)
        if history["current_medications"] is None and self._profile_meds:
            history["current_medications"] = "、".join(self._profile_meds)
        return history

    def _triage_summary_from(self, raw) -> dict:
        data = raw if isinstance(raw, dict) else {}
        triage = self._record_raw("triage") or {}
        primary = self._first_text(
            data.get("primary_department"),
            _department_text(triage.get("primary_department")),
        ) or None
        secondary = self._first_text(
            data.get("secondary_department"),
            _department_text(triage.get("secondary_department")),
        ) or None
        return {
            "protocol_id": self._first_text(data.get("protocol_id"), triage.get("protocol_id")) or None,
            "protocol_label": self._first_text(data.get("protocol_label"), triage.get("protocol_label")) or None,
            "primary_department": primary,
            "secondary_department": secondary,
            "reason": self._first_text(
                data.get("reason"),
                _department_reason(triage.get("primary_department")),
                _department_reason(triage.get("secondary_department")),
            ) or None,
        }

    def _merge_narrative_with_record(self, raw) -> str:
        text = _payload_text(raw)
        fragments = self._record_narrative_fragments()
        if not fragments:
            return text

        if not text:
            return "；".join(f"{label}：{value}" for label, value in fragments) + "。"

        missing = [
            f"{label}：{value}"
            for label, value in fragments
            if value and value not in text
        ]
        if missing:
            return f"{text} 补充采集信息：{'；'.join(missing)}。"
        return text

    def _record_narrative_fragments(self) -> list[tuple[str, str]]:
        associated = "、".join(self._merge_lists(
            None,
            "hpi.associated_symptoms",
            "hpi.specific.associated_fever_symptoms",
        ))
        severity = self._severity_from(None) or ""
        fragments = [
            ("主诉", self._first_text(
                self._record_value("cc.generated"),
                self._record_value("cc.draft"),
                self._record_value("hpi.chief_complaint"),
            )),
            ("起病时间", self._record_value("hpi.onset")),
            ("相关暴露", self._record_value("hpi.exposure_event")),
            ("暴露后情况", self._record_value("hpi.exposure_symptoms")),
            ("部位", self._record_value("hpi.location")),
            ("性质", self._record_value("hpi.character")),
            ("持续时间", self._record_value("hpi.duration")),
            ("严重程度", severity),
            ("时间特征", self._record_value("hpi.timing")),
            ("进展", self._record_value("hpi.progression")),
            ("加重/缓解因素", self._record_value("hpi.aggravating_alleviating")),
            ("放射", self._record_value("hpi.radiation")),
            ("伴随症状", associated),
            ("检查/诊断经过", self._record_value("hpi.diagnostic_history")),
            ("治疗经过", self._record_value("hpi.therapeutic_history")),
            ("退热处理", self._record_value("hpi.specific.antipyretics")),
            ("相关既往史", self._record_value("hpi.relevant_history")),
        ]
        return [(label, value) for label, value in fragments if value]

    def _record_coverage_from(self, raw, result: dict) -> dict:
        data = raw if isinstance(raw, dict) else {}
        used_sections = _dedupe_text(
            _text_list(data.get("used_sections"))
            + [
                section
                for section in ("cc", "hpi", "ph", "triage")
                if _has_record_content((self._medical_record or {}).get(section))
            ]
        )
        missing = _dedupe_text(
            _text_list(data.get("missing_or_unknown"))
            + [
                label
                for label, value in (
                    ("起病时间", result.get("onset")),
                    ("部位", result.get("location")),
                    ("持续时间", result.get("duration")),
                    ("性质", result.get("character")),
                    ("严重程度", result.get("severity_score")),
                    ("诊疗经过", result.get("diagnostic_history") or result.get("therapeutic_history")),
                )
                if not value
            ]
        )
        return {
            "used_sections": used_sections,
            "missing_or_unknown": missing,
        }


class EpisodePersistor:
    """将最终分诊结果保存为跨会话软记忆。"""

    def __init__(self, episodic: EpisodicMemory) -> None:
        self._episodic = episodic

    async def persist(
        self,
        symptom_summary: str,
        patient_output: PatientOutput,
        department_candidates: list[dict],
    ) -> None:
        # 优先用 LLM 修正后的科室，而不是直接用向量检索结果。
        primary = patient_output.get("primary_department") or {}
        top_department = (
            primary.get("department") if primary.get("department")
            else (department_candidates[0]["department"] if department_candidates else "待确认")
        )
        await self._episodic.save(
            symptom_summary=symptom_summary,
            advice=patient_output["patient_advice"],
            department=top_department,
        )


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
    medical_record = state.get("medical_record") or {}
    symptom_summary = build_symptom_summary(symptom_data)
    fact_store = FactStore.from_state(state.get("intake_facts") or [])
    fact_context = fact_store.prompt_context()
    medical_record_context = _format_medical_record_context(medical_record)

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
    profile_user_id = state.get("user_id") or getattr(health_profile, "user_id", "")
    profile_age = getattr(health_profile, "age", None)
    profile_gender = getattr(health_profile, "gender", None)

    timestamp = datetime.now(timezone.utc).isoformat()

    user_context = f"""[症状摘要（OPQRST）]
{symptom_summary}

[结构化病历草稿（医生报告优先依据）]
{medical_record_context}

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
        fallback_medical_record=medical_record,
        profile_meds=profile_meds,
        profile_allergies=profile_allergies,
        profile_pmh=profile_pmh,
        profile_user_id=profile_user_id,
        profile_age=profile_age,
        profile_gender=profile_gender,
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
    await EpisodePersistor(episodic).persist(
        symptom_summary=symptom_summary,
        patient_output=patient_output,
        department_candidates=department_candidates,
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
    fallback_medical_record=None,
    profile_meds=None,
    profile_allergies=None,
    profile_pmh=None,
    profile_user_id="",
    profile_age=None,
    profile_gender=None,
    timestamp="",
    session_id_val="",
) -> tuple[PatientOutput, DoctorHPI]:
    """单次 LLM 调用生成 patient_output + doctor_hpi"""
    patient_builder = PatientOutputBuilder()
    doctor_builder = DoctorHpiBuilder(
        user_id=profile_user_id,
        age=profile_age,
        gender=profile_gender,
        consultation_time=timestamp,
        session_id=session_id_val,
        profile_meds=profile_meds,
        profile_allergies=profile_allergies,
        profile_pmh=profile_pmh,
        medical_record=fallback_medical_record,
    )
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
            max_tokens=2500,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        patient_output = patient_builder.build_from_llm(parsed.get("patient_output", {}))
        doctor_hpi = doctor_builder.build_from_llm(parsed.get("doctor_hpi", {}))
        return patient_output, doctor_hpi

    except Exception:
        # LLM 或解析失败：用已有结构化数据构建降级输出
        return (
            patient_builder.build_fallback(
                fallback_departments or [],
                fallback_urgency,
                fallback_urgency_reason,
            ),
            doctor_builder.build_fallback(
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
    return PatientOutputBuilder().build_from_llm(data)


def _parse_doctor_hpi(data: dict) -> DoctorHPI:
    return DoctorHpiBuilder().build_from_llm(data)


def _fallback_patient_output(departments, urgency, urgency_reason) -> PatientOutput:
    return PatientOutputBuilder().build_fallback(departments, urgency, urgency_reason)


def _fallback_doctor_hpi(
    departments, differentials, meds, allergies, pmh,
    urgency, timestamp, session_id
) -> DoctorHPI:
    return DoctorHpiBuilder().build_fallback(
        differentials,
        meds,
        allergies,
        pmh,
        urgency,
        timestamp,
        session_id,
    )


def _format_medical_record_context(medical_record: dict | None) -> str:
    if not medical_record:
        return "（暂无结构化病历草稿）"
    try:
        return json.dumps(medical_record, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(medical_record)


def _payload_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return _payload_text(value.get("value"))
        if "draft" in value:
            return _payload_text(value.get("draft"))
        if "generated" in value:
            return _payload_text(value.get("generated"))
        if "department" in value:
            return _department_text(value)
        return ""
    if isinstance(value, (list, tuple, set)):
        return "、".join(_text_list(value))
    return str(value).strip()


def _text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            items.extend(_text_list(item))
        return items
    text = _payload_text(value)
    if not text:
        return []
    return [item.strip() for item in re.split(r"[、,，;；\n]+", text) if item.strip()]


def _dedupe_text(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _department_text(raw) -> str:
    if not isinstance(raw, dict):
        return _payload_text(raw)
    return str(raw.get("department") or "").strip()


def _department_reason(raw) -> str:
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("reason") or "").strip()


def _has_record_content(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        if "value" in value or "draft" in value or "generated" in value or "department" in value:
            return bool(_payload_text(value))
        return any(_has_record_content(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_record_content(item) for item in value)
    return bool(str(value).strip())


def _clean_severity_score(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:℃|度)\s*/\s*10", r"\1度", text)
    return text
