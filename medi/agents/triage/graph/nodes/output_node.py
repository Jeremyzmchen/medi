"""
OutputNode — 双向输出节点

职责：
  1. 生成患者侧输出：科室建议、紧急程度、就医指引（TriagePatientOutput）
  2. 生成医生侧输出：结构化预诊报告（DoctorReport JSON）
  3. 通过 AsyncStreamBus 发布 RESULT 事件（含 triage_output）
  4. 持久化到 EpisodicMemory（visit_records）

双向输出通过单次 LLM 调用（JSON 模式）生成，保证一致性。
结果统一收口为 triage_output，内部再分 patient / doctor_report。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from medi.agents.triage.graph.state import (
    TriageGraphState,
    TriageOutput,
    TriageOutputMeta,
    TriagePatientOutput,
    DoctorReport,
    DepartmentResult,
)
from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.clinical_summary import build_symptom_summary_from_record
from medi.agents.triage.prompts.triage_output import (
    TRIAGE_OUTPUT_SYSTEM_PROMPT,
    build_triage_output_user_context,
)
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback
from medi.memory.episodic import EpisodicMemory


TRIAGE_OUTPUT_SCHEMA_VERSION = "triage_output.v1"


class TriagePatientOutputBuilder:
    """构建患者侧输出，封装 LLM JSON 解析和降级输出。"""

    def build_from_llm(self, data: dict) -> TriagePatientOutput:
        primary = self._department(data.get("primary_department")) or self._unknown_department()
        alternatives = self._department_list(data.get("alternative_departments", []))

        return TriagePatientOutput(
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
    ) -> TriagePatientOutput:
        depts = self._department_list(departments)
        primary = depts[0] if depts else self._unknown_department()
        return TriagePatientOutput(
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


class DoctorReportBuilder:
    """构建医生侧预诊报告，封装结构化字段清洗和降级输出。"""

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
        preconsultation_record: dict | None = None,
    ) -> None:
        self._user_id = user_id
        self._age = age
        self._gender = gender
        self._consultation_time = consultation_time
        self._session_id = session_id
        self._profile_meds = list(profile_meds or [])
        self._profile_allergies = list(profile_allergies or [])
        self._profile_pmh = list(profile_pmh or [])
        self._preconsultation_record = preconsultation_record or {}

    def build_from_llm(self, data: dict) -> DoctorReport:
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
        result = DoctorReport(
            user_id=self._user_id or data.get("user_id", ""),
            age=self._age if self._age is not None else data.get("age"),
            gender=self._gender if self._gender is not None else data.get("gender"),
            chief_complaint=self._first_text(
                data.get("chief_complaint"),
                self._record_value("t4_chief_complaint.generated"),
                self._record_value("t4_chief_complaint.draft"),
                self._record_value("t2_hpi.chief_complaint"),
            ),
            hpi_narrative=self._merge_narrative_with_record(data.get("hpi_narrative")),
            onset=self._first_text(data.get("onset"), self._record_value("t2_hpi.onset")) or None,
            location=self._first_text(data.get("location"), self._record_value("t2_hpi.location")) or None,
            duration=self._first_text(data.get("duration"), self._record_value("t2_hpi.duration")) or None,
            character=self._first_text(data.get("character"), self._record_value("t2_hpi.character")) or None,
            alleviating_aggravating_factors=self._first_text(
                data.get("alleviating_aggravating_factors"),
                self._record_value("t2_hpi.aggravating_alleviating"),
            ) or None,
            radiation=self._first_text(data.get("radiation"), self._record_value("t2_hpi.radiation")) or None,
            timing=self._first_text(
                data.get("timing"),
                self._record_value("t2_hpi.timing"),
                self._record_value("t2_hpi.progression"),
            ) or None,
            severity_score=severity_score,
            associated_symptoms=self._merge_lists(
                data.get("associated_symptoms"),
                "t2_hpi.associated_symptoms",
                "t2_hpi.specific.associated_fever_symptoms",
            ),
            pertinent_negatives=self._merge_lists(
                data.get("pertinent_negatives"),
                "t2_hpi.exposure_symptoms",
            ),
            diagnostic_history=self._first_text(
                data.get("diagnostic_history"),
                self._record_value("t2_hpi.diagnostic_history"),
            ) or None,
            therapeutic_history=self._first_text(
                data.get("therapeutic_history"),
                self._record_value("t2_hpi.therapeutic_history"),
                self._record_value("t2_hpi.specific.antipyretics"),
            ) or None,
            general_condition=general_condition,
            past_history=past_history,
            relevant_pmh=self._merge_lists(
                data.get("relevant_pmh"),
                "t2_hpi.relevant_history",
                "t3_background.disease_history",
                extra=self._profile_pmh,
            ),
            current_medications=self._merge_lists(
                data.get("current_medications"),
                "t3_background.current_medications",
                extra=self._profile_meds,
            ),
            allergies=self._merge_lists(
                data.get("allergies"),
                "t3_background.allergy_history",
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
    ) -> DoctorReport:
        consultation_time = self._consultation_time or timestamp
        general_condition = self._general_condition_from({})
        past_history = self._past_history_from({})
        triage_summary = self._triage_summary_from({})
        result = DoctorReport(
            user_id=self._user_id,
            age=self._age,
            gender=self._gender,
            chief_complaint=self._first_text(
                self._record_value("t4_chief_complaint.generated"),
                self._record_value("t4_chief_complaint.draft"),
                self._record_value("t2_hpi.chief_complaint"),
                "见症状摘要",
            ),
            hpi_narrative=(
                self._merge_narrative_with_record("")
                or "患者经预诊系统采集，详见结构化症状数据。"
            ),
            onset=self._record_value("t2_hpi.onset") or None,
            location=self._record_value("t2_hpi.location") or None,
            duration=self._record_value("t2_hpi.duration") or None,
            character=self._record_value("t2_hpi.character") or None,
            alleviating_aggravating_factors=self._record_value("t2_hpi.aggravating_alleviating") or None,
            radiation=self._record_value("t2_hpi.radiation") or None,
            timing=self._first_text(
                self._record_value("t2_hpi.timing"),
                self._record_value("t2_hpi.progression"),
            ) or None,
            severity_score=self._severity_from(None),
            associated_symptoms=self._merge_lists(
                None,
                "t2_hpi.associated_symptoms",
                "t2_hpi.specific.associated_fever_symptoms",
            ),
            pertinent_negatives=self._merge_lists(None, "t2_hpi.exposure_symptoms"),
            diagnostic_history=self._record_value("t2_hpi.diagnostic_history") or None,
            therapeutic_history=self._first_text(
                self._record_value("t2_hpi.therapeutic_history"),
                self._record_value("t2_hpi.specific.antipyretics"),
            ) or None,
            general_condition=general_condition,
            past_history=past_history,
            relevant_pmh=self._merge_lists(
                None,
                "t2_hpi.relevant_history",
                "t3_background.disease_history",
                extra=pmh,
            ),
            current_medications=self._merge_lists(None, "t3_background.current_medications", extra=meds),
            allergies=self._merge_lists(None, "t3_background.allergy_history", extra=allergies),
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
        current = self._preconsultation_record
        for part in path.split("."):
            if not isinstance(current, dict):
                return ""
            current = current.get(part)
        return _payload_text(current)

    def _record_raw(self, path: str):
        current = self._preconsultation_record
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
        max_temperature = self._record_value("t2_hpi.specific.max_temperature")
        severity = self._first_text(
            raw,
            self._record_value("t2_hpi.severity"),
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
            field: self._first_text(data.get(field), self._record_value(f"t2_hpi.general_condition.{field}")) or None
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
            field: self._first_text(data.get(field), self._record_value(f"t3_background.{field}")) or None
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
        triage = self._record_raw("t1_triage") or {}
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
            "t2_hpi.associated_symptoms",
            "t2_hpi.specific.associated_fever_symptoms",
        ))
        severity = self._severity_from(None) or ""
        fragments = [
            ("主诉", self._first_text(
                self._record_value("t4_chief_complaint.generated"),
                self._record_value("t4_chief_complaint.draft"),
                self._record_value("t2_hpi.chief_complaint"),
            )),
            ("起病时间", self._record_value("t2_hpi.onset")),
            ("相关暴露", self._record_value("t2_hpi.exposure_event")),
            ("暴露后情况", self._record_value("t2_hpi.exposure_symptoms")),
            ("部位", self._record_value("t2_hpi.location")),
            ("性质", self._record_value("t2_hpi.character")),
            ("持续时间", self._record_value("t2_hpi.duration")),
            ("严重程度", severity),
            ("时间特征", self._record_value("t2_hpi.timing")),
            ("进展", self._record_value("t2_hpi.progression")),
            ("加重/缓解因素", self._record_value("t2_hpi.aggravating_alleviating")),
            ("放射", self._record_value("t2_hpi.radiation")),
            ("伴随症状", associated),
            ("检查/诊断经过", self._record_value("t2_hpi.diagnostic_history")),
            ("治疗经过", self._record_value("t2_hpi.therapeutic_history")),
            ("退热处理", self._record_value("t2_hpi.specific.antipyretics")),
            ("相关既往史", self._record_value("t2_hpi.relevant_history")),
        ]
        return [(label, value) for label, value in fragments if value]

    def _record_coverage_from(self, raw, result: dict) -> dict:
        data = raw if isinstance(raw, dict) else {}
        used_sections = _dedupe_text(
            _text_list(data.get("used_sections"))
            + [
                section
                for section in (
                    "t1_triage",
                    "t2_hpi",
                    "t3_background",
                    "t4_chief_complaint",
                )
                if _has_record_content((self._preconsultation_record or {}).get(section))
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
        patient: TriagePatientOutput,
        department_candidates: list[dict],
    ) -> None:
        # 优先用 LLM 修正后的科室，而不是直接用向量检索结果。
        primary = patient.get("primary_department") or {}
        top_department = (
            primary.get("department") if primary.get("department")
            else (department_candidates[0]["department"] if department_candidates else "待确认")
        )
        await self._episodic.save(
            symptom_summary=symptom_summary,
            advice=patient["patient_advice"],
            department=top_department,
        )


async def output_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    smart_chain: list,
    constraint_prompt: str,
    history_prompt: str,
    profile_snapshot,
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

    preconsultation_record = state.get("preconsultation_record") or {}
    clinical_facts = state.get("clinical_facts") or []
    symptom_summary = build_symptom_summary_from_record(
        preconsultation_record,
        clinical_facts,
        state.get("messages") or [],
    )
    fact_store = ClinicalFactStore.from_state(clinical_facts)
    fact_context = fact_store.prompt_context()

    clinical_assessment = state.get("clinical_assessment") or {}
    department_candidates = clinical_assessment.get("department_candidates") or []
    urgency = clinical_assessment.get("urgency") or {}
    urgency_level = urgency.get("level") or "normal"
    urgency_reason = urgency.get("reason") or ""
    differential_diagnoses = clinical_assessment.get("differential_diagnoses") or []
    risk_factors = clinical_assessment.get("risk_factors") or {}
    risk_factors_summary = risk_factors.get("summary") or ""

    # ── 构建 context ──
    profile_meds = getattr(profile_snapshot, "current_medications", []) or []
    profile_allergies = getattr(profile_snapshot, "allergies", []) or []
    profile_pmh = getattr(profile_snapshot, "chronic_conditions", []) or []
    profile_user_id = state.get("user_id") or getattr(profile_snapshot, "user_id", "")
    profile_age = getattr(profile_snapshot, "age", None)
    profile_gender = getattr(profile_snapshot, "gender", None)

    timestamp = datetime.now(timezone.utc).isoformat()

    user_context = build_triage_output_user_context(
        symptom_summary=symptom_summary,
        preconsultation_record=preconsultation_record,
        fact_context=fact_context,
        department_candidates=department_candidates,
        differential_diagnoses=differential_diagnoses,
        urgency_level=urgency_level,
        urgency_reason=urgency_reason,
        risk_factors_summary=risk_factors_summary,
        constraint_prompt=constraint_prompt,
        history_prompt=history_prompt,
        profile_user_id=profile_user_id,
        profile_age=profile_age,
        profile_gender=profile_gender,
        profile_meds=profile_meds,
        profile_allergies=profile_allergies,
        profile_pmh=profile_pmh,
        session_id=session_id,
        timestamp=timestamp,
    )

    messages = state.get("messages") or []

    # ── LLM 单次调用，生成双向输出 ──
    triage_output = await _generate_triage_output(
        system_prompt=TRIAGE_OUTPUT_SYSTEM_PROMPT,
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
        fallback_preconsultation_record=preconsultation_record,
        profile_meds=profile_meds,
        profile_allergies=profile_allergies,
        profile_pmh=profile_pmh,
        profile_user_id=profile_user_id,
        profile_age=profile_age,
        profile_gender=profile_gender,
        timestamp=timestamp,
        session_id_val=session_id,
    )
    patient = triage_output["patient"]

    # ── 发布 RESULT 事件（含结构化字段） ──
    await bus.emit(StreamEvent(
        type=EventType.RESULT,
        data={
            "content": patient["patient_advice"],
            "triage_output": triage_output,
        },
        session_id=session_id,
    ))

    # ── 持久化到 EpisodicMemory ──
    await EpisodePersistor(episodic).persist(
        symptom_summary=symptom_summary,
        patient=patient,
        department_candidates=department_candidates,
    )

    return {
        "triage_output": triage_output,
        "workflow_control": {
            "next_node": "done",
            "intake_complete": True,
            "graph_iteration": (state.get("workflow_control") or {}).get("graph_iteration", 0),
        },
    }


async def _generate_triage_output(
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
    fallback_preconsultation_record=None,
    profile_meds=None,
    profile_allergies=None,
    profile_pmh=None,
    profile_user_id="",
    profile_age=None,
    profile_gender=None,
    timestamp="",
    session_id_val="",
) -> TriageOutput:
    """单次 LLM 调用生成 triage_output。"""
    patient_builder = TriagePatientOutputBuilder()
    doctor_builder = DoctorReportBuilder(
        user_id=profile_user_id,
        age=profile_age,
        gender=profile_gender,
        consultation_time=timestamp,
        session_id=session_id_val,
        profile_meds=profile_meds,
        profile_allergies=profile_allergies,
        profile_pmh=profile_pmh,
        preconsultation_record=fallback_preconsultation_record,
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
            call_type="triage_output",
            messages=llm_messages,
            max_tokens=2500,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        raw_output = parsed.get("triage_output")
        if not isinstance(raw_output, dict):
            raise ValueError("missing triage_output")

        patient = patient_builder.build_from_llm(raw_output.get("patient", {}))
        doctor_report = doctor_builder.build_from_llm(raw_output.get("doctor_report", {}))
        return _build_triage_output(
            patient=patient,
            doctor_report=doctor_report,
            generated_at=timestamp,
        )

    except Exception:
        # LLM 或解析失败：用已有结构化数据构建降级输出
        patient = patient_builder.build_fallback(
                fallback_departments or [],
                fallback_urgency,
                fallback_urgency_reason,
            )
        doctor_report = doctor_builder.build_fallback(
                fallback_differentials or [],
                profile_meds or [],
                profile_allergies or [],
                profile_pmh or [],
                fallback_urgency,
                timestamp,
                session_id_val,
            )
        return _build_triage_output(
            patient=patient,
            doctor_report=doctor_report,
            generated_at=timestamp,
            fallback_used=True,
            fallback_reason="llm_output_failed_or_invalid_json",
        )


def _build_triage_output(
    patient: TriagePatientOutput,
    doctor_report: DoctorReport,
    generated_at: str,
    *,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> TriageOutput:
    meta = TriageOutputMeta(
        schema_version=TRIAGE_OUTPUT_SCHEMA_VERSION,
        generated_at=generated_at,
        source="output_node",
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    return TriageOutput(
        meta=meta,
        patient=patient,
        doctor_report=doctor_report,
    )


def _parse_patient(data: dict) -> TriagePatientOutput:
    return TriagePatientOutputBuilder().build_from_llm(data)


def _parse_doctor_report(data: dict) -> DoctorReport:
    return DoctorReportBuilder().build_from_llm(data)


def _fallback_patient(departments, urgency, urgency_reason) -> TriagePatientOutput:
    return TriagePatientOutputBuilder().build_fallback(departments, urgency, urgency_reason)


def _fallback_doctor_report(
    departments, differentials, meds, allergies, pmh,
    urgency, timestamp, session_id
) -> DoctorReport:
    return DoctorReportBuilder().build_fallback(
        differentials,
        meds,
        allergies,
        pmh,
        urgency,
        timestamp,
        session_id,
    )


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



