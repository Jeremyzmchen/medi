"""
ClinicalNode — 临床推理节点

职责：
  1. 调用 DepartmentRouter 检索症状-科室知识库（复用现有模块）
  2. 调用 UrgencyEvaluator LLM 层评估紧急程度（规则层已在 runner 前置）
  3. 评估患者特异性风险因子（结合 ProfileSnapshot）
  4. 调用 LLM 生成鉴别诊断列表（JSON 结构化输出）
  5. 判断是否需要回追问（信息不足时 back-loop 到 IntakeNode）

输出写入 state：
  - clinical_assessment.department_candidates
  - clinical_assessment.urgency
  - clinical_assessment.differential_diagnoses
  - clinical_assessment.risk_factors
  - clinical_assessment.missing_slots: 需要回补的明确字段
  - workflow_control.next_node: "output" 或 "intake"（back-loop）
"""

from __future__ import annotations

import json

from medi.agents.triage.graph.state import (
    ClinicalAssessment,
    TriageGraphState,
    DifferentialDiagnosis,
    DepartmentResult,
)
from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.clinical_summary import (
    build_department_query_from_record,
    build_symptom_summary_from_record,
)
from medi.agents.triage.prompts.differential import build_differential_prompt
from medi.agents.triage.tools.clinical_tools import evaluate_risk_factors
from medi.agents.triage.urgency_evaluator import evaluate_urgency_by_llm, UrgencyLevel
from medi.agents.triage.department_router import DepartmentRouter
from medi.core.stream_bus import AsyncStreamBus, EventType, StreamEvent
from medi.core.llm_client import call_with_fallback


async def clinical_node(
    state: TriageGraphState,
    bus: AsyncStreamBus,
    router: DepartmentRouter,
    smart_chain: list,
    fast_chain: list,
    profile_snapshot,
    constraint_prompt: str,
    session_id: str,
    obs=None,
) -> dict:
    """
    ClinicalNode 执行函数。

    back-loop 条件：鉴别诊断不明确、存在明确可补字段，且 graph_iteration < 2。
    """
    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "clinical"},
        session_id=session_id,
    ))

    messages = state.get("messages") or []
    preconsultation_record = state.get("preconsultation_record") or {}
    clinical_facts = state.get("clinical_facts") or []
    symptom_summary = build_symptom_summary_from_record(
        preconsultation_record,
        clinical_facts,
        messages,
    )
    query_text = build_department_query_from_record(
        preconsultation_record,
        clinical_facts,
        messages,
    )

    # ── 1. 科室路由检索 ──
    raw_candidates = await router.route(query_text, top_k=3)
    department_candidates: list[DepartmentResult] = [
        DepartmentResult(
            department=c.department,
            confidence=c.confidence,
            reason=c.reason,
        )
        for c in raw_candidates
    ]

    # ── 2. LLM 紧急程度评估（语义层，规则层已在 runner 前置） ──
    urgency_result = await evaluate_urgency_by_llm(
        symptom_text=symptom_summary,
        call_with_fallback=call_with_fallback,
        fast_chain=fast_chain,
        bus=bus,
        session_id=session_id,
        obs=obs,
    )
    urgency_level = urgency_result.level.value
    urgency_reason = urgency_result.reason

    # ── 3. 风险因子评估（ProfileSnapshot 交叉分析） ──
    risk_result = evaluate_risk_factors(symptom_summary, profile_snapshot)
    risk_factors_summary = risk_result["risk_summary"]

    # 若风险因子建议提升紧急程度（且当前为 normal/watchful），升级为 urgent
    if risk_result["elevated_urgency"] and urgency_level in ("normal", "watchful"):
        urgency_level = UrgencyLevel.URGENT.value
        urgency_reason = f"因患者基础疾病风险，建议提升就医紧急程度。{urgency_reason}"

    # ── 4. LLM 鉴别诊断（JSON 结构化输出） ──
    differential_diagnoses = await _generate_differential(
        symptom_summary=symptom_summary,
        department_candidates=department_candidates,
        risk_factors_summary=risk_factors_summary,
        constraint_prompt=constraint_prompt,
        smart_chain=smart_chain,
        bus=bus,
        session_id=session_id,
        obs=obs,
    )

    # ── 5. back-loop 判断 ──
    # 只有明确知道缺哪个字段时才回追问，避免“不自信所以再问一轮”的宽泛循环。
    workflow_control = state.get("workflow_control") or {}
    graph_iteration = workflow_control.get("graph_iteration", 1)
    has_high_likelihood = any(d["likelihood"] == "high" for d in differential_diagnoses)
    clinical_missing_slots = _missing_for_diagnosis(state)

    if not has_high_likelihood and clinical_missing_slots and graph_iteration < 2:
        next_node = "intake"
    else:
        next_node = "output"

    clinical_assessment = ClinicalAssessment(
        department_candidates=department_candidates,
        urgency={
            "level": urgency_level,
            "reason": urgency_reason,
        },
        differential_diagnoses=differential_diagnoses,
        risk_factors={
            "items": risk_result.get("risk_factors", []),
            "summary": risk_factors_summary,
            "elevated_urgency": bool(risk_result.get("elevated_urgency")),
        },
        missing_slots=clinical_missing_slots,
        status="needs_more_info" if next_node == "intake" else "complete",
    )

    return {
        "clinical_assessment": clinical_assessment,
        "workflow_control": {
            "next_node": next_node,
            "intake_complete": True,
            "graph_iteration": graph_iteration,
        },
    }


async def _generate_differential(
    symptom_summary: str,
    department_candidates: list[DepartmentResult],
    risk_factors_summary: str,
    constraint_prompt: str,
    smart_chain: list,
    bus: AsyncStreamBus,
    session_id: str,
    obs=None,
) -> list[DifferentialDiagnosis]:
    """调用 LLM 生成鉴别诊断，解析 JSON 响应"""
    prompt = build_differential_prompt(
        symptom_summary=symptom_summary,
        department_candidates=department_candidates,
        risk_factors_summary=risk_factors_summary,
        constraint_prompt=constraint_prompt,
    )

    try:
        response = await call_with_fallback(
            chain=smart_chain,
            bus=bus,
            session_id=session_id,
            obs=obs,
            call_type="clinical_differential",
            messages=[
                {"role": "system", "content": "你是一位专业临床医生，输出严格的 JSON 格式。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        items = parsed.get("differential_diagnoses", [])

        result: list[DifferentialDiagnosis] = []
        for item in items[:4]:  # 最多 4 个
            result.append(DifferentialDiagnosis(
                condition=str(item.get("condition", "")),
                likelihood=str(item.get("likelihood", "medium")),
                reasoning=str(item.get("reasoning", "")),
                supporting_symptoms=list(item.get("supporting_symptoms") or []),
                risk_factors=list(item.get("risk_factors") or []),
            ))
        return result

    except Exception:
        # LLM 或解析失败：返回基于科室候选的降级诊断
        return [
            DifferentialDiagnosis(
                condition=f"需进一步评估（{c['department']}相关）",
                likelihood="medium",
                reasoning="基于症状-科室知识库匹配",
                supporting_symptoms=[],
                risk_factors=[],
            )
            for c in department_candidates[:2]
        ]


def _missing_for_diagnosis(state: TriageGraphState) -> list[str]:
    intake_plan = state.get("intake_plan") or {}
    protocol_id = intake_plan.get("protocol_id") or "generic_opqrst"
    store = ClinicalFactStore.from_state(state.get("clinical_facts") or [])
    record = state.get("preconsultation_record") or {}
    missing: list[str] = []

    if not (
        _slot_collected_or_has_record_value(store, "hpi.onset", record, "t2_hpi.onset")
        or _slot_collected_or_has_record_value(store, "hpi.timing", record, "t2_hpi.timing")
    ):
        missing.append("hpi.onset")
    if not (
        _slot_collected_or_has_record_value(
            store,
            "hpi.associated_symptoms",
            record,
            "t2_hpi.associated_symptoms",
        )
        or _slot_collected_or_has_record_value(
            store,
            "specific.associated_fever_symptoms",
            record,
            "t2_hpi.specific.associated_fever_symptoms",
        )
    ):
        missing.append("hpi.associated_symptoms")

    if protocol_id in {"chest_pain", "abdominal_pain", "headache", "trauma"}:
        if not _slot_collected_or_has_record_value(store, "hpi.location", record, "t2_hpi.location"):
            missing.append("hpi.location")
    if protocol_id == "chest_pain":
        if not _slot_collected_or_has_record_value(store, "hpi.radiation", record, "t2_hpi.radiation"):
            missing.append("hpi.radiation")
        _append_missing_pattern(missing, store, "dyspnea_sweating")
        _append_missing_pattern(missing, store, "exertional_related")
    elif protocol_id == "abdominal_pain":
        _append_missing_pattern(missing, store, "stool_or_bleeding")
    elif protocol_id == "headache":
        _append_missing_pattern(missing, store, "neuro_deficits")
        _append_missing_pattern(missing, store, "sudden_or_worst")
    elif protocol_id == "dyspnea":
        _append_missing_pattern(missing, store, "rest_or_exertion")
        _append_missing_pattern(missing, store, "chest_pain_or_wheeze")
    elif protocol_id == "diarrhea_vomiting":
        _append_missing_pattern(missing, store, "frequency")
        _append_missing_pattern(missing, store, "blood_or_black_stool")

    return _unique(missing)


def _slot_collected_or_has_record_value(
    store: ClinicalFactStore,
    slot: str,
    record: dict,
    record_path: str,
) -> bool:
    if store.is_collected(slot):
        return True
    return bool(_record_text(record, record_path))


def _record_text(record: dict, path: str) -> str:
    current = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    if isinstance(current, dict):
        for key in ("value", "generated", "draft", "department"):
            if key in current:
                return _record_text({key: current.get(key)}, key)
        return ""
    if isinstance(current, (list, tuple, set)):
        return "、".join(str(item).strip() for item in current if str(item).strip())
    return str(current or "").strip()


def _append_missing_pattern(
    items: list[str],
    store: ClinicalFactStore,
    key: str,
) -> None:
    if not store.is_collected(f"specific.{key}"):
        items.append(f"specific.{key}")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
