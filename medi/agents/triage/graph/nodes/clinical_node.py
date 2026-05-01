"""
ClinicalNode — 临床推理节点

职责：
  1. 调用 DepartmentRouter 检索症状-科室知识库（复用现有模块）
  2. 调用 UrgencyEvaluator LLM 层评估紧急程度（规则层已在 runner 前置）
  3. 评估患者特异性风险因子（结合 HealthProfile）
  4. 调用 LLM 生成鉴别诊断列表（JSON 结构化输出）
  5. 判断是否需要回追问（信息不足时 back-loop 到 IntakeNode）

输出写入 state：
  - department_candidates
  - urgency_level / urgency_reason
  - differential_diagnoses
  - risk_factors_summary
  - next_node: "output" 或 "intake"（back-loop）
"""

from __future__ import annotations

import json

from medi.agents.triage.graph.state import TriageGraphState, DifferentialDiagnosis, DepartmentResult
from medi.agents.triage.tools.clinical_tools import (
    evaluate_risk_factors,
    build_differential_prompt,
)
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
    health_profile,         # HealthProfile | None
    constraint_prompt: str,
    session_id: str,
    obs=None,
) -> dict:
    """
    ClinicalNode 执行函数。

    back-loop 条件：鉴别诊断不明确（top-2 差距 < 0.2）且有关键字段缺失，
    且 graph_iteration < 2（最多回一次）。
    """
    await bus.emit(StreamEvent(
        type=EventType.STAGE_START,
        data={"stage": "clinical"},
        session_id=session_id,
    ))

    symptom_data = state.get("symptom_data") or {}
    symptom_summary = _build_symptom_summary(symptom_data)
    query_text = _build_query_text(symptom_data, state.get("messages") or [])

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

    # ── 3. 风险因子评估（HealthProfile 交叉分析） ──
    risk_result = evaluate_risk_factors(symptom_data, health_profile)
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
    # 条件：诊断模糊（无高可能性项）+ 有关键缺失字段 + 还没回追问过
    graph_iteration = state.get("graph_iteration", 1)
    has_high_likelihood = any(d["likelihood"] == "high" for d in differential_diagnoses)
    has_key_missing = not symptom_data.get("region") and not symptom_data.get("time_pattern")

    if not has_high_likelihood and has_key_missing and graph_iteration < 2:
        next_node = "intake"
    else:
        next_node = "output"

    return {
        "department_candidates": department_candidates,
        "urgency_level": urgency_level,
        "urgency_reason": urgency_reason,
        "differential_diagnoses": differential_diagnoses,
        "risk_factors_summary": risk_factors_summary,
        "next_node": next_node,
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


def _build_symptom_summary(symptom_data: dict) -> str:
    """从 SymptomData 生成可读摘要，供 LLM prompt 使用"""
    lines = []
    if symptom_data.get("region"):
        lines.append(f"部位：{symptom_data['region']}")
    if symptom_data.get("time_pattern"):
        lines.append(f"时间：{symptom_data['time_pattern']}")
    if symptom_data.get("onset"):
        lines.append(f"诱因：{symptom_data['onset']}")
    if symptom_data.get("quality"):
        lines.append(f"性质：{symptom_data['quality']}")
    if symptom_data.get("severity"):
        lines.append(f"程度：{symptom_data['severity']}/10")
    if symptom_data.get("radiation"):
        lines.append(f"放射痛：{symptom_data['radiation']}")
    if symptom_data.get("provocation"):
        lines.append(f"加重/缓解：{symptom_data['provocation']}")
    if symptom_data.get("accompanying"):
        lines.append(f"伴随症状：{', '.join(symptom_data['accompanying'])}")
    if symptom_data.get("relevant_history"):
        lines.append(f"既往史：{symptom_data['relevant_history']}")
    if symptom_data.get("medications"):
        lines.append(f"用药：{', '.join(symptom_data['medications'])}")
    if symptom_data.get("allergies"):
        lines.append(f"过敏：{', '.join(symptom_data['allergies'])}")
    # 若字段都为空，退回到原始描述
    if not lines and symptom_data.get("raw_descriptions"):
        return " ".join(symptom_data["raw_descriptions"][-3:])
    return "\n".join(lines) if lines else "（症状信息待采集）"


def _build_query_text(symptom_data: dict, messages: list[dict]) -> str:
    """构建向量检索用的文本：优先用结构化字段，退回到对话历史"""
    parts = []
    for field in ("region", "quality", "time_pattern", "onset"):
        val = symptom_data.get(field)
        if val:
            parts.append(val)
    parts.extend(symptom_data.get("accompanying") or [])
    if not parts:
        # 退回到最近的用户消息
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        parts = user_msgs[-3:]
    return " ".join(parts)
