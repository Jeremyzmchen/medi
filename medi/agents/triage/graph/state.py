"""
TriageGraphState — LangGraph 多 Agent 分诊状态

所有跨节点字段定义在此。TypedDict 保证 JSON 可序列化，
支持 LangGraph checkpointing（多轮中断恢复）。

字段分组：
  - Identity：session/user 标识（不可变）
  - Conversation：消息历史（append-only）
  - Intake：护士采集状态（IntakeNode 维护）
  - Clinical：诊断推理结果（ClinicalNode 填写）
  - Output：双向输出（OutputNode 填写）
  - Routing：节点间路由信号
  - Meta：循环防护、错误处理
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


# ─────────────────────────────────────────────
# Intake collection status
# ─────────────────────────────────────────────

class OPQRSTStatus(TypedDict, total=False):
    """OPQRST 各字段的采集状态"""
    onset: str          # "complete" | "partial" | "missing"
    provocation: str
    quality: str
    location: str
    severity: str
    time_pattern: str
    radiation: str      # 放射痛（新增，护士必问）


class CollectionStatus(TypedDict, total=False):
    """
    护士视角的采集完整性评估。

    每轮由 LLM 更新，终止条件由此驱动。
    LLM 输出此结构，系统验证 can_conclude + 最低字段后决定是否继续。
    """
    chief_complaint: str            # 主诉
    opqrst: OPQRSTStatus
    associated_symptoms: str        # 伴随症状
    relevant_history: str           # 既往史（相关）
    medications_allergies: str      # 用药 + 过敏（医疗安全必采）
    pattern_specific: dict          # 症状模式特异性字段，如 {"visual_aura": "missing"}
    can_conclude: bool              # LLM 判断是否可结束采集
    reason: str                     # 可解释性：为什么结束/为什么继续


# ─────────────────────────────────────────────
# Nested output schemas
# ─────────────────────────────────────────────

class SymptomData(TypedDict, total=False):
    """从对话中提取的结构化症状数据（供 ClinicalNode 使用）"""
    raw_descriptions: list[str]
    onset: str | None
    provocation: str | None
    quality: str | None
    region: str | None
    severity: str | None
    time_pattern: str | None
    radiation: str | None
    accompanying: list[str]
    relevant_history: str | None
    medications: list[str]
    allergies: list[str]


class DepartmentResult(TypedDict):
    department: str
    confidence: float
    reason: str


class DifferentialDiagnosis(TypedDict):
    condition: str
    likelihood: str             # "high" | "medium" | "low"
    reasoning: str
    supporting_symptoms: list[str]
    risk_factors: list[str]


class PatientOutput(TypedDict):
    """患者侧输出：科室方向 + 紧急程度 + 就医指引"""
    recommended_departments: list[DepartmentResult]
    urgency_level: str          # "emergency" | "urgent" | "normal" | "watchful"
    urgency_reason: str
    patient_advice: str
    red_flags_to_watch: list[str]


class DoctorHPI(TypedDict):
    """
    医生侧输出：结构化 HPI（History of Present Illness）

    OLDCARTS 格式，供接诊医生在见患者前快速了解病情。
    护士采集的所有信息都汇聚于此，减少医生重复问诊。
    """
    chief_complaint: str
    hpi_narrative: str                      # 自由文本叙述（完整 HPI）
    onset: str | None
    location: str | None
    duration: str | None
    character: str | None
    alleviating_aggravating_factors: str | None
    radiation: str | None
    timing: str | None
    severity_score: str | None
    associated_symptoms: list[str]
    pertinent_negatives: list[str]          # LLM 推断的临床相关阴性症状
    relevant_pmh: list[str]                 # 相关既往史
    current_medications: list[str]
    allergies: list[str]
    differential_diagnoses: list[DifferentialDiagnosis]
    recommended_workup: list[str]
    urgency_level: str
    triage_timestamp: str
    session_id: str


# ─────────────────────────────────────────────
# Main graph state
# ─────────────────────────────────────────────

class TriageGraphState(TypedDict, total=False):
    # ── Identity (immutable) ──
    session_id: str
    user_id: str

    # ── Conversation history (append-only) ──
    messages: Annotated[list[dict], operator.add]

    # ── Intake nurse state ──
    symptom_data: SymptomData
    collection_status: CollectionStatus     # 护士每轮更新的采集状态
    follow_up_count: int                    # 已追问轮数（硬上限兜底）
    intake_complete: bool                   # True → 进入 ClinicalNode

    # ── Clinical reasoning ──
    department_candidates: list[DepartmentResult]
    urgency_level: str
    urgency_reason: str
    differential_diagnoses: list[DifferentialDiagnosis]
    risk_factors_summary: str

    # ── Final outputs ──
    patient_output: PatientOutput | None
    doctor_hpi: DoctorHPI | None

    # ── Routing ──
    next_node: str

    # ── Loop guard & error ──
    graph_iteration: int
    error: str | None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def empty_symptom_data() -> SymptomData:
    return SymptomData(
        raw_descriptions=[],
        onset=None, provocation=None, quality=None,
        region=None, severity=None, time_pattern=None,
        radiation=None, accompanying=[],
        relevant_history=None, medications=[], allergies=[],
    )


def empty_collection_status() -> CollectionStatus:
    opqrst = OPQRSTStatus(
        onset="missing", provocation="missing", quality="missing",
        location="missing", severity="missing", time_pattern="missing",
        radiation="missing",
    )
    return CollectionStatus(
        chief_complaint="missing",
        opqrst=opqrst,
        associated_symptoms="missing",
        relevant_history="missing",
        medications_allergies="missing",
        pattern_specific={},
        can_conclude=False,
        reason="采集尚未开始",
    )


MAX_INTAKE_ROUNDS = 10


def check_minimum_fields(status: CollectionStatus) -> tuple[bool, list[str]]:
    """
    校验最低必要字段是否已采集。
    返回 (passed, missing_fields)

    规则：
    - medications_allergies 必须是 "complete"（护士必须亲口问到）
    - 其余关键字段只要不是 "missing"（complete 或 partial 均可）
    """
    missing = []
    opqrst = status.get("opqrst") or {}

    # 主诉
    if status.get("chief_complaint") == "missing":
        missing.append("chief_complaint")
    # 部位
    if opqrst.get("location", "missing") == "missing":
        missing.append("opqrst.location")
    # 发作时间
    if opqrst.get("onset", "missing") == "missing":
        missing.append("opqrst.onset")
    # 性质
    if opqrst.get("quality", "missing") == "missing":
        missing.append("opqrst.quality")
    # 严重程度
    if opqrst.get("severity", "missing") == "missing":
        missing.append("opqrst.severity")
    # 时间特征
    if opqrst.get("time_pattern", "missing") == "missing":
        missing.append("opqrst.time_pattern")
    # 用药/过敏：必须是 complete
    if status.get("medications_allergies") != "complete":
        missing.append("medications_allergies")

    return len(missing) == 0, missing
