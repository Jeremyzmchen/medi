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


class MonitorResult(TypedDict, total=False):
    """
    Monitor 节点输出：纯打分，不做调度决策。

    对应 Monitor 角色：评估每个子任务的完成度，
    输出 0-100 分数和缺失槽位列表，由 Controller 读取后决定下一步。
    """
    score: int                          # 0-100 预诊档案价值分
    red_flags_checked: bool             # 危险信号槽全部已回答
    safety_slots_covered: bool          # 用药 + 过敏均已回答
    doctor_summary_ready: bool          # 达到医生预诊所需核心信息
    required_slots_covered: bool        # 所有协议必填槽均已采集
    high_value_missing_slots: list[str] # 有价值但尚未采集的槽位（已去重）
    reason: str                         # 可解释性说明


class SafetyGateResult(TypedDict, total=False):
    """Graph-level safety gate decision for the latest user message."""
    status: str                         # "passed" | "blocked"
    urgency_level: str | None           # emergency when blocked by red-flag rules
    reason: str                         # decision explanation
    triggered_by_rule: bool             # True when deterministic red-flag rules fired
    method: str                         # "rule" | "llm" | "none" | fallback marker
    risk_concept: str | None            # matched emergency concept
    confidence: float | None            # classifier confidence when available


class ControllerDecision(TypedDict, total=False):
    """
    Controller 节点输出：全局调度决策。

    对应 Controller 角色：读取 Monitor 分数，
    选择下一个最高价值子任务，交给 Prompter 生成问题，或放行到 ClinicalNode。
    """
    can_finish_intake: bool             # True → 路由到 ClinicalNode
    next_best_task: str | None          # 选定的下一个子任务
    task_priority_score: float | None   # 本轮任务调度分
    task_instruction: str | None        # 给 Prompter/Inquirer 的任务说明
    next_best_question: str | None      # Prompter 生成的问题文本
    reason: str                         # 决策理由


# ─────────────────────────────────────────────
# Pre-consultation task orchestration state
# ─────────────────────────────────────────────

class PreconsultationRecord(TypedDict, total=False):
    """本次 Encounter 的预诊档案视图。"""
    meta: dict
    t1_triage: dict
    t2_hpi: dict
    t3_background: dict
    t4_chief_complaint: dict


class TaskProgress(TypedDict, total=False):
    """Monitor 对单个子任务的完成度评估。"""
    task_id: str
    group_id: str
    label: str
    description: str
    base_priority: int
    critical: bool
    score: float                  # 0.0-1.0，子任务完成度
    status: str                   # "pending" | "partial" | "complete" | "skipped"
    completed_requirements: list[str]
    missing_requirements: list[str]
    requirement_details: list[dict]
    reason: str


class TaskBoard(TypedDict, total=False):
    """预诊任务看板：记录 T1/T2/T3/T4 的完成度和调度状态。"""
    monitor: MonitorResult
    controller: ControllerDecision
    tree: dict
    progress: dict[str, TaskProgress]
    pending_tasks: list[str]
    current_task: str | None
    task_rounds: dict[str, int]


class IntakePlanState(TypedDict, total=False):
    """当前预问诊采集协议选择结果。"""
    protocol_id: str
    protocol_label: str
    overlay_ids: list[str]


class WorkflowControl(TypedDict, total=False):
    """LangGraph 运行控制信号，不承载临床数据。"""
    next_node: str
    intake_complete: bool
    graph_iteration: int


# ─────────────────────────────────────────────
# Nested output schemas
# ─────────────────────────────────────────────

class GeneralCondition(TypedDict, total=False):
    """
    一般情况（General Condition）

    记录患者在本次发病期间的基础功能状态，对内科、儿科、老年科尤为重要。
    全部字段可选，缺失时不影响最低预诊要求。
    """
    mental_status: str | None        # 精神状态：清醒/嗜睡/烦躁/意识模糊
    sleep: str | None                # 睡眠：正常/差/入睡困难/多睡
    appetite: str | None             # 食欲：正常/减退/无食欲/不能进食
    bowel: str | None                # 大便：正常/便秘/腹泻/黑便/血便
    urination: str | None            # 小便：正常/减少/尿频/尿痛/无尿
    weight_change: str | None        # 体重变化：无/近期明显下降/近期增加


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


class UrgencyAssessment(TypedDict, total=False):
    """ClinicalNode 生成的紧急程度判断。"""
    level: str                  # "emergency" | "urgent" | "normal" | "watchful"
    reason: str


class RiskFactorAssessment(TypedDict, total=False):
    """ClinicalNode 生成的患者特异性风险因子分析。"""
    items: list[str]
    summary: str
    elevated_urgency: bool


class ClinicalAssessment(TypedDict, total=False):
    """ClinicalNode 生成的临床判断视图。"""
    department_candidates: list[DepartmentResult]
    urgency: UrgencyAssessment
    differential_diagnoses: list[DifferentialDiagnosis]
    risk_factors: RiskFactorAssessment
    missing_slots: list[str]
    status: str                 # "complete" | "needs_more_info" | "escalated"


class TriagePatientOutput(TypedDict):
    """患者侧输出：科室方向 + 紧急程度 + 就医指引"""
    primary_department: DepartmentResult
    alternative_departments: list[DepartmentResult]
    urgency_level: str          # "emergency" | "urgent" | "normal" | "watchful"
    urgency_reason: str
    patient_advice: str
    red_flags_to_watch: list[str]


class DoctorReport(TypedDict):
    """
    医生侧输出：结构化预诊报告

    以 HPI 为核心，同时包含患者背景、分诊摘要、鉴别诊断和建议检查。
    供接诊医生在见患者前快速了解病情，减少重复问诊。
    """
    user_id: str
    age: int | None
    gender: str | None
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
    diagnostic_history: str | None          # 本次发病后已经做过的检查/诊断
    therapeutic_history: str | None         # 本次发病后已经用过的药物/处理
    general_condition: GeneralCondition | None
    past_history: dict
    relevant_pmh: list[str]                 # 相关既往史
    current_medications: list[str]
    allergies: list[str]
    differential_diagnoses: list[DifferentialDiagnosis]
    recommended_workup: list[str]
    triage_summary: dict
    record_coverage: dict
    urgency_level: str
    consultation_time: str
    triage_timestamp: str
    session_id: str


class TriageOutputMeta(TypedDict, total=False):
    """最终分诊输出元信息。"""
    schema_version: str
    generated_at: str
    source: str
    fallback_used: bool
    fallback_reason: str | None


class TriageOutput(TypedDict):
    """最终分诊输出：统一收口患者侧与医生侧报告。"""
    meta: TriageOutputMeta
    patient: TriagePatientOutput
    doctor_report: DoctorReport


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
    safety_gate: SafetyGateResult
    intake_plan: IntakePlanState            # 当前匹配到的主诉采集协议和叠加规则
    clinical_facts: list[dict]                # 已抽取的临床事实槽，供下一轮合并去重
    preconsultation_record: PreconsultationRecord  # ClinicalFacts 投影出的 T1/T2/T3/T4 预诊档案
    task_board: TaskBoard                   # Monitor/Controller/Inquirer 共享的任务看板

    # ── Clinical reasoning ──
    clinical_assessment: ClinicalAssessment

    # ── Final output ──
    triage_output: TriageOutput | None

    # ── Workflow control ──
    workflow_control: WorkflowControl


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def empty_intake_plan() -> IntakePlanState:
    return IntakePlanState(
        protocol_id="generic_opqrst",
        protocol_label="通用 OPQRST",
        overlay_ids=[],
    )


def empty_workflow_control() -> WorkflowControl:
    return WorkflowControl(
        next_node="intake",
        intake_complete=False,
        graph_iteration=0,
    )


def empty_preconsultation_record() -> PreconsultationRecord:
    return PreconsultationRecord(
        meta={
            "schema_version": "preconsultation_record.v1",
            "source": "clinical_facts",
        },
        t1_triage={},
        t2_hpi={
            "general_condition": {},
            "specific": {},
        },
        t3_background={},
        t4_chief_complaint={},
    )


def empty_task_board() -> TaskBoard:
    from medi.agents.triage.task_definitions import initial_task_progress, task_ids

    return TaskBoard(
        monitor={},
        controller={},
        tree={},
        progress=initial_task_progress(),
        pending_tasks=task_ids(),
        current_task=None,
        task_rounds={},
    )


def empty_clinical_assessment() -> ClinicalAssessment:
    return ClinicalAssessment(
        department_candidates=[],
        urgency={
            "level": "normal",
            "reason": "",
        },
        differential_diagnoses=[],
        risk_factors={
            "items": [],
            "summary": "",
            "elevated_urgency": False,
        },
        missing_slots=[],
        status="pending",
    )


MAX_INTAKE_ROUNDS = 10

