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


class MonitorResult(TypedDict, total=False):
    """
    Monitor 节点输出：纯打分，不做调度决策。

    对应论文 4.2 节 Monitor —— 评估每个子任务的完成度，
    输出 0-100 分数和缺失槽位列表，由 Controller 读取后决定下一步。
    """
    score: int                          # 0-100 预诊档案价值分
    red_flags_checked: bool             # 危险信号槽全部已回答
    safety_slots_covered: bool          # 用药 + 过敏均已回答
    doctor_summary_ready: bool          # 达到医生预诊所需核心信息
    required_slots_covered: bool        # 所有协议必填槽均已采集
    high_value_missing_slots: list[str] # 有价值但尚未采集的槽位（已去重）
    reason: str                         # 可解释性说明


class ControllerDecision(TypedDict, total=False):
    """
    Controller 节点输出：全局调度决策。

    对应论文 4.2 节 Controller —— 读取 Monitor 分数，
    选择下一个最高价值子任务，生成上下文感知问题或放行到 ClinicalNode。
    """
    can_finish_intake: bool             # True → 路由到 ClinicalNode
    next_best_slot: str | None          # 选定的下一个采集槽
    next_best_question: str | None      # 对应的上下文感知问题文本
    reason: str                         # 决策理由


class IntakeReviewStatus(TypedDict, total=False):
    """
    预诊质量门评估结果。

    兼容当前 IntakeReviewNode，同时把 Monitor/Controller 的结果
    汇总成一个可直接持久化、可直接给 UI 展示的状态快照。
    """
    doctor_value_score: int
    can_finish_intake: bool
    doctor_summary_ready: bool
    red_flags_checked: bool
    required_slots_covered: bool
    safety_slots_covered: bool
    high_value_missing_slots: list[str]
    next_best_slot: str | None
    next_best_question: str | None
    reason: str
    task_tree: dict


# ─────────────────────────────────────────────
# Nested output schemas
# ─────────────────────────────────────────────

class GeneralCondition(TypedDict, total=False):
    """
    一般情况（论文 Table 5: General Condition）

    记录患者在本次发病期间的基础功能状态，对内科、儿科、老年科尤为重要。
    全部字段可选，缺失时不影响最低预诊要求。
    """
    mental_status: str | None        # 精神状态：清醒/嗜睡/烦躁/意识模糊
    sleep: str | None                # 睡眠：正常/差/入睡困难/多睡
    appetite: str | None             # 食欲：正常/减退/无食欲/不能进食
    bowel: str | None                # 大便：正常/便秘/腹泻/黑便/血便
    urination: str | None            # 小便：正常/减少/尿频/尿痛/无尿
    weight_change: str | None        # 体重变化：无/近期明显下降/近期增加


class SymptomData(TypedDict, total=False):
    """从对话中提取的结构化症状数据（供 ClinicalNode 使用）"""
    raw_descriptions: list[str]
    onset: str | None
    provocation: str | None
    quality: str | None
    region: str | None
    severity: str | None
    max_temperature: str | None
    frequency: str | None
    time_pattern: str | None
    radiation: str | None
    accompanying: list[str]
    relevant_history: str | None
    medications: list[str]
    allergies: list[str]
    general_condition: GeneralCondition | None  # 一般情况（论文新增）


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
    intake_protocol_id: str                 # 当前匹配到的主诉采集协议
    intake_overlays: list[str]              # 叠加的人群/风险 overlay
    intake_facts: list[dict]                # 已抽取的临床事实槽，供下一轮合并去重
    requested_slots: Annotated[list[str], operator.add]  # 已追问过的槽位，避免反复追问
    monitor_result: MonitorResult           # Monitor 节点：纯打分结果
    controller_decision: ControllerDecision # Controller 节点：调度决策
    intake_review: IntakeReviewStatus       # 预诊质量门状态快照
    task_tree: dict                         # 分层预诊任务树（T1/T2/T3/T4）
    follow_up_count: int                    # 已追问轮数（硬上限兜底）
    intake_complete: bool                   # True → 进入 ClinicalNode

    # ── Clinical reasoning ──
    department_candidates: list[DepartmentResult]
    urgency_level: str
    urgency_reason: str
    differential_diagnoses: list[DifferentialDiagnosis]
    risk_factors_summary: str
    clinical_missing_slots: list[str]

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
        region=None, severity=None, max_temperature=None, frequency=None, time_pattern=None,
        radiation=None, accompanying=[],
        relevant_history=None, medications=[], allergies=[],
        general_condition=None,
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


def check_minimum_fields(
    status: CollectionStatus,
    assistant_count: int = 0,
    required_fields: list[str] | tuple[str, ...] | None = None,
    required_pattern_fields: list[str] | tuple[str, ...] | None = None,
) -> tuple[bool, list[str]]:
    """
    校验最低必要字段是否已采集。
    返回 (passed, missing_fields)

    规则：
    - medications_allergies 必须是 "complete"（护士必须亲口问到）
    - 其余关键字段只要不是 "missing"（complete 或 partial 均可）
    - 宽松模式（assistant_count >= 5）：quality/severity 不再强制要求，
      避免对不适用的症状类型死追疼痛量表问题
    """
    if required_fields is not None:
        missing = []
        for field in required_fields:
            if _field_missing(status, field):
                missing.append(field)
        pattern = status.get("pattern_specific") or {}
        for field in required_pattern_fields or ():
            if pattern.get(field, "missing") == "missing":
                missing.append(f"pattern_specific.{field}")
        return len(missing) == 0, missing

    missing = []
    opqrst = status.get("opqrst") or {}
    relaxed = assistant_count >= 5   # 5轮后进入宽松模式

    # 主诉
    if status.get("chief_complaint") == "missing":
        missing.append("chief_complaint")
    # 部位
    if opqrst.get("location", "missing") == "missing":
        missing.append("opqrst.location")
    # 发作时间
    if opqrst.get("onset", "missing") == "missing":
        missing.append("opqrst.onset")
    # 性质（宽松模式下跳过）
    if not relaxed and opqrst.get("quality", "missing") == "missing":
        missing.append("opqrst.quality")
    # 严重程度（宽松模式下跳过）
    if not relaxed and opqrst.get("severity", "missing") == "missing":
        missing.append("opqrst.severity")
    # 时间特征
    if opqrst.get("time_pattern", "missing") == "missing":
        missing.append("opqrst.time_pattern")
    # 用药/过敏：必须是 complete，无论哪种模式
    if status.get("medications_allergies") != "complete":
        missing.append("medications_allergies")

    return len(missing) == 0, missing


def _field_missing(status: CollectionStatus, field: str) -> bool:
    """
    协议化最低字段校验。
    用药/过敏必须 complete；其他字段允许 partial，避免非典型症状被死追。
    """
    if field == "medications_allergies":
        return status.get(field) != "complete"
    if field.startswith("opqrst."):
        key = field.split(".", 1)[1]
        return (status.get("opqrst") or {}).get(key, "missing") == "missing"
    return status.get(field, "missing") == "missing"


def count_partial_fields(status: CollectionStatus) -> int:
    """统计 OPQRST + 顶层字段里值为 partial 的数量"""
    partial = 0
    opqrst = status.get("opqrst") or {}
    for f in ("onset", "provocation", "quality", "location", "severity", "time_pattern", "radiation"):
        if opqrst.get(f) == "partial":
            partial += 1
    for f in ("chief_complaint", "associated_symptoms", "relevant_history"):
        if status.get(f) == "partial":
            partial += 1
    return partial
