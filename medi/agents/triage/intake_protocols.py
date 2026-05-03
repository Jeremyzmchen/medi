"""
主诉采集协议注册表。

协议只覆盖高频/高风险问诊路径，未命中时回退到 generic_opqrst。
人群和风险因素用 overlay 叠加，避免为“儿童发热”“老人腹痛”等组合爆炸式建枚举。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


StatusField = str
PatternField = tuple[str, str]


@dataclass(frozen=True)
class IntakeProtocol:
    id: str
    label: str
    keywords: tuple[str, ...]
    required_fields: tuple[StatusField, ...]
    pattern_required: tuple[PatternField, ...] = ()
    red_flags: tuple[str, ...] = ()
    question_focus: tuple[str, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class IntakeOverlay:
    id: str
    label: str
    keywords: tuple[str, ...]
    required_fields: tuple[StatusField, ...] = ()
    pattern_required: tuple[PatternField, ...] = ()
    red_flags: tuple[str, ...] = ()
    question_focus: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedIntakePlan:
    protocol: IntakeProtocol
    overlays: tuple[IntakeOverlay, ...]
    required_fields: tuple[StatusField, ...]
    pattern_required: tuple[PatternField, ...]
    red_flags: tuple[str, ...]
    question_focus: tuple[str, ...]

    @property
    def protocol_id(self) -> str:
        return self.protocol.id

    @property
    def protocol_label(self) -> str:
        return self.protocol.label

    @property
    def overlay_ids(self) -> list[str]:
        return [o.id for o in self.overlays]

    def prompt_section(
        self,
        completed_fields: Iterable[str] | None = None,
        completed_pattern_keys: Iterable[str] | None = None,
    ) -> str:
        completed = set(completed_fields or ())
        completed_patterns = set(completed_pattern_keys or ())
        overlays = "、".join(o.label for o in self.overlays) or "无"
        required = "\n".join(
            f"- {_status_marker(f in completed)} {field_label(f)}"
            for f in self.required_fields
        )
        pattern = "\n".join(
            f"- {_status_marker(key in completed_patterns)} pattern_specific.{key}: {label}"
            for key, label in self.pattern_required
        ) or "- 无"
        red_flags = "\n".join(f"- {item}" for item in self.red_flags) or "- 无"
        focus = "；".join(self.question_focus) or "按缺失字段优先级自然追问"

        return f"""[本轮采集协议]
基础协议：{self.protocol_label}（{self.protocol_id}）
叠加规则：{overlays}

最低必须采集字段：
{required}

症状特异性字段：
{pattern}

需要重点排查的危险信号：
{red_flags}

追问重点：{focus}

要求：
- pattern_specific 必须包含上方列出的 key，取值仍为 complete|partial|missing
- 如果某个症状特异性字段患者已明确回答，请标为 complete 或 partial
- 追问时优先补齐最低必须采集字段，再补症状特异性字段"""


DEFAULT_REQUIRED_FIELDS: tuple[StatusField, ...] = (
    "chief_complaint",
    "opqrst.location",
    "opqrst.onset",
    "opqrst.quality",
    "opqrst.severity",
    "opqrst.time_pattern",
    "medications_allergies",
)


PROTOCOLS: tuple[IntakeProtocol, ...] = (
    IntakeProtocol(
        id="chest_pain",
        label="胸痛/胸闷",
        keywords=("胸痛", "胸闷", "胸口", "心痛", "心悸", "胸部不适"),
        required_fields=(
            "chief_complaint",
            "opqrst.location",
            "opqrst.onset",
            "opqrst.quality",
            "opqrst.severity",
            "opqrst.time_pattern",
            "opqrst.radiation",
            "opqrst.provocation",
            "medications_allergies",
        ),
        pattern_required=(
            ("exertional_related", "是否与活动、情绪激动或休息相关"),
            ("dyspnea_sweating", "是否伴呼吸困难、出汗、恶心或濒死感"),
            ("cardiovascular_history", "是否有高血压、冠心病、糖尿病等心血管风险"),
        ),
        red_flags=("压榨样胸痛", "胸痛伴呼吸困难/大汗", "胸痛放射至左臂/下颌/背部", "晕厥或濒死感"),
        question_focus=("放射痛", "活动相关性", "呼吸困难/大汗", "心血管风险"),
        priority=90,
    ),
    IntakeProtocol(
        id="dyspnea",
        label="呼吸困难",
        keywords=("呼吸困难", "喘不过气", "气短", "憋气", "呼吸急促", "喘"),
        required_fields=(
            "chief_complaint",
            "opqrst.onset",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("rest_or_exertion", "静息时还是活动后出现"),
            ("chest_pain_or_wheeze", "是否伴胸痛、喘鸣、咳痰"),
            ("cyanosis_or_spo2", "是否口唇发紫或有血氧数据"),
        ),
        red_flags=("静息状态呼吸困难", "口唇发紫", "不能完整说话", "胸痛伴气短"),
        question_focus=("严重程度", "是否静息发作", "胸痛/喘鸣", "血氧或紫绀"),
        priority=88,
    ),
    IntakeProtocol(
        id="fever",
        label="发热",
        keywords=("发热", "发烧", "高烧", "低烧", "体温", "退烧", "烧到"),
        required_fields=(
            "chief_complaint",
            "opqrst.onset",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("max_temperature", "最高体温"),
            ("antipyretics", "是否用过退烧药以及效果"),
            ("associated_fever_symptoms", "寒战、咳嗽、咽痛、皮疹、腹泻等伴随症状"),
        ),
        red_flags=("体温 >= 40℃", "意识改变或抽搐", "颈项强直", "呼吸困难", "持续高热不退"),
        question_focus=("最高体温", "持续时间", "体温测量方式", "退烧药效果", "伴随症状"),
        priority=70,
    ),
    IntakeProtocol(
        id="abdominal_pain",
        label="腹痛",
        keywords=("腹痛", "肚子痛", "胃痛", "右上腹", "右下腹", "腹部", "绞痛"),
        required_fields=(
            "chief_complaint",
            "opqrst.location",
            "opqrst.onset",
            "opqrst.quality",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("vomiting_diarrhea", "是否伴恶心、呕吐、腹泻"),
            ("stool_or_bleeding", "是否黑便、血便或呕血"),
            ("food_related", "是否与进食、饮酒或油腻饮食相关"),
        ),
        red_flags=("突发剧烈腹痛", "腹痛伴发热/持续呕吐", "黑便/血便/呕血", "右下腹进行性疼痛"),
        question_focus=("具体部位", "进食相关", "呕吐腹泻", "消化道出血信号"),
        priority=65,
    ),
    IntakeProtocol(
        id="headache",
        label="头痛",
        keywords=("头痛", "头疼", "偏头痛", "后脑痛", "后脑疼", "太阳穴痛", "太阳穴疼"),
        required_fields=(
            "chief_complaint",
            "opqrst.location",
            "opqrst.onset",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("sudden_or_worst", "是否突发或一生最严重头痛"),
            ("neuro_deficits", "是否伴肢体无力、麻木、口角歪斜或言语不清"),
            ("fever_neck_stiffness", "是否伴发热、颈项强直或喷射性呕吐"),
        ),
        red_flags=("雷击样头痛", "头痛伴神经功能缺损", "发热伴颈项强直", "外伤后头痛"),
        question_focus=("起病速度", "严重程度", "神经症状", "发热/颈项强直"),
        priority=60,
    ),
    IntakeProtocol(
        id="trauma",
        label="外伤",
        keywords=("摔", "撞", "扭", "外伤", "受伤", "骨折", "流血", "刀伤", "崴脚"),
        required_fields=(
            "chief_complaint",
            "opqrst.location",
            "opqrst.onset",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("injury_mechanism", "受伤机制和高度/速度"),
            ("function_or_weight_bearing", "是否能活动、负重或使用患处"),
            ("bleeding_deformity", "是否出血、畸形、开放伤口或麻木"),
        ),
        red_flags=("头颈部外伤伴意识异常", "开放性骨折", "大量出血", "肢体麻木或无法负重"),
        question_focus=("受伤机制", "活动/负重能力", "出血畸形", "麻木无力"),
        priority=58,
    ),
    IntakeProtocol(
        id="diarrhea_vomiting",
        label="腹泻/呕吐",
        keywords=("腹泻", "拉肚子", "拉稀", "呕吐", "吐了", "恶心"),
        required_fields=(
            "chief_complaint",
            "opqrst.onset",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("frequency", "腹泻或呕吐次数"),
            ("stool_or_vomit_character", "大便或呕吐物性状、颜色"),
            ("dehydration_signs", "口渴、尿少、乏力等脱水表现"),
            ("blood_or_black_stool", "是否血便、黑便或呕血"),
        ),
        red_flags=("血便/黑便", "持续呕吐不能进水", "明显脱水", "高热伴腹泻"),
        question_focus=("次数", "性状", "脱水表现", "是否带血"),
        priority=55,
    ),
    IntakeProtocol(
        id="rash_allergy",
        label="皮疹/过敏",
        keywords=("皮疹", "红疹", "起疹", "荨麻疹", "过敏", "瘙痒", "肿了"),
        required_fields=(
            "chief_complaint",
            "opqrst.location",
            "opqrst.onset",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("itch_or_pain", "是否瘙痒或疼痛"),
            ("trigger_exposure", "近期食物、药物、接触物或虫咬暴露"),
            ("mucosal_or_swelling", "是否口唇/眼睑/喉头肿胀或黏膜受累"),
        ),
        red_flags=("喉头紧缩或呼吸困难", "口唇舌头肿胀", "全身皮疹伴低血压/晕厥", "皮疹伴高热"),
        question_focus=("暴露诱因", "范围变化", "口唇喉头肿胀", "呼吸情况"),
        priority=50,
    ),
    IntakeProtocol(
        id="dizziness_syncope",
        label="头晕/晕厥",
        keywords=("头晕", "眩晕", "晕倒", "晕厥", "眼前发黑", "站不稳"),
        required_fields=(
            "chief_complaint",
            "opqrst.onset",
            "opqrst.severity",
            "opqrst.time_pattern",
            "medications_allergies",
        ),
        pattern_required=(
            ("loss_of_consciousness", "是否真正失去意识以及持续多久"),
            ("neuro_deficits", "是否伴肢体无力、麻木、言语不清"),
            ("palpitations_chest_pain", "是否伴心悸、胸痛或气短"),
        ),
        red_flags=("晕厥伴胸痛/心悸", "持续神经功能缺损", "反复晕厥", "外伤后晕厥"),
        question_focus=("是否失去意识", "心悸胸痛", "神经症状", "诱发体位"),
        priority=48,
    ),
    IntakeProtocol(
        id="generic_opqrst",
        label="通用 OPQRST",
        keywords=(),
        required_fields=DEFAULT_REQUIRED_FIELDS,
        question_focus=("部位", "起病时间", "性质", "严重程度", "持续/间歇"),
        priority=0,
    ),
)


OVERLAYS: tuple[IntakeOverlay, ...] = (
    IntakeOverlay(
        id="pediatric",
        label="儿童",
        keywords=("孩子", "宝宝", "小孩", "儿童", "幼儿", "儿子", "女儿", "月龄", "岁半"),
        pattern_required=(
            ("age", "年龄或月龄"),
            ("mental_status", "精神反应、嗜睡或烦躁情况"),
            ("intake_urination", "饮水、进食和尿量情况"),
        ),
        red_flags=("3 月龄以下发热", "精神差/嗜睡", "抽搐", "尿量明显减少"),
        question_focus=("年龄/月龄", "精神反应", "饮水尿量"),
    ),
    IntakeOverlay(
        id="elderly",
        label="老年",
        keywords=("老人", "老年", "高龄", "爷爷", "奶奶", "外公", "外婆"),
        pattern_required=(
            ("baseline_function", "平时活动能力和本次变化"),
            ("fall_or_confusion", "是否跌倒、意识混乱或反应变差"),
        ),
        red_flags=("意识混乱", "跌倒后疼痛或活动受限", "基础病突然加重"),
        question_focus=("平时基础状态", "意识变化", "跌倒风险"),
    ),
    IntakeOverlay(
        id="pregnancy",
        label="妊娠/产后",
        keywords=("怀孕", "孕", "妊娠", "产后", "月经推迟"),
        pattern_required=(
            ("pregnancy_weeks", "孕周或产后时间"),
            ("vaginal_bleeding_or_abdominal_pain", "是否阴道流血、腹痛或胎动异常"),
        ),
        red_flags=("孕期腹痛或阴道流血", "严重头痛/视物模糊", "胎动明显减少"),
        question_focus=("孕周", "阴道流血", "腹痛/胎动"),
    ),
    IntakeOverlay(
        id="immunocompromised",
        label="免疫抑制/肿瘤治疗",
        keywords=("化疗", "放疗", "免疫抑制", "移植", "白细胞低", "肿瘤", "艾滋", "HIV", "长期激素"),
        pattern_required=(
            ("immunosuppression_status", "免疫抑制原因和最近治疗时间"),
            ("infection_exposure", "近期感染接触、导管或伤口情况"),
        ),
        red_flags=("免疫抑制患者发热", "白细胞低伴感染症状", "导管相关感染表现"),
        question_focus=("近期治疗", "感染暴露", "导管/伤口"),
    ),
)


FIELD_LABELS: dict[str, str] = {
    "chief_complaint": "主诉",
    "associated_symptoms": "伴随症状",
    "relevant_history": "相关既往史",
    "medications_allergies": "当前用药和过敏史",
    "opqrst.onset": "发作时间/诱因",
    "opqrst.provocation": "加重或缓解因素",
    "opqrst.quality": "症状性质",
    "opqrst.location": "具体部位",
    "opqrst.severity": "严重程度",
    "opqrst.time_pattern": "持续时间和时间特征",
    "opqrst.radiation": "放射痛/扩散",
}


FIELD_QUESTIONS: dict[str, str] = {
    "chief_complaint": "您今天主要是哪里不舒服，能描述一下吗？",
    "opqrst.location": "您能告诉我具体是哪个部位不舒服吗？",
    "opqrst.onset": "这个症状是什么时候开始的？",
    "opqrst.quality": "您能描述一下是什么样的感觉或特征吗？",
    "opqrst.severity": "这个不适大概有多严重？比如疼痛 0 到 10 分，或发热的最高体温、腹泻次数等。",
    "opqrst.time_pattern": "这个症状是一直持续还是时好时坏？大概持续多久了？",
    "opqrst.provocation": "有什么情况会让它加重或缓解吗？",
    "opqrst.radiation": "这个不适会向其他部位扩散或放射吗？",
    "associated_symptoms": "除了这个主要不适，还有其他伴随症状吗？",
    "relevant_history": "以前有过类似情况，或有什么相关疾病史吗？",
    "medications_allergies": "您目前在服用什么药物吗？有没有药物或食物过敏？",
}


def resolve_intake_plan(
    messages: Iterable[dict],
    health_profile=None,
    fixed_protocol_id: str | None = None,
) -> ResolvedIntakePlan:
    text = _conversation_text(messages)
    protocol = _protocol_by_id(fixed_protocol_id) if fixed_protocol_id else _match_protocol(text)
    overlays = _match_overlays(text, health_profile)    # 叠加规则：老人/小孩/孕妇/...

    required_fields = _unique(
        protocol.required_fields,
        *(o.required_fields for o in overlays),
    )
    pattern_required = _unique_pattern(
        protocol.pattern_required,
        *(o.pattern_required for o in overlays),
    )
    red_flags = _unique(protocol.red_flags, *(o.red_flags for o in overlays))
    question_focus = _unique(protocol.question_focus, *(o.question_focus for o in overlays))

    return ResolvedIntakePlan(
        protocol=protocol,
        overlays=overlays,
        required_fields=required_fields,
        pattern_required=pattern_required,
        red_flags=red_flags,
        question_focus=question_focus,
    )


def field_label(field: str) -> str:
    if field.startswith("pattern_specific."):
        return field.removeprefix("pattern_specific.")
    return FIELD_LABELS.get(field, field)


def _status_marker(done: bool) -> str:
    return "[✓ 已采集]" if done else "[待采集]"


def question_for_missing_field(
    missing_fields: list[str],
    plan: ResolvedIntakePlan | None = None,
) -> str:
    for field in missing_fields:
        if field in FIELD_QUESTIONS:
            return FIELD_QUESTIONS[field]
        if field.startswith("pattern_specific.") and plan is not None:
            key = field.removeprefix("pattern_specific.")
            label = dict(plan.pattern_required).get(key, key)
            return f"关于{plan.protocol_label}，还需要了解{label}，方便的话请补充一下？"
    return "请问还有什么重要信息需要告诉我吗？"


def _conversation_text(messages: Iterable[dict]) -> str:
    return " ".join(
        str(m.get("content", ""))
        for m in messages
        if m.get("role") == "user"
    )


def _match_protocol(text: str) -> IntakeProtocol:
    best = next(p for p in PROTOCOLS if p.id == "generic_opqrst")
    best_score = 0
    for protocol in PROTOCOLS:
        if protocol.id == "generic_opqrst":
            continue
        matches = sum(1 for kw in protocol.keywords if kw and _keyword_asserted(text, kw))
        if matches == 0:
            continue
        score = matches * 100 + protocol.priority
        if score > best_score:
            best = protocol
            best_score = score
    return best


def _protocol_by_id(protocol_id: str | None) -> IntakeProtocol:
    if protocol_id:
        for protocol in PROTOCOLS:
            if protocol.id == protocol_id:
                return protocol
    return next(p for p in PROTOCOLS if p.id == "generic_opqrst")


def _match_overlays(text: str, health_profile=None) -> tuple[IntakeOverlay, ...]:
    matched: list[IntakeOverlay] = []
    for overlay in OVERLAYS:
        if any(kw in text for kw in overlay.keywords):
            matched.append(overlay)

    age = getattr(health_profile, "age", None)
    if age is not None and age < 14:
        _append_overlay(matched, "pediatric")
    if age is not None and age >= 65:
        _append_overlay(matched, "elderly")

    conditions = " ".join(getattr(health_profile, "chronic_conditions", []) or [])
    meds = " ".join(getattr(health_profile, "current_medications", []) or [])
    allergies = " ".join(getattr(health_profile, "allergies", []) or [])
    pregnancy_weeks = getattr(health_profile, "pregnancy_weeks", None)
    is_pregnant = bool(getattr(health_profile, "is_pregnant", False))
    profile_text = f"{conditions} {meds} {allergies}"
    if (
        is_pregnant
        or pregnancy_weeks is not None
        or any(kw in profile_text for kw in ("怀孕", "妊娠", "孕期", "孕周", "产后"))
    ):
        _append_overlay(matched, "pregnancy")
    if any(kw in profile_text for kw in ("肿瘤", "化疗", "移植", "免疫抑制", "长期激素", "HIV", "艾滋")):
        _append_overlay(matched, "immunocompromised")

    return tuple(matched)


def _keyword_asserted(text: str, keyword: str) -> bool:
    """
    关键词正向命中。

    过滤“没有胸痛”“无腹泻”“不头晕”等否定语境，避免把否认症状当主诉。
    """
    for match in re.finditer(re.escape(keyword), text):
        prefix = text[max(0, match.start() - 6):match.start()]
        if any(neg in prefix for neg in ("没有", "无", "不", "不是", "未", "否认")):
            continue
        return True
    return False


def _append_overlay(items: list[IntakeOverlay], overlay_id: str) -> None:
    if any(o.id == overlay_id for o in items):
        return
    overlay = next(o for o in OVERLAYS if o.id == overlay_id)
    items.append(overlay)


def _unique(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def _unique_pattern(*groups: Iterable[PatternField]) -> tuple[PatternField, ...]:
    seen: set[str] = set()
    result: list[PatternField] = []
    for group in groups:
        for key, label in group:
            if key not in seen:
                seen.add(key)
                result.append((key, label))
    return tuple(result)
