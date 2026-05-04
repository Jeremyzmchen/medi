"""Prompt builders for triage clinical fact extraction."""

from __future__ import annotations

from medi.agents.triage.clinical_facts import (
    ClinicalFactStore,
    extraction_slot_prompt,
)
from medi.agents.triage.intake_protocols import ResolvedIntakePlan


FACT_EXTRACT_SYSTEM_PROMPT = """你是预诊分诊系统里的临床事实抽取器，不是对话生成器。

你的任务是从护士-患者对话中抽取事实，严格输出 JSON，不要生成追问、建议或诊断。

抽取规则：
- 只记录患者明确说过的信息，不能把护士问题当作患者回答。
- 患者说”没有””否认””不是”某症状时，status 用 absent。
- 患者提到”布洛芬””退烧药””降压药”等各种药物名称时，用 safety.current_medications。
- 过敏史必须单独抽取到 safety.allergies；不要因为患者提到用药，就推断过敏史已回答。
- 如果患者没有回答某个槽位，不要输出这个槽位，除非患者明确表示不知道。
- value 尽量保留患者原话里的关键短语，例如”三天””最高39度””咽痛，没有腹泻”。
- confidence 使用 0 到 1 的小数。
- hpi.onset 只记录症状真正开始的时间，不要把例如去潜水、飞行、游泳、外伤等暴露事件时间当作起病时间。
- 症状出现前的相关经历抽取到 hpi.exposure_event，例如“上周菲律宾潜水”。
- 如果患者明确说暴露当时或暴露后没有某症状，抽取到 hpi.exposure_symptoms，status 用 absent，例如“潜水当时及之后无耳痛”。

一般情况（gc.*）抽取规则：
- 患者描述精神差/嗜睡/烦躁/意识模糊时，抽取到 gc.mental_status。
- 患者描述睡眠不好/失眠/嗜睡时，抽取到 gc.sleep。
- 患者描述不想吃/没胃口/能吃时，抽取到 gc.appetite。
- 患者描述大便异常（便秘/腹泻/黑便/血便）时，抽取到 gc.bowel。
- 患者描述小便异常（尿少/尿频/尿痛/无尿）时，抽取到 gc.urination。
- 患者描述体重明显变化时，抽取到 gc.weight_change。

现病史扩展槽（hpi.*）抽取规则：
- 患者描述每次发作持续多久或总病程多久时，抽取到 hpi.duration。
- 患者描述症状逐渐加重、减轻、反复或无明显变化时，抽取到 hpi.progression。
- 患者描述发病后做过检查及结果时，抽取到 hpi.diagnostic_history。
- 患者描述发病后用药、处理及效果时，抽取到 hpi.therapeutic_history。

既往史（ph.*）抽取规则：
- 患者描述既往重要疾病、传染病、慢性病时，抽取到 ph.disease_history。
- 患者描述疫苗接种情况时，抽取到 ph.immunization_history。
- 患者描述手术史、外伤史、输血史时，分别抽取到 ph.surgical_history、ph.trauma_history、ph.blood_transfusion_history。
- 患者描述过敏史时，优先抽取到 safety.allergies；若明确是在既往史段落回答，也可以同时抽取到 ph.allergy_history。

允许的 status：
- present: 明确存在或明确回答了肯定信息
- absent: 明确否认或明确回答没有
- partial: 有信息但不完整
- unknown: 患者明确表示不知道/不清楚
- not_applicable: 对当前症状明显不适用且患者已表达

输出格式：
{
  “facts”: [
    {
      “slot”: “hpi.onset”,
      “status”: “present”,
      “value”: “昨晚开始”,
      “evidence”: “我从昨晚开始发烧”,
      “confidence”: 0.9
    }
  ]
}
"""


def build_fact_extract_prompt(
    intake_plan: ResolvedIntakePlan,
    store: ClinicalFactStore,
) -> str:
    plan_section = intake_plan.prompt_section(
        completed_fields=_completed_protocol_fields(intake_plan, store),
        completed_pattern_keys=_completed_pattern_keys(intake_plan, store),
    )
    return f"""{FACT_EXTRACT_SYSTEM_PROMPT}

当前主诉采集重点：
{plan_section}

允许抽取的槽位：
{extraction_slot_prompt(intake_plan)}

当前已知事实：
{store.prompt_context()}
"""


def _completed_protocol_fields(
    intake_plan: ResolvedIntakePlan,
    store: ClinicalFactStore,
) -> list[str]:
    completed: list[str] = []
    field_slots = {
        "chief_complaint": ("hpi.chief_complaint",),
        "opqrst.location": ("hpi.location",),
        "opqrst.onset": ("hpi.onset",),
        "opqrst.quality": ("hpi.character",),
        "opqrst.severity": ("hpi.severity", "specific.max_temperature", "specific.frequency"),
        "opqrst.time_pattern": ("hpi.timing",),
        "opqrst.provocation": ("hpi.aggravating_alleviating",),
        "opqrst.radiation": ("hpi.radiation",),
        "associated_symptoms": ("hpi.associated_symptoms", "specific.associated_fever_symptoms"),
        "relevant_history": ("hpi.relevant_history",),
        "medications_allergies": ("safety.current_medications", "safety.allergies"),
    }
    for field in intake_plan.required_fields:
        slots = field_slots.get(field, ())
        if field == "medications_allergies":
            answered = all(store.is_answered(slot) for slot in slots)
        else:
            answered = any(store.is_answered(slot) for slot in slots)
        if slots and answered:
            completed.append(field)
    return completed


def _completed_pattern_keys(
    intake_plan: ResolvedIntakePlan,
    store: ClinicalFactStore,
) -> list[str]:
    return [
        key for key, _ in intake_plan.pattern_required
        if store.is_answered(f"specific.{key}")
    ]

