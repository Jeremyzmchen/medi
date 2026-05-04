from medi.agents.triage.clinical_facts import ClinicalFactStore, required_slots_for_plan
from medi.agents.triage.graph.nodes.intake_prompter_node import _fallback_question
from medi.agents.triage.intake_protocols import (
    resolve_intake_plan,
    question_for_missing_field,
)


def _messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_resolve_fever_protocol_with_pediatric_overlay() -> None:
    plan = resolve_intake_plan(_messages("我儿子发烧到39度，今天还没退"))

    assert plan.protocol_id == "fever"
    assert plan.overlay_ids == ["pediatric"]
    pattern_keys = {key for key, _ in plan.pattern_required}
    assert "max_temperature" in pattern_keys
    assert "measurement_method" not in pattern_keys
    assert "age" in pattern_keys
    assert "mental_status" in pattern_keys


def test_chest_pain_protocol_requires_radiation_and_provocation() -> None:
    plan = resolve_intake_plan(_messages("昨天开始胸口压榨样疼痛，活动后加重"))

    assert plan.protocol_id == "chest_pain"
    assert "opqrst.radiation" in plan.required_fields
    assert "opqrst.provocation" in plan.required_fields
    assert "medications_allergies" in plan.required_fields


def test_unknown_complaint_falls_back_to_generic_opqrst() -> None:
    plan = resolve_intake_plan(_messages("我最近不太舒服，说不上来哪里怪怪的"))

    assert plan.protocol_id == "generic_opqrst"
    assert "opqrst.location" in plan.required_fields
    assert "opqrst.quality" in plan.required_fields


def test_dizziness_protocol_wins_for_head_dizziness() -> None:
    plan = resolve_intake_plan(_messages("我今天一直头晕，站不稳"))

    assert plan.protocol_id == "dizziness_syncope"


def test_negated_protocol_keyword_does_not_trigger_match() -> None:
    plan = resolve_intake_plan(_messages("我没有胸痛，就是头晕"))

    assert plan.protocol_id == "dizziness_syncope"


def test_assistant_questions_do_not_pollute_protocol_matching() -> None:
    messages = [
        {"role": "user", "content": "我头晕，站起来更明显"},
        {"role": "assistant", "content": "有没有胸痛、呼吸困难或大汗？"},
    ]

    plan = resolve_intake_plan(messages)

    assert plan.protocol_id == "dizziness_syncope"


def test_non_generic_protocol_can_be_locked_across_turns() -> None:
    messages = [
        {"role": "user", "content": "我发烧到39度"},
        {"role": "assistant", "content": "还有什么伴随症状吗？"},
        {"role": "user", "content": "后来有点拉稀"},
    ]

    plan = resolve_intake_plan(messages, fixed_protocol_id="fever")

    assert plan.protocol_id == "fever"


def test_generic_protocol_can_upgrade_when_user_adds_specific_complaint() -> None:
    messages = [
        {"role": "user", "content": "我不太舒服"},
        {"role": "assistant", "content": "具体哪里不舒服？"},
        {"role": "user", "content": "其实是头晕"},
    ]

    plan = resolve_intake_plan(messages, fixed_protocol_id=None)

    assert plan.protocol_id == "dizziness_syncope"


def test_pregnancy_overlay_can_come_from_health_profile() -> None:
    class Profile:
        age = 30
        chronic_conditions = ["怀孕32周"]
        current_medications = []
        allergies = []

    plan = resolve_intake_plan(_messages("我有点腹痛"), profile_snapshot=Profile())

    assert "pregnancy" in plan.overlay_ids


def test_protocol_minimum_fields_do_not_force_generic_pain_slots() -> None:
    plan = resolve_intake_plan(_messages("我发烧39度，从昨天晚上开始，吃了布洛芬退了一点"))
    slots = required_slots_for_plan(plan)

    assert "specific.max_temperature" in slots
    assert "specific.measurement_method" not in slots
    assert "hpi.character" not in slots
    assert "hpi.location" not in slots


def test_pattern_missing_field_generates_protocol_question() -> None:
    plan = resolve_intake_plan(_messages("孩子发烧了"))

    question = question_for_missing_field(["pattern_specific.age"], plan)

    assert "发热" in question
    assert "年龄" in question


def test_partial_medications_allergies_asks_allergy_only() -> None:
    plan = resolve_intake_plan(_messages("我发烧，吃了布洛芬"))
    store = ClinicalFactStore()
    store.merge_items(
        [{"slot": "safety.current_medications", "status": "present", "value": "布洛芬"}],
        source_turn=1,
    )

    question = _fallback_question("safety.allergies", plan, store)

    assert "过敏" in question
    assert "目前在服用什么药物" not in question
