from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.intake_rules import extract_deterministic_facts
from medi.agents.triage.graph.nodes.intake_prompter_node import _fallback_question


def _messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_deterministic_rules_extract_ibuprofen_as_current_medication() -> None:
    facts = extract_deterministic_facts(
        _messages("我从昨晚开始发烧，最高39度，吃了布洛芬退了一点"),
        protocol_id="fever",
    )

    by_slot = {fact["slot"]: fact for fact in facts}

    assert by_slot["safety.current_medications"]["value"] == "布洛芬"
    assert by_slot["specific.antipyretics"]["value"].startswith("布洛芬")
    assert "退了" in by_slot["specific.antipyretics"]["value"]


def test_ibuprofen_in_first_turn_prevents_reasking_current_medication() -> None:
    messages = _messages("我从昨晚开始发烧，最高39度，吃了布洛芬退了一点")
    plan = resolve_intake_plan(messages)
    store = ClinicalFactStore()
    store.merge_items([
        {"slot": "hpi.chief_complaint", "status": "present", "value": "发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "昨晚"},
        {"slot": "hpi.severity", "status": "present", "value": "最高39度"},
        {"slot": "hpi.timing", "status": "present", "value": "昨晚开始"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "specific.associated_fever_symptoms", "status": "present", "value": "咽痛"},
        *extract_deterministic_facts(messages, protocol_id=plan.protocol_id),
    ], source_turn=1)

    question = _fallback_question("safety.allergies", plan, store)

    assert store.is_answered("safety.current_medications") is True
    assert store.is_answered("safety.allergies") is False
    assert "过敏" in question
    assert "平时在用什么药" not in question


def test_deterministic_rules_extract_no_drug_allergy() -> None:
    facts = extract_deterministic_facts(
        _messages("布洛芬，没有药物过敏"),
        protocol_id="fever",
    )

    by_slot = {fact["slot"]: fact for fact in facts}

    assert by_slot["safety.current_medications"]["value"] == "布洛芬"
    assert by_slot["safety.allergies"]["status"] == "absent"


def test_deterministic_rules_use_question_context_for_short_negative_answers() -> None:
    facts = extract_deterministic_facts([
        {"role": "assistant", "content": "有没有去看过医生或者接受过任何治疗？"},
        {"role": "user", "content": "没有"},
        {"role": "assistant", "content": "以前有没有得过传染病，比如肺结核或肝炎等？"},
        {"role": "user", "content": "没有"},
    ])

    by_slot = {fact["slot"]: fact for fact in facts}

    assert by_slot["hpi.diagnostic_history"]["status"] == "absent"
    assert by_slot["hpi.therapeutic_history"]["status"] == "absent"
    assert by_slot["ph.disease_history"]["status"] == "absent"


def test_exposure_timeline_keeps_diving_separate_from_symptom_onset() -> None:
    facts = extract_deterministic_facts(
        _messages("上周去菲律宾潜水没发现耳朵痛，今天耳朵里有些刺痛")
    )
    by_slot = {fact["slot"]: fact for fact in facts}

    assert by_slot["hpi.exposure_event"]["value"] == "上周去菲律宾潜水"
    assert by_slot["hpi.exposure_symptoms"]["status"] == "absent"
    assert by_slot["hpi.exposure_symptoms"]["value"] == "潜水当时或之后无耳痛"
    assert by_slot["hpi.onset"]["value"] == "今天耳朵里有些刺痛"
