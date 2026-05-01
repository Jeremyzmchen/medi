from medi.agents.triage.intake_facts import FactStore, collection_status_from_facts
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.intake_rules import extract_deterministic_facts
from medi.agents.triage.graph.nodes.intake_review_node import review_intake_quality


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
    store = FactStore()
    store.merge_items([
        {"slot": "hpi.chief_complaint", "status": "present", "value": "发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "昨晚"},
        {"slot": "hpi.severity", "status": "present", "value": "最高39度"},
        {"slot": "hpi.timing", "status": "present", "value": "昨晚开始"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "specific.associated_fever_symptoms", "status": "present", "value": "咽痛"},
        *extract_deterministic_facts(messages, protocol_id=plan.protocol_id),
    ], source_turn=1)

    status = collection_status_from_facts(store, plan, complete=False, reason="")
    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=[],
        assistant_count=2,
    )

    assert status["medications_allergies"] == "partial"
    assert review.next_best_slot == "safety.allergies"
    assert "过敏" in (review.next_best_question or "")
    assert "平时在用什么药" not in (review.next_best_question or "")


def test_deterministic_rules_extract_no_drug_allergy() -> None:
    facts = extract_deterministic_facts(
        _messages("布洛芬，没有药物过敏"),
        protocol_id="fever",
    )

    by_slot = {fact["slot"]: fact for fact in facts}

    assert by_slot["safety.current_medications"]["value"] == "布洛芬"
    assert by_slot["safety.allergies"]["status"] == "absent"


def test_exposure_timeline_keeps_diving_separate_from_symptom_onset() -> None:
    facts = extract_deterministic_facts(
        _messages("上周去菲律宾潜水没发现耳朵痛，今天耳朵里有些刺痛")
    )
    by_slot = {fact["slot"]: fact for fact in facts}

    assert by_slot["hpi.exposure_event"]["value"] == "上周去菲律宾潜水"
    assert by_slot["hpi.exposure_symptoms"]["status"] == "absent"
    assert by_slot["hpi.exposure_symptoms"]["value"] == "潜水当时或之后无耳痛"
    assert by_slot["hpi.onset"]["value"] == "今天耳朵里有些刺痛"
