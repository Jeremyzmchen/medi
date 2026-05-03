from medi.agents.triage.intake_facts import (
    FactStore,
    collection_status_from_facts,
    required_slots_for_plan,
)
from medi.agents.triage.intake_protocols import resolve_intake_plan


def _plan(text: str):
    return resolve_intake_plan([{"role": "user", "content": text}])


def _store(*facts: dict) -> FactStore:
    store = FactStore()
    store.merge_items(facts, source_turn=1)
    return store


def test_medication_fact_does_not_complete_allergy_slot() -> None:
    plan = _plan("我发烧最高39度，吃了布洛芬")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "昨晚开始"},
        {"slot": "hpi.severity", "status": "present", "value": "最高39度"},
        {"slot": "hpi.timing", "status": "present", "value": "反复发热"},
        {"slot": "hpi.associated_symptoms", "status": "present", "value": "咽痛，没有腹泻"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "specific.antipyretics", "status": "present", "value": "布洛芬有效"},
        {"slot": "specific.associated_fever_symptoms", "status": "present", "value": "咽痛，没有腹泻"},
        {"slot": "safety.current_medications", "status": "present", "value": "布洛芬"},
    )

    status = collection_status_from_facts(store, plan, complete=False, reason="")

    assert status["medications_allergies"] == "partial"


def test_unknown_fact_is_collected_but_not_answered() -> None:
    store = _store(
        {"slot": "hpi.onset", "status": "unknown", "value": None},
    )

    assert store.is_collected("hpi.onset") is True
    assert store.is_answered("hpi.onset") is False


def test_fever_required_slots_do_not_include_measurement_method() -> None:
    plan = _plan("我从昨晚开始发烧，最高39度")
    slots = required_slots_for_plan(plan)

    assert plan.protocol_id == "fever"
    assert "specific.max_temperature" in slots
    assert "specific.measurement_method" not in slots


def test_fact_store_preserves_exposure_timeline_in_symptom_data() -> None:
    store = _store(
        {"slot": "hpi.exposure_event", "status": "present", "value": "上周去菲律宾潜水"},
        {"slot": "hpi.exposure_symptoms", "status": "absent", "value": "潜水当时或之后无耳痛"},
        {"slot": "hpi.onset", "status": "present", "value": "今天耳朵里刺痛"},
    )

    symptom_data = store.to_symptom_data(["上周去菲律宾潜水没发现耳朵痛，今天耳朵里有些刺痛"])

    assert symptom_data["exposure_event"] == "上周去菲律宾潜水"
    assert symptom_data["exposure_symptoms"] == "潜水当时或之后无耳痛"
    assert symptom_data["onset"] == "今天耳朵里刺痛"


def test_fact_store_corrects_exposure_misattributed_as_onset() -> None:
    store = _store(
        {"slot": "hpi.onset", "status": "present", "value": "上周潜水后", "confidence": 0.9},
        {"slot": "hpi.onset", "status": "present", "value": "今天耳朵里刺痛", "confidence": 0.92},
    )

    assert store.value("hpi.onset") == "今天耳朵里刺痛"
