from medi.agents.triage.intake_facts import FactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.medical_record import update_medical_record


def _store(*facts: dict) -> FactStore:
    store = FactStore()
    store.merge_items(facts, source_turn=1)
    return store


def test_update_medical_record_projects_facts_into_sections() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "孩子发烧，最高39度"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "孩子发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "昨晚"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "gc.mental_status", "status": "present", "value": "精神还可以"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    record = update_medical_record(None, store=store, plan=plan)

    assert record["triage"]["protocol_id"] == "fever"
    assert record["triage"]["overlay_ids"] == ["pediatric"]
    assert record["hpi"]["chief_complaint"]["value"] == "孩子发烧"
    assert record["hpi"]["onset"]["value"] == "昨晚"
    assert record["hpi"]["specific"]["max_temperature"]["value"] == "39度"
    assert record["hpi"]["general_condition"]["mental_status"]["value"] == "精神还可以"
    assert record["ph"]["allergy_history"]["status"] == "absent"
    assert record["cc"]["draft"] == "孩子发烧"


def test_update_medical_record_preserves_existing_unmanaged_fields() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "腹痛"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "腹痛"},
    )
    current = {
        "triage": {"manual_note": "keep"},
        "hpi": {"custom_field": {"value": "keep"}},
        "ph": {},
        "cc": {"generated": {"value": "existing summary"}},
    }

    record = update_medical_record(current, store=store, plan=plan)

    assert record["triage"]["manual_note"] == "keep"
    assert record["hpi"]["custom_field"]["value"] == "keep"
    assert record["cc"]["generated"]["value"] == "existing summary"
    assert record["cc"]["draft"] == "腹痛"
