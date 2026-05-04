from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.preconsultation_record import update_preconsultation_record


def _store(*facts: dict) -> ClinicalFactStore:
    store = ClinicalFactStore()
    store.merge_items(facts, source_turn=1)
    return store


def test_update_preconsultation_record_projects_facts_into_t_sections() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "孩子发烧，最高39度"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "孩子发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "昨晚"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "gc.mental_status", "status": "present", "value": "精神还可以"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    record = update_preconsultation_record(None, store=store, plan=plan)

    assert record["meta"]["schema_version"] == "preconsultation_record.v1"
    assert record["t1_triage"]["protocol_id"] == "fever"
    assert record["t1_triage"]["overlay_ids"] == ["pediatric"]
    assert record["t2_hpi"]["chief_complaint"]["value"] == "孩子发烧"
    assert record["t2_hpi"]["onset"]["value"] == "昨晚"
    assert record["t2_hpi"]["specific"]["max_temperature"]["value"] == "39度"
    assert record["t2_hpi"]["general_condition"]["mental_status"]["value"] == "精神还可以"
    assert record["t3_background"]["allergy_history"]["status"] == "absent"
    assert record["t4_chief_complaint"]["draft"] == "孩子发烧"


def test_update_preconsultation_record_preserves_existing_unmanaged_fields() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "腹痛"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "腹痛"},
    )
    current = {
        "t1_triage": {"manual_note": "keep"},
        "t2_hpi": {"custom_field": {"value": "keep"}},
        "t3_background": {},
        "t4_chief_complaint": {"generated": {"value": "existing summary"}},
    }

    record = update_preconsultation_record(current, store=store, plan=plan)

    assert record["t1_triage"]["manual_note"] == "keep"
    assert record["t2_hpi"]["custom_field"]["value"] == "keep"
    assert record["t4_chief_complaint"]["generated"]["value"] == "existing summary"
    assert record["t4_chief_complaint"]["draft"] == "腹痛"


def test_update_preconsultation_record_migrates_legacy_in_memory_sections() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "头痛"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "头痛"},
    )

    record = update_preconsultation_record(
        {
            "triage": {"manual_note": "legacy"},
            "hpi": {},
            "ph": {},
            "cc": {},
        },
        store=store,
        plan=plan,
    )

    assert "triage" not in record
    assert record["t1_triage"]["manual_note"] == "legacy"
    assert record["t2_hpi"]["chief_complaint"]["value"] == "头痛"
