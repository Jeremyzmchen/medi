from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.task_tree import build_intake_task_tree, slot_task_priority


def _plan(text: str):
    return resolve_intake_plan([{"role": "user", "content": text}])


def _store(*facts: dict) -> ClinicalFactStore:
    store = ClinicalFactStore()
    store.merge_items(facts, source_turn=1)
    return store


def _group(tree: dict, group_id: str) -> dict:
    return next(group for group in tree["groups"] if group["id"] == group_id)


def test_task_tree_groups_flat_slots_into_preconsultation_tasks() -> None:
    plan = _plan("我发烧三天，最高39度，咽痛，吃了布洛芬")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "三天前"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "specific.associated_fever_symptoms", "status": "present", "value": "咽痛"},
        {"slot": "safety.current_medications", "status": "present", "value": "布洛芬"},
    )

    tree = build_intake_task_tree(store, plan)

    assert tree["protocol_id"] == "fever"
    assert [group["id"] for group in tree["groups"]] == ["T1", "T2", "T3", "T4"]
    assert "safety.allergies" in _group(tree, "T3")["missing_slots"]
    assert "hpi.chief_complaint" not in tree["pending_slots"]


def test_pediatric_overlay_risk_fields_land_in_triage_group() -> None:
    plan = _plan("孩子发烧三天，最高39度")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "孩子发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "三天前"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
    )

    tree = build_intake_task_tree(store, plan)
    triage_group = _group(tree, "T1")

    assert "specific.age" in triage_group["missing_slots"]
    assert "specific.mental_status" in triage_group["missing_slots"]
    assert slot_task_priority("specific.age", plan) == 10

