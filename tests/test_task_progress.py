from medi.agents.triage.intake_facts import FactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan
from medi.agents.triage.medical_record import update_medical_record
from medi.agents.triage.task_progress import evaluate_task_progress


def _store(*facts: dict) -> FactStore:
    store = FactStore()
    store.merge_items(facts, source_turn=1)
    return store


def test_evaluate_task_progress_scores_completed_partial_and_pending_tasks() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "孩子发烧，最高39度"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "孩子发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "昨晚"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )
    record = update_medical_record(None, store=store, plan=plan)

    progress, pending_tasks = evaluate_task_progress(medical_record=record)

    assert progress["T2_ONSET"]["score"] == 0.33
    assert progress["T2_ONSET"]["status"] == "partial"
    assert progress["T3_ALLERGY_HISTORY"]["status"] == "complete"
    assert progress["T3_CURRENT_MEDICATIONS"]["status"] == "pending"
    assert progress["T1_PRIMARY_DEPARTMENT"]["status"] == "complete"
    assert "onset_trigger" in progress["T2_ONSET"]["missing_requirements"]
    assert "T2_ONSET" in pending_tasks
    assert "T3_CURRENT_MEDICATIONS" in pending_tasks
    assert "T1_PRIMARY_DEPARTMENT" not in pending_tasks
    assert "T4_CHIEF_COMPLAINT_GENERATION" not in pending_tasks


def test_evaluate_task_progress_tracks_current_medications_and_child_general_condition() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "孩子发烧"}])
    store = _store(
        {"slot": "specific.mental_status", "status": "present", "value": "精神还可以"},
        {"slot": "specific.intake_urination", "status": "present", "value": "喝水少，尿量正常"},
        {"slot": "safety.current_medications", "status": "absent", "value": "未用药"},
    )
    record = update_medical_record(None, store=store, plan=plan)

    progress, pending_tasks = evaluate_task_progress(medical_record=record)

    assert record["hpi"]["general_condition"]["mental_status"]["value"] == "精神还可以"
    assert record["hpi"]["general_condition"]["urination"]["value"] == "喝水少，尿量正常"
    assert record["ph"]["current_medications"]["status"] == "absent"
    assert progress["T2_GENERAL_CONDITION"]["status"] == "partial"
    assert progress["T3_CURRENT_MEDICATIONS"]["status"] == "complete"
    assert "T3_CURRENT_MEDICATIONS" not in pending_tasks


def test_evaluate_task_progress_accepts_triage_values_from_medical_record() -> None:
    plan = resolve_intake_plan([{"role": "user", "content": "头痛"}])
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "头痛"},
    )
    record = update_medical_record(
        {
            "triage": {
                "primary_department": {"department": "神经内科", "confidence": 0.8},
                "secondary_department": "头痛门诊",
            },
            "hpi": {},
            "ph": {},
            "cc": {},
        },
        store=store,
        plan=plan,
    )

    progress, pending_tasks = evaluate_task_progress(medical_record=record)

    assert progress["T1_PRIMARY_DEPARTMENT"]["status"] == "complete"
    assert progress["T1_SECONDARY_DEPARTMENT"]["status"] == "complete"
    assert "T1_PRIMARY_DEPARTMENT" not in pending_tasks
    assert "T1_SECONDARY_DEPARTMENT" not in pending_tasks
