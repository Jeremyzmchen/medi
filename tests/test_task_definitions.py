from medi.agents.triage.task_definitions import (
    TASK_BY_ID,
    TASK_COMPLETION_THRESHOLD,
    initial_task_progress,
    task_ids,
    tasks_for_group,
)


def test_task_definitions_cover_all_preconsultation_groups() -> None:
    assert TASK_COMPLETION_THRESHOLD == 0.85
    assert len(task_ids()) == 15
    assert {task.group_id for task in TASK_BY_ID.values()} == {"T1", "T2", "T3", "T4"}
    assert [task.id for task in tasks_for_group("T1")] == [
        "T1_PRIMARY_DEPARTMENT",
        "T1_SECONDARY_DEPARTMENT",
    ]
    assert TASK_BY_ID["T1_PRIMARY_DEPARTMENT"].base_priority == 100
    assert len(tasks_for_group("T2")) == 6
    assert len(tasks_for_group("T3")) == 6
    assert len(tasks_for_group("T4")) == 1


def test_initial_task_progress_is_json_ready_and_pending() -> None:
    progress = initial_task_progress()

    assert list(progress.keys()) == task_ids()
    assert progress["T3_ALLERGY_HISTORY"]["score"] == 0.0
    assert progress["T3_ALLERGY_HISTORY"]["base_priority"] == 65
    assert progress["T3_ALLERGY_HISTORY"]["status"] == "pending"
    assert progress["T3_ALLERGY_HISTORY"]["missing_requirements"] == ["allergy_history"]
    assert progress["T3_CURRENT_MEDICATIONS"]["missing_requirements"] == ["current_medications"]
    assert progress["T4_CHIEF_COMPLAINT_GENERATION"]["missing_requirements"] == [
        "generated_chief_complaint"
    ]
