"""
Pre-consultation task definitions.

This module models the target task decomposition: T1 triage, T2 HPI
collection, T3 past history collection, and T4 chief complaint generation.
Each subtask is defined by semantic requirements over the evolving
T1/T2/T3/T4 PreconsultationRecord, rather than by intake extraction fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskGroupId = Literal["T1", "T2", "T3", "T4"]
CompletionMode = Literal["all", "any"]

TASK_COMPLETION_THRESHOLD = 0.85


@dataclass(frozen=True)
class TaskRequirement:
    id: str
    description: str
    record_paths: tuple[str, ...]
    completion_mode: CompletionMode = "any"


@dataclass(frozen=True)
class PreconsultationTaskSpec:
    id: str
    group_id: TaskGroupId
    label: str
    description: str
    base_priority: int
    requirements: tuple[TaskRequirement, ...]
    critical: bool = False


TASK_GROUP_LABELS: dict[TaskGroupId, str] = {
    "T1": "Triage",
    "T2": "History of Present Illness",
    "T3": "Past History",
    "T4": "Chief Complaint Generation",
}


TASK_SPECS: tuple[PreconsultationTaskSpec, ...] = (
    PreconsultationTaskSpec(
        id="T1_PRIMARY_DEPARTMENT",
        group_id="T1",
        label="Primary Department Identification",
        description="Determine the primary department the patient should visit.",
        base_priority=100,
        requirements=(
            TaskRequirement(
                id="primary_department",
                description="Primary department recommendation is available.",
                record_paths=("t1_triage.primary_department",),
            ),
            TaskRequirement(
                id="presenting_problem",
                description="Patient's presenting problem is documented.",
                record_paths=("t2_hpi.chief_complaint", "t4_chief_complaint.draft", "t4_chief_complaint.generated"),
            ),
        ),
        critical=True,
    ),
    PreconsultationTaskSpec(
        id="T1_SECONDARY_DEPARTMENT",
        group_id="T1",
        label="Secondary Department Identification",
        description="Identify the specific secondary department based on the primary department.",
        base_priority=90,
        requirements=(
            TaskRequirement(
                id="secondary_department",
                description="Specific secondary department recommendation is available.",
                record_paths=("t1_triage.secondary_department",),
            ),
        ),
        critical=True,
    ),
    PreconsultationTaskSpec(
        id="T2_ONSET",
        group_id="T2",
        label="Onset",
        description="Record time, location, mode of onset, prodromal symptoms, and possible causes or triggers.",
        base_priority=85,
        requirements=(
            TaskRequirement(
                id="onset_time",
                description="Time or period when the illness began.",
                record_paths=("t2_hpi.onset", "t2_hpi.timing"),
            ),
            TaskRequirement(
                id="onset_location",
                description="Initial or current symptom location when applicable.",
                record_paths=("t2_hpi.location",),
            ),
            TaskRequirement(
                id="onset_trigger",
                description="Mode of onset, trigger, exposure, or relieving/aggravating context.",
                record_paths=("t2_hpi.exposure_event", "t2_hpi.aggravating_alleviating"),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T2_MAIN_SYMPTOM_CHARACTERISTICS",
        group_id="T2",
        label="Main Symptom Characteristics",
        description="Describe location, nature, duration, severity, and aggravating or relieving factors.",
        base_priority=80,
        requirements=(
            TaskRequirement(
                id="symptom_location",
                description="Main symptom location is documented.",
                record_paths=("t2_hpi.location",),
            ),
            TaskRequirement(
                id="symptom_nature",
                description="Nature or character of the main symptom is documented.",
                record_paths=("t2_hpi.character",),
            ),
            TaskRequirement(
                id="symptom_duration",
                description="Duration or persistence of the main symptom is documented.",
                record_paths=("t2_hpi.duration", "t2_hpi.timing"),
            ),
            TaskRequirement(
                id="symptom_severity",
                description="Severity or quantified intensity is documented.",
                record_paths=("t2_hpi.severity", "t2_hpi.specific.max_temperature", "t2_hpi.specific.frequency"),
            ),
            TaskRequirement(
                id="aggravating_relieving",
                description="Aggravating or relieving factors are documented.",
                record_paths=("t2_hpi.aggravating_alleviating",),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T2_DISEASE_PROGRESSION",
        group_id="T2",
        label="Disease Progression",
        description="Describe the progression and evolution of the illness in chronological order.",
        base_priority=70,
        requirements=(
            TaskRequirement(
                id="chronology",
                description="Symptom chronology or time pattern is documented.",
                record_paths=("t2_hpi.timing", "t2_hpi.onset"),
            ),
            TaskRequirement(
                id="progression",
                description="Progression or evolution of illness is documented.",
                record_paths=("t2_hpi.progression",),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T2_ACCOMPANYING_SYMPTOMS",
        group_id="T2",
        label="Accompanying Symptoms",
        description="Record accompanying symptoms and their relationship with the main symptoms.",
        base_priority=75,
        requirements=(
            TaskRequirement(
                id="associated_symptoms",
                description="Associated symptoms or relevant negatives are documented.",
                record_paths=("t2_hpi.associated_symptoms", "t2_hpi.pertinent_negatives"),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T2_DIAGNOSTIC_THERAPEUTIC_HISTORY",
        group_id="T2",
        label="Diagnostic and Therapeutic History",
        description="Record examinations or treatments after onset and their outcomes if applicable.",
        base_priority=60,
        requirements=(
            TaskRequirement(
                id="diagnostic_or_treatment_history",
                description="Post-onset examinations, treatments, or response are documented if applicable.",
                record_paths=("t2_hpi.diagnostic_history", "t2_hpi.therapeutic_history"),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T2_GENERAL_CONDITION",
        group_id="T2",
        label="General Condition",
        description="Briefly record mental state, sleep, appetite, bowel and bladder functions, and weight after onset.",
        base_priority=72,
        requirements=(
            TaskRequirement(
                id="mental_status",
                description="Mental state or responsiveness is documented.",
                record_paths=(
                    "t2_hpi.general_condition.mental_status",
                    "t2_hpi.specific.mental_status",
                ),
            ),
            TaskRequirement(
                id="intake_urination",
                description="Intake, eating, drinking, or urination status is documented.",
                record_paths=(
                    "t2_hpi.general_condition.urination",
                    "t2_hpi.specific.intake_urination",
                ),
            ),
            TaskRequirement(
                id="daily_function",
                description="Sleep, appetite, bowel function, or weight change is documented when relevant.",
                record_paths=(
                    "t2_hpi.general_condition.sleep",
                    "t2_hpi.general_condition.appetite",
                    "t2_hpi.general_condition.bowel",
                    "t2_hpi.general_condition.weight_change",
                ),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T3_DISEASE_HISTORY",
        group_id="T3",
        label="Disease History",
        description="Record past illnesses, including infectious diseases such as tuberculosis and hepatitis.",
        base_priority=55,
        requirements=(
            TaskRequirement(
                id="disease_history",
                description="Past illnesses or relevant medical history are documented.",
                record_paths=("t3_background.disease_history", "t2_hpi.relevant_history"),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T3_IMMUNIZATION_HISTORY",
        group_id="T3",
        label="Immunization History",
        description="Inquire about the patient's vaccination history.",
        base_priority=20,
        requirements=(
            TaskRequirement(
                id="immunization_history",
                description="Vaccination history is documented.",
                record_paths=("t3_background.immunization_history",),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T3_SURGICAL_TRAUMA_HISTORY",
        group_id="T3",
        label="Surgical and Trauma History",
        description="Record history of surgeries and traumas.",
        base_priority=35,
        requirements=(
            TaskRequirement(
                id="surgical_or_trauma_history",
                description="Surgical or trauma history is documented.",
                record_paths=("t3_background.surgical_history", "t3_background.trauma_history"),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T3_BLOOD_TRANSFUSION_HISTORY",
        group_id="T3",
        label="Blood Transfusion History",
        description="Inquire about blood transfusions and adverse reactions.",
        base_priority=25,
        requirements=(
            TaskRequirement(
                id="blood_transfusion_history",
                description="Blood transfusion history and reactions are documented.",
                record_paths=("t3_background.blood_transfusion_history",),
            ),
        ),
    ),
    PreconsultationTaskSpec(
        id="T3_ALLERGY_HISTORY",
        group_id="T3",
        label="Allergy History",
        description="Inquire about food or drug allergies.",
        base_priority=65,
        requirements=(
            TaskRequirement(
                id="allergy_history",
                description="Food or drug allergy history is documented.",
                record_paths=("t3_background.allergy_history",),
            ),
        ),
        critical=True,
    ),
    PreconsultationTaskSpec(
        id="T3_CURRENT_MEDICATIONS",
        group_id="T3",
        label="Current Medications",
        description="Record current medications or confirm that the patient is not using medication.",
        base_priority=70,
        requirements=(
            TaskRequirement(
                id="current_medications",
                description="Current medication use is documented or explicitly denied.",
                record_paths=("t3_background.current_medications",),
            ),
        ),
        critical=True,
    ),
    PreconsultationTaskSpec(
        id="T4_CHIEF_COMPLAINT_GENERATION",
        group_id="T4",
        label="Chief Complaint Generation",
        description="Generate a concise chief complaint consistent with the full clinical narrative.",
        base_priority=10,
        requirements=(
            TaskRequirement(
                id="generated_chief_complaint",
                description="Concise chief complaint is generated.",
                record_paths=("t4_chief_complaint.generated",),
            ),
        ),
    ),
)


TASK_BY_ID: dict[str, PreconsultationTaskSpec] = {
    spec.id: spec for spec in TASK_SPECS
}


def task_ids() -> list[str]:
    return [spec.id for spec in TASK_SPECS]


def tasks_for_group(group_id: TaskGroupId) -> list[PreconsultationTaskSpec]:
    return [spec for spec in TASK_SPECS if spec.group_id == group_id]


def initial_task_progress() -> dict[str, dict]:
    return {
        spec.id: {
            "task_id": spec.id,
            "group_id": spec.group_id,
            "base_priority": spec.base_priority,
            "score": 0.0,
            "status": "pending",
            "completed_requirements": [],
            "missing_requirements": [req.id for req in spec.requirements],
            "reason": "任务尚未评估",
        }
        for spec in TASK_SPECS
    }
