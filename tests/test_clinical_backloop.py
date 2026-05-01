from medi.agents.triage.graph.nodes.clinical_node import _missing_for_diagnosis


def test_clinical_missing_for_diagnosis_returns_explicit_slots() -> None:
    state = {
        "intake_protocol_id": "chest_pain",
        "collection_status": {
            "pattern_specific": {
                "dyspnea_sweating": "missing",
                "exertional_related": "complete",
            }
        },
    }
    symptom_data = {
        "onset": "今天",
        "time_pattern": "持续",
        "accompanying": ["出汗"],
    }

    missing = _missing_for_diagnosis(state, symptom_data)

    assert "hpi.location" in missing
    assert "hpi.radiation" in missing
    assert "specific.dyspnea_sweating" in missing


def test_clinical_missing_for_diagnosis_empty_when_no_actionable_gap() -> None:
    state = {
        "intake_protocol_id": "fever",
        "collection_status": {
            "pattern_specific": {
                "max_temperature": "complete",
                "associated_fever_symptoms": "complete",
            }
        },
    }
    symptom_data = {
        "onset": "昨晚",
        "time_pattern": "反复发热",
        "max_temperature": "39度",
        "accompanying": ["咽痛"],
    }

    assert _missing_for_diagnosis(state, symptom_data) == []


def test_clinical_missing_for_diagnosis_does_not_repeat_unknown_answer() -> None:
    state = {
        "intake_protocol_id": "chest_pain",
        "intake_facts": [
            {"slot": "hpi.onset", "status": "unknown", "value": None},
            {"slot": "hpi.associated_symptoms", "status": "absent", "value": "无"},
            {"slot": "hpi.location", "status": "present", "value": "胸口"},
            {"slot": "hpi.radiation", "status": "unknown", "value": None},
            {"slot": "specific.dyspnea_sweating", "status": "unknown", "value": None},
        ],
        "collection_status": {
            "pattern_specific": {
                "dyspnea_sweating": "missing",
                "exertional_related": "complete",
            }
        },
    }
    symptom_data = {}

    missing = _missing_for_diagnosis(state, symptom_data)

    assert "hpi.onset" not in missing
    assert "hpi.associated_symptoms" not in missing
    assert "hpi.radiation" not in missing
    assert "specific.dyspnea_sweating" not in missing
