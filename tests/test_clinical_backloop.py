from medi.agents.triage.graph.nodes.clinical_node import _missing_for_diagnosis


def test_clinical_missing_for_diagnosis_returns_explicit_slots() -> None:
    state = {
        "intake_plan": {"protocol_id": "chest_pain"},
        "clinical_facts": [
            {"slot": "specific.exertional_related", "status": "present", "value": "活动后加重"},
        ],
        "preconsultation_record": {
            "t2_hpi": {
                "onset": {"value": "今天"},
                "timing": {"value": "持续"},
                "associated_symptoms": {"value": "出汗"},
            },
        },
    }

    missing = _missing_for_diagnosis(state)

    assert "hpi.location" in missing
    assert "hpi.radiation" in missing
    assert "specific.dyspnea_sweating" in missing


def test_clinical_missing_for_diagnosis_empty_when_no_actionable_gap() -> None:
    state = {
        "intake_plan": {"protocol_id": "fever"},
        "preconsultation_record": {
            "t2_hpi": {
                "onset": {"value": "昨晚"},
                "timing": {"value": "反复发热"},
                "associated_symptoms": {"value": "咽痛"},
                "specific": {
                    "max_temperature": {"value": "39度"},
                },
            },
        },
    }

    assert _missing_for_diagnosis(state) == []


def test_clinical_missing_for_diagnosis_does_not_repeat_unknown_answer() -> None:
    state = {
        "intake_plan": {"protocol_id": "chest_pain"},
        "clinical_facts": [
            {"slot": "hpi.onset", "status": "unknown", "value": None},
            {"slot": "hpi.associated_symptoms", "status": "absent", "value": "无"},
            {"slot": "hpi.location", "status": "present", "value": "胸口"},
            {"slot": "hpi.radiation", "status": "unknown", "value": None},
            {"slot": "specific.dyspnea_sweating", "status": "unknown", "value": None},
            {"slot": "specific.exertional_related", "status": "present", "value": "活动后无明显变化"},
        ],
    }
    missing = _missing_for_diagnosis(state)

    assert "hpi.onset" not in missing
    assert "hpi.associated_symptoms" not in missing
    assert "hpi.radiation" not in missing
    assert "specific.dyspnea_sweating" not in missing
