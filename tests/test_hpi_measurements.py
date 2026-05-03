from medi.agents.triage.graph.nodes.output_node import DoctorHpiBuilder, PatientOutputBuilder, _parse_doctor_hpi
from medi.agents.triage.symptom_utils import build_symptom_summary


def test_symptom_summary_formats_temperature_without_pain_score_suffix() -> None:
    summary = build_symptom_summary({
        "onset": "昨晚",
        "max_temperature": "39度",
        "severity": "39度",
        "accompanying": ["咽痛"],
        "medications": ["布洛芬"],
    })

    assert "最高体温：39度" in summary
    assert "39度/10" not in summary


def test_doctor_hpi_parser_cleans_temperature_with_nrs_suffix() -> None:
    hpi = _parse_doctor_hpi({
        "chief_complaint": "发热",
        "hpi_narrative": "发热伴咽痛",
        "severity_score": "39度/10",
    })

    assert hpi["severity_score"] == "39度"


def test_patient_output_builder_splits_primary_and_alternatives_from_legacy_list() -> None:
    output = PatientOutputBuilder().build_from_llm({
        "recommended_departments": [
            {"department": "消化内科", "confidence": 0.82, "reason": "腹泻腹痛优先考虑"},
            {"department": "普通内科", "confidence": 0.61, "reason": "也可初步评估"},
        ],
        "patient_advice": "建议先就诊消化内科。",
    })

    assert output["primary_department"]["department"] == "消化内科"
    assert output["alternative_departments"][0]["department"] == "普通内科"
    assert "recommended_departments" not in output


def test_doctor_hpi_builder_uses_system_patient_metadata() -> None:
    hpi = DoctorHpiBuilder(
        user_id="u1",
        age=33,
        gender="男",
        consultation_time="2026-05-02T10:00:00+00:00",
        session_id="s1",
    ).build_from_llm({
        "user_id": "hallucinated",
        "age": 99,
        "gender": "未知",
        "consultation_time": "bad-time",
        "triage_timestamp": "bad-time",
        "session_id": "bad-session",
        "chief_complaint": "头痛",
    })

    assert hpi["user_id"] == "u1"
    assert hpi["age"] == 33
    assert hpi["gender"] == "男"
    assert hpi["consultation_time"] == "2026-05-02T10:00:00+00:00"
    assert hpi["triage_timestamp"] == "2026-05-02T10:00:00+00:00"
    assert hpi["session_id"] == "s1"


def test_doctor_hpi_builder_fills_report_from_medical_record() -> None:
    record = {
        "cc": {"generated": "发热伴咽痛1天"},
        "hpi": {
            "onset": {"value": "昨晚"},
            "location": {"value": "咽部"},
            "character": {"value": "咽痛"},
            "associated_symptoms": {"value": "咳嗽、乏力"},
            "diagnostic_history": {"value": "未做检测"},
            "therapeutic_history": {"value": "服用布洛芬后体温下降"},
            "general_condition": {
                "mental_status": {"value": "精神尚可"},
                "appetite": {"value": "食欲下降"},
            },
            "specific": {
                "max_temperature": {"value": "39度"},
            },
        },
        "ph": {
            "disease_history": {"value": "哮喘"},
            "allergy_history": {"value": "无"},
            "current_medications": {"value": "沙丁胺醇"},
        },
        "triage": {
            "protocol_id": "fever",
            "protocol_label": "发热",
            "primary_department": {"department": "发热门诊", "reason": "发热伴呼吸道症状"},
        },
    }

    hpi = DoctorHpiBuilder(medical_record=record).build_from_llm({
        "hpi_narrative": "患者发热。",
    })

    assert hpi["chief_complaint"] == "发热伴咽痛1天"
    assert hpi["onset"] == "昨晚"
    assert hpi["location"] == "咽部"
    assert hpi["severity_score"] == "最高体温39度"
    assert hpi["associated_symptoms"] == ["咳嗽", "乏力"]
    assert hpi["diagnostic_history"] == "未做检测"
    assert hpi["therapeutic_history"] == "服用布洛芬后体温下降"
    assert hpi["general_condition"]["mental_status"] == "精神尚可"
    assert hpi["past_history"]["disease_history"] == "哮喘"
    assert hpi["triage_summary"]["primary_department"] == "发热门诊"
    assert "昨晚" in hpi["hpi_narrative"]
    assert "服用布洛芬后体温下降" in hpi["hpi_narrative"]
    assert "hpi" in hpi["record_coverage"]["used_sections"]


def test_symptom_summary_distinguishes_exposure_from_onset() -> None:
    summary = build_symptom_summary({
        "onset": "今天耳朵里刺痛",
        "exposure_event": "上周去菲律宾潜水",
        "exposure_symptoms": "潜水当时或之后无耳痛",
        "region": "左耳",
        "quality": "刺痛",
    })

    assert "起病时间：今天耳朵里刺痛" in summary
    assert "相关暴露：上周去菲律宾潜水" in summary
    assert "暴露后症状：潜水当时或之后无耳痛" in summary
