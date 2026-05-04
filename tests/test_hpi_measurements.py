from medi.agents.triage.graph.nodes.output_node import DoctorReportBuilder, TriagePatientOutputBuilder, _parse_doctor_report
from medi.agents.triage.clinical_summary import build_symptom_summary_from_record


def test_symptom_summary_formats_temperature_without_pain_score_suffix() -> None:
    summary = build_symptom_summary_from_record(
        {
            "t2_hpi": {
                "onset": {"value": "昨晚"},
                "severity": {"value": "39度"},
                "associated_symptoms": {"value": "咽痛"},
                "specific": {"max_temperature": {"value": "39度"}},
            },
            "t3_background": {
                "current_medications": {"value": "布洛芬"},
            },
        },
        [],
        [],
    )

    assert "最高体温：39度" in summary
    assert "39度/10" not in summary


def test_doctor_report_parser_cleans_temperature_with_nrs_suffix() -> None:
    report = _parse_doctor_report({
        "chief_complaint": "发热",
        "hpi_narrative": "发热伴咽痛",
        "severity_score": "39度/10",
    })

    assert report["severity_score"] == "39度"


def test_patient_builder_uses_explicit_primary_and_alternatives() -> None:
    output = TriagePatientOutputBuilder().build_from_llm({
        "primary_department": {"department": "消化内科", "confidence": 0.82, "reason": "腹泻腹痛优先考虑"},
        "alternative_departments": [
            {"department": "普通内科", "confidence": 0.61, "reason": "也可初步评估"},
        ],
        "patient_advice": "建议先就诊消化内科。",
    })

    assert output["primary_department"]["department"] == "消化内科"
    assert output["alternative_departments"][0]["department"] == "普通内科"


def test_doctor_report_builder_uses_system_patient_metadata() -> None:
    report = DoctorReportBuilder(
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

    assert report["user_id"] == "u1"
    assert report["age"] == 33
    assert report["gender"] == "男"
    assert report["consultation_time"] == "2026-05-02T10:00:00+00:00"
    assert report["triage_timestamp"] == "2026-05-02T10:00:00+00:00"
    assert report["session_id"] == "s1"


def test_doctor_report_builder_fills_report_from_preconsultation_record() -> None:
    record = {
        "t4_chief_complaint": {"generated": "发热伴咽痛1天"},
        "t2_hpi": {
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
        "t3_background": {
            "disease_history": {"value": "哮喘"},
            "allergy_history": {"value": "无"},
            "current_medications": {"value": "沙丁胺醇"},
        },
        "t1_triage": {
            "protocol_id": "fever",
            "protocol_label": "发热",
            "primary_department": {"department": "发热门诊", "reason": "发热伴呼吸道症状"},
        },
    }

    report = DoctorReportBuilder(preconsultation_record=record).build_from_llm({
        "hpi_narrative": "患者发热。",
    })

    assert report["chief_complaint"] == "发热伴咽痛1天"
    assert report["onset"] == "昨晚"
    assert report["location"] == "咽部"
    assert report["severity_score"] == "最高体温39度"
    assert report["associated_symptoms"] == ["咳嗽", "乏力"]
    assert report["diagnostic_history"] == "未做检测"
    assert report["therapeutic_history"] == "服用布洛芬后体温下降"
    assert report["general_condition"]["mental_status"] == "精神尚可"
    assert report["past_history"]["disease_history"] == "哮喘"
    assert report["triage_summary"]["primary_department"] == "发热门诊"
    assert "昨晚" in report["hpi_narrative"]
    assert "服用布洛芬后体温下降" in report["hpi_narrative"]
    assert "t2_hpi" in report["record_coverage"]["used_sections"]


def test_symptom_summary_distinguishes_exposure_from_onset() -> None:
    summary = build_symptom_summary_from_record(
        {
            "t2_hpi": {
                "onset": {"value": "今天耳朵里刺痛"},
                "exposure_event": {"value": "上周去菲律宾潜水"},
                "exposure_symptoms": {"value": "潜水当时或之后无耳痛"},
                "location": {"value": "左耳"},
                "character": {"value": "刺痛"},
            },
        },
        [],
        [],
    )

    assert "起病时间：今天耳朵里刺痛" in summary
    assert "相关暴露：上周去菲律宾潜水" in summary
    assert "暴露后症状：潜水当时或之后无耳痛" in summary
