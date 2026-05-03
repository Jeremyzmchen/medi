from medi.agents.triage.graph.nodes.clinical_node import _build_symptom_summary
from medi.agents.triage.graph.nodes.output_node import _parse_doctor_hpi


def test_symptom_summary_formats_temperature_without_pain_score_suffix() -> None:
    summary = _build_symptom_summary({
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


def test_symptom_summary_distinguishes_exposure_from_onset() -> None:
    summary = _build_symptom_summary({
        "onset": "今天耳朵里刺痛",
        "exposure_event": "上周去菲律宾潜水",
        "exposure_symptoms": "潜水当时或之后无耳痛",
        "region": "左耳",
        "quality": "刺痛",
    })

    assert "起病时间：今天耳朵里刺痛" in summary
    assert "相关暴露：上周去菲律宾潜水" in summary
    assert "暴露后症状：潜水当时或之后无耳痛" in summary
