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
