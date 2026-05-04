from datetime import datetime

from medi.memory.health_profile import HealthProfile, VisitRecord
from medi.memory.profile_snapshot import build_profile_snapshot


def test_profile_snapshot_captures_read_only_profile_copy():
    profile = HealthProfile(
        user_id="u1",
        age=68,
        gender="男",
        chronic_conditions=["高血压"],
        allergies=["青霉素"],
        current_medications=["氨氯地平"],
        visit_history=[
            VisitRecord(
                visit_date=datetime(2026, 5, 1, 9, 30),
                department="神经内科",
                chief_complaint="头痛2天",
                conclusion="建议神经内科普通门诊",
            )
        ],
        updated_at="2026-05-01T10:00:00",
    )

    snapshot = build_profile_snapshot(
        profile,
        captured_at=datetime(2026, 5, 3, 12, 0),
    )

    profile.chronic_conditions.append("糖尿病")
    profile.current_medications.append("二甲双胍")

    assert snapshot.user_id == "u1"
    assert snapshot.age == 68
    assert snapshot.gender == "男"
    assert snapshot.chronic_conditions == ("高血压",)
    assert snapshot.current_medications == ("氨氯地平",)
    assert snapshot.source_updated_at == "2026-05-01T10:00:00"
    assert snapshot.captured_at == "2026-05-03T12:00:00"
    assert snapshot.recent_visits[0].department == "神经内科"


def test_profile_snapshot_exports_json_serializable_dict():
    profile = HealthProfile(
        user_id="guest",
        allergies=["  花粉  ", ""],
    )

    snapshot = build_profile_snapshot(profile, captured_at="2026-05-03T12:00:00")

    assert snapshot.to_dict() == {
        "user_id": "guest",
        "age": None,
        "gender": None,
        "chronic_conditions": [],
        "allergies": ["花粉"],
        "current_medications": [],
        "recent_visits": [],
        "source_updated_at": None,
        "captured_at": "2026-05-03T12:00:00",
    }
