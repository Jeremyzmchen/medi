"""
ProfileSnapshot — per-encounter read-only copy of HealthProfile data.

HealthProfile is the long-lived database model. ProfileSnapshot captures what
was known when an encounter started, so triage decisions remain reproducible
even if the long-term profile changes later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from medi.memory.health_profile import HealthProfile, VisitRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VisitSnapshot:
    visit_date: str
    department: str
    chief_complaint: str
    conclusion: str

    @classmethod
    def from_visit_record(cls, record: VisitRecord) -> "VisitSnapshot":
        return cls(
            visit_date=_to_iso(record.visit_date),
            department=record.department,
            chief_complaint=record.chief_complaint,
            conclusion=record.conclusion,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "visit_date": self.visit_date,
            "department": self.department,
            "chief_complaint": self.chief_complaint,
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True)
class ProfileSnapshot:
    user_id: str
    age: int | None = None
    gender: str | None = None
    chronic_conditions: tuple[str, ...] = field(default_factory=tuple)
    allergies: tuple[str, ...] = field(default_factory=tuple)
    current_medications: tuple[str, ...] = field(default_factory=tuple)
    recent_visits: tuple[VisitSnapshot, ...] = field(default_factory=tuple)
    source_updated_at: str | None = None
    captured_at: str = field(default_factory=_utc_now_iso)

    def is_complete(self) -> bool:
        return self.age is not None and self.gender is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "gender": self.gender,
            "chronic_conditions": list(self.chronic_conditions),
            "allergies": list(self.allergies),
            "current_medications": list(self.current_medications),
            "recent_visits": [visit.to_dict() for visit in self.recent_visits],
            "source_updated_at": self.source_updated_at,
            "captured_at": self.captured_at,
        }


def build_profile_snapshot(
    profile: HealthProfile,
    *,
    captured_at: datetime | str | None = None,
    recent_visit_limit: int = 10,
) -> ProfileSnapshot:
    """Capture a read-only snapshot from the long-lived HealthProfile."""
    return ProfileSnapshot(
        user_id=profile.user_id,
        age=profile.age,
        gender=profile.gender,
        chronic_conditions=_clean_tuple(profile.chronic_conditions),
        allergies=_clean_tuple(profile.allergies),
        current_medications=_clean_tuple(profile.current_medications),
        recent_visits=tuple(
            VisitSnapshot.from_visit_record(record)
            for record in profile.visit_history[:recent_visit_limit]
        ),
        source_updated_at=profile.updated_at,
        captured_at=_to_iso(captured_at) if captured_at is not None else _utc_now_iso(),
    )


def _clean_tuple(items: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in items or () if str(item).strip())


def _to_iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
