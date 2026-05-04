"""
EncounterState — business lifecycle for one medical consultation event.

Session is the technical chat container. EncounterState is the medical event
container that later owns facts, records, assessments, and outputs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from medi.memory.profile_snapshot import ProfileSnapshot


class EncounterIntent(str, Enum):
    TRIAGE = "triage"
    MEDICATION = "medication"
    HEALTH_REPORT = "health_report"
    UNKNOWN = "unknown"


class EncounterStatus(str, Enum):
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class EncounterState:
    encounter_id: str
    session_id: str
    user_id: str
    intent: EncounterIntent
    status: EncounterStatus
    profile_snapshot: ProfileSnapshot | None
    created_at: str
    updated_at: str | None = None


def create_encounter(
    *,
    session_id: str,
    user_id: str,
    intent: EncounterIntent,
    profile_snapshot: ProfileSnapshot | None,
    encounter_id: str | None = None,
    created_at: str | None = None,
) -> EncounterState:
    now = created_at or _utc_now_iso()
    return EncounterState(
        encounter_id=encounter_id or _new_encounter_id(),
        session_id=session_id,
        user_id=user_id,
        intent=intent,
        status=EncounterStatus.ACTIVE,
        profile_snapshot=profile_snapshot,
        created_at=now,
        updated_at=now,
    )


def mark_active(encounter: EncounterState) -> EncounterState:
    return mark_status(encounter, EncounterStatus.ACTIVE)


def mark_waiting(encounter: EncounterState) -> EncounterState:
    return mark_status(encounter, EncounterStatus.WAITING)


def mark_completed(encounter: EncounterState) -> EncounterState:
    return mark_status(encounter, EncounterStatus.COMPLETED)


def mark_cancelled(encounter: EncounterState) -> EncounterState:
    return mark_status(encounter, EncounterStatus.CANCELLED)


def mark_status(
    encounter: EncounterState,
    status: EncounterStatus,
    *,
    updated_at: str | None = None,
) -> EncounterState:
    encounter.status = status
    encounter.updated_at = updated_at or _utc_now_iso()
    return encounter


def _new_encounter_id() -> str:
    return f"enc_{uuid.uuid4().hex[:12]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
