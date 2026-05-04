from medi.api.session_store import (
    Session,
    active_encounter,
    ensure_active_encounter,
    mark_active_encounter_completed,
    mark_active_encounter_waiting,
)
from medi.core.context import UnifiedContext
from medi.core.encounter import (
    EncounterIntent,
    EncounterStatus,
    create_encounter,
    mark_completed,
    mark_waiting,
)
from medi.memory.profile_snapshot import ProfileSnapshot


def test_encounter_state_lifecycle():
    snapshot = ProfileSnapshot(
        user_id="u1",
        age=68,
        captured_at="2026-05-03T12:00:00",
    )

    encounter = create_encounter(
        encounter_id="enc_1",
        session_id="s1",
        user_id="u1",
        intent=EncounterIntent.TRIAGE,
        profile_snapshot=snapshot,
        created_at="2026-05-03T12:00:00",
    )

    assert encounter.encounter_id == "enc_1"
    assert encounter.session_id == "s1"
    assert encounter.user_id == "u1"
    assert encounter.intent is EncounterIntent.TRIAGE
    assert encounter.status is EncounterStatus.ACTIVE
    assert encounter.profile_snapshot is snapshot

    mark_waiting(encounter)
    assert encounter.status is EncounterStatus.WAITING

    mark_completed(encounter)
    assert encounter.status is EncounterStatus.COMPLETED


def test_session_tracks_active_encounter_by_id_only():
    snapshot = ProfileSnapshot(user_id="u1", captured_at="2026-05-03T12:00:00")
    ctx = UnifiedContext(user_id="u1", session_id="s1", profile_snapshot=snapshot)
    session = Session(
        session_id="s1",
        ctx=ctx,
        orchestrator=object(),
        agent=object(),
        medication_agent=object(),
        health_report_agent=object(),
        obs=object(),
    )

    encounter = ensure_active_encounter(session, EncounterIntent.TRIAGE)

    assert ctx.active_encounter_id == encounter.encounter_id
    assert active_encounter(session) is encounter
    assert encounter.profile_snapshot is snapshot
    assert "active_encounter_id" in UnifiedContext.__dataclass_fields__
    assert "encounters" not in UnifiedContext.__dataclass_fields__

    reused = ensure_active_encounter(session, EncounterIntent.TRIAGE)
    assert reused is encounter

    mark_active_encounter_waiting(session)
    assert encounter.status is EncounterStatus.WAITING

    mark_active_encounter_completed(session)
    assert encounter.status is EncounterStatus.COMPLETED

    next_encounter = ensure_active_encounter(session, EncounterIntent.TRIAGE)
    assert next_encounter is not encounter
    assert ctx.active_encounter_id == next_encounter.encounter_id
