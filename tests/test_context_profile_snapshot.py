from medi.core.context import UnifiedContext
from medi.memory.profile_snapshot import ProfileSnapshot


def test_unified_context_constraint_prompt_uses_profile_snapshot():
    assert "health_profile" not in UnifiedContext.__dataclass_fields__

    ctx = UnifiedContext(
        user_id="u1",
        session_id="s1",
        profile_snapshot=ProfileSnapshot(
            user_id="u1",
            age=68,
            gender="男",
            chronic_conditions=("高血压",),
            allergies=("青霉素",),
            current_medications=("氨氯地平",),
            captured_at="2026-05-03T12:00:00",
        ),
    )

    prompt = ctx.build_constraint_prompt()

    assert "高血压" in prompt
    assert "青霉素" in prompt
    assert "氨氯地平" in prompt
    assert "68岁" in prompt
