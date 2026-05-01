from medi.agents.triage.graph.nodes.intake_review_node import review_intake_quality
from medi.agents.triage.intake_facts import FactStore
from medi.agents.triage.intake_protocols import resolve_intake_plan


def _plan(text: str):
    return resolve_intake_plan([{"role": "user", "content": text}])


def _store(*facts: dict) -> FactStore:
    store = FactStore()
    store.merge_items(facts, source_turn=1)
    return store


def test_review_finishes_when_fever_hpi_has_doctor_value() -> None:
    plan = _plan("我发烧三天，最高39度，咽痛，吃了布洛芬")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "三天前"},
        {"slot": "hpi.severity", "status": "present", "value": "最高39度"},
        {"slot": "hpi.timing", "status": "present", "value": "反复发热三天"},
        {"slot": "hpi.associated_symptoms", "status": "present", "value": "咽痛，没有腹泻"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "specific.antipyretics", "status": "present", "value": "布洛芬有效但反复"},
        {"slot": "specific.associated_fever_symptoms", "status": "present", "value": "咽痛，没有腹泻"},
        {"slot": "safety.current_medications", "status": "present", "value": "布洛芬"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
        {"slot": "hpi.relevant_history", "status": "absent", "value": "无相关病史"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=[],
        assistant_count=4,
    )

    assert review.can_finish_intake is True
    assert review.doctor_summary_ready is True
    assert review.safety_slots_covered is True
    assert review.red_flags_checked is True
    assert review.task_tree["status"] == "complete"
    assert review.task_tree["completion"] == 1.0


def test_review_asks_diarrhea_frequency_before_routing() -> None:
    plan = _plan("我拉稀一天了")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "拉稀"},
        {"slot": "hpi.onset", "status": "present", "value": "一天前"},
        {"slot": "hpi.timing", "status": "present", "value": "持续一天"},
        {"slot": "hpi.severity", "status": "present", "value": "有点严重"},
        {"slot": "safety.current_medications", "status": "absent", "value": "没有"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=[],
        assistant_count=2,
    )

    assert review.can_finish_intake is False
    assert review.next_best_slot == "specific.frequency"
    assert "几次" in (review.next_best_question or "")


def test_review_skips_repeated_missing_slot_and_asks_next_high_value_slot() -> None:
    plan = _plan("我拉稀一天了")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "拉稀"},
        {"slot": "hpi.onset", "status": "present", "value": "一天前"},
        {"slot": "hpi.timing", "status": "present", "value": "持续一天"},
        {"slot": "hpi.severity", "status": "present", "value": "有点严重"},
        {"slot": "safety.current_medications", "status": "absent", "value": "没有"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=["specific.frequency", "specific.frequency"],
        assistant_count=3,
    )

    assert review.can_finish_intake is False
    assert review.next_best_slot == "specific.dehydration_signs"
    assert "尿少" in (review.next_best_question or "")


def test_review_asks_allergy_only_when_medication_is_known() -> None:
    plan = _plan("我发烧，吃了布洛芬")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "今天"},
        {"slot": "hpi.severity", "status": "present", "value": "38.5度"},
        {"slot": "hpi.associated_symptoms", "status": "present", "value": "无其他不适"},
        {"slot": "specific.max_temperature", "status": "present", "value": "38.5度"},
        {"slot": "specific.associated_fever_symptoms", "status": "absent", "value": "无"},
        {"slot": "safety.current_medications", "status": "present", "value": "布洛芬"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=[],
        assistant_count=4,
    )

    assert review.can_finish_intake is False
    assert review.next_best_slot == "safety.allergies"
    assert "过敏" in (review.next_best_question or "")
    assert "目前在用什么药" not in (review.next_best_question or "")


def test_review_requires_pediatric_overlay_red_flag_context() -> None:
    plan = _plan("孩子发烧三天，最高39度，咽痛")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "孩子发烧"},
        {"slot": "hpi.onset", "status": "present", "value": "三天前"},
        {"slot": "hpi.severity", "status": "present", "value": "最高39度"},
        {"slot": "hpi.timing", "status": "present", "value": "反复发热三天"},
        {"slot": "hpi.associated_symptoms", "status": "present", "value": "咽痛"},
        {"slot": "specific.max_temperature", "status": "present", "value": "39度"},
        {"slot": "specific.antipyretics", "status": "present", "value": "未用药"},
        {"slot": "specific.associated_fever_symptoms", "status": "present", "value": "咽痛"},
        {"slot": "safety.current_medications", "status": "absent", "value": "没有"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=[],
        assistant_count=4,
    )

    assert review.can_finish_intake is False
    assert review.red_flags_checked is False
    assert review.next_best_slot == "specific.age"
    assert "多大" in (review.next_best_question or "")
    triage_group = next(group for group in review.task_tree["groups"] if group["id"] == "T1")
    assert "specific.age" in triage_group["missing_slots"]


def test_review_relaxes_generic_quality_and_severity_after_five_rounds() -> None:
    plan = _plan("我最近不太舒服")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "不舒服"},
        {"slot": "hpi.onset", "status": "present", "value": "最近"},
        {"slot": "hpi.timing", "status": "present", "value": "持续几天"},
        {"slot": "hpi.location", "status": "present", "value": "上腹部"},
        {"slot": "hpi.associated_symptoms", "status": "absent", "value": "无明显伴随症状"},
        {"slot": "safety.current_medications", "status": "absent", "value": "没有"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=["hpi.character", "hpi.character", "hpi.severity", "hpi.severity"],
        assistant_count=5,
    )

    assert "hpi.character" not in review.high_value_missing_slots
    assert "hpi.severity" not in review.high_value_missing_slots
    assert review.can_finish_intake is True


def test_clinical_missing_slot_is_prioritized_by_review() -> None:
    plan = _plan("我胸痛")
    store = _store(
        {"slot": "hpi.chief_complaint", "status": "present", "value": "胸痛"},
        {"slot": "hpi.onset", "status": "present", "value": "今天"},
        {"slot": "hpi.severity", "status": "present", "value": "6分"},
        {"slot": "hpi.timing", "status": "present", "value": "持续"},
        {"slot": "hpi.location", "status": "present", "value": "胸口"},
        {"slot": "hpi.associated_symptoms", "status": "absent", "value": "无"},
        {"slot": "safety.current_medications", "status": "absent", "value": "没有"},
        {"slot": "safety.allergies", "status": "absent", "value": "无"},
    )

    review = review_intake_quality(
        store=store,
        plan=plan,
        requested_slots=[],
        assistant_count=2,
        clinical_missing_slots=["hpi.radiation"],
    )

    assert review.next_best_slot == "hpi.radiation"
