"""Tests for the composite task scoring and surfacing engine.

Covers:
- Individual signal scoring (all 6 dimensions)
- Composite score computation with configurable weights
- Task eligibility filtering
- Ranking and result ordering
- Anti-shame design principles
- Edge cases and boundary conditions
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask,
    SignalScore,
    SignalType,
    SurfaceableTask,
    SurfacingWeights,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)


def _make_task(**kwargs) -> SurfaceableTask:
    defaults: dict[str, object] = dict(
        content="test task",
        status=TaskStatus.ACTIVE,
        created_at=_now() - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return SurfaceableTask(**defaults)  # type: ignore[arg-type]


def _make_context(**kwargs) -> UserContext:
    defaults: dict[str, object] = dict(now=_now())
    defaults.update(kwargs)
    return UserContext(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SignalScore validation
# ---------------------------------------------------------------------------

class TestSignalScore:
    def test_valid_score(self):
        s = SignalScore(signal=SignalType.FRESHNESS, value=0.5, reason="test")
        assert s.value == 0.5

    def test_score_bounds_low(self):
        s = SignalScore(signal=SignalType.FRESHNESS, value=0.0)
        assert s.value == 0.0

    def test_score_bounds_high(self):
        s = SignalScore(signal=SignalType.FRESHNESS, value=1.0)
        assert s.value == 1.0

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SignalScore(signal=SignalType.FRESHNESS, value=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SignalScore(signal=SignalType.FRESHNESS, value=1.1)


# ---------------------------------------------------------------------------
# SurfacingWeights
# ---------------------------------------------------------------------------

class TestSurfacingWeights:
    def test_default_weights_sum(self):
        w = SurfacingWeights()
        assert abs(w.total - 1.0) < 1e-9

    def test_as_dict_covers_all_signals(self):
        w = SurfacingWeights()
        d = w.as_dict()
        assert set(d.keys()) == set(SignalType)

    def test_custom_weights(self):
        w = SurfacingWeights(time_relevance=0.5, energy_match=0.5,
                             context_relevance=0.0, emotional_alignment=0.0,
                             momentum=0.0, freshness=0.0)
        assert w.total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Task eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_active_task_eligible(self):
        engine = TaskScoringEngine()
        assert engine._is_eligible(_make_task(status=TaskStatus.ACTIVE))

    def test_completed_task_not_eligible(self):
        engine = TaskScoringEngine()
        assert not engine._is_eligible(_make_task(status=TaskStatus.COMPLETED))

    def test_deferred_task_not_eligible(self):
        engine = TaskScoringEngine()
        assert not engine._is_eligible(_make_task(status=TaskStatus.DEFERRED))

    def test_dropped_task_not_eligible(self):
        engine = TaskScoringEngine()
        assert not engine._is_eligible(_make_task(status=TaskStatus.DROPPED))


# ---------------------------------------------------------------------------
# Time relevance scoring
# ---------------------------------------------------------------------------

class TestTimeRelevance:
    def test_no_due_date_moderate(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(due_at=None)
        score = engine._score_time_relevance(task, ctx)
        assert score.value == pytest.approx(0.4)
        assert score.signal == SignalType.TIME_RELEVANCE

    def test_due_within_2_hours_high(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(due_at=_now() + timedelta(hours=1))
        score = engine._score_time_relevance(task, ctx)
        assert score.value == pytest.approx(0.9)

    def test_due_within_24_hours(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(due_at=_now() + timedelta(hours=12))
        score = engine._score_time_relevance(task, ctx)
        assert 0.4 < score.value < 0.9

    def test_due_far_future(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(due_at=_now() + timedelta(days=30))
        score = engine._score_time_relevance(task, ctx)
        assert score.value == pytest.approx(0.15)

    def test_past_due_gentle_not_punitive(self):
        """Anti-shame: past-due tasks get gentle constant, NOT increasing penalty."""
        engine = TaskScoringEngine()
        ctx = _make_context()

        # 1 hour past due
        task_1h = _make_task(due_at=_now() - timedelta(hours=1))
        score_1h = engine._score_time_relevance(task_1h, ctx)

        # 1 week past due — should be SAME score (not increasing guilt)
        task_1w = _make_task(due_at=_now() - timedelta(weeks=1))
        score_1w = engine._score_time_relevance(task_1w, ctx)

        assert score_1h.value == score_1w.value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Energy match scoring
# ---------------------------------------------------------------------------

class TestEnergyMatch:
    def test_perfect_match(self):
        engine = TaskScoringEngine()
        ctx = _make_context(energy=EnergyLevel.HIGH)
        task = _make_task(estimated_energy=EnergyLevel.HIGH)
        score = engine._score_energy_match(task, ctx)
        assert score.value == pytest.approx(1.0)

    def test_one_level_off(self):
        engine = TaskScoringEngine()
        ctx = _make_context(energy=EnergyLevel.HIGH)
        task = _make_task(estimated_energy=EnergyLevel.MEDIUM)
        score = engine._score_energy_match(task, ctx)
        assert score.value == pytest.approx(0.5)

    def test_two_levels_off(self):
        engine = TaskScoringEngine()
        ctx = _make_context(energy=EnergyLevel.LOW)
        task = _make_task(estimated_energy=EnergyLevel.HIGH)
        score = engine._score_energy_match(task, ctx)
        assert score.value == pytest.approx(0.2)

    def test_symmetric(self):
        engine = TaskScoringEngine()
        ctx_high = _make_context(energy=EnergyLevel.HIGH)
        task_low = _make_task(estimated_energy=EnergyLevel.LOW)
        ctx_low = _make_context(energy=EnergyLevel.LOW)
        task_high = _make_task(estimated_energy=EnergyLevel.HIGH)
        s1 = engine._score_energy_match(task_low, ctx_high)
        s2 = engine._score_energy_match(task_high, ctx_low)
        assert s1.value == s2.value


# ---------------------------------------------------------------------------
# Context relevance scoring
# ---------------------------------------------------------------------------

class TestContextRelevance:
    def test_no_active_context_moderate(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(entity_names=["Alice"])
        score = engine._score_context_relevance(task, ctx)
        assert score.value == pytest.approx(0.3)

    def test_entity_overlap(self):
        engine = TaskScoringEngine()
        ctx = _make_context(active_entity_names=["Alice", "Bob"])
        task = _make_task(entity_names=["Alice"])
        score = engine._score_context_relevance(task, ctx)
        assert score.value > 0.3

    def test_project_match(self):
        engine = TaskScoringEngine()
        ctx = _make_context(
            active_entity_names=["x"],
            active_project="Blurt"
        )
        task = _make_task(project="Blurt")
        score = engine._score_context_relevance(task, ctx)
        assert score.value == pytest.approx(0.7)

    def test_project_and_entity_overlap(self):
        engine = TaskScoringEngine()
        ctx = _make_context(
            active_entity_names=["Alice"],
            active_project="Blurt"
        )
        task = _make_task(entity_names=["Alice"], project="Blurt")
        score = engine._score_context_relevance(task, ctx)
        assert score.value >= 0.8 - 1e-9

    def test_no_overlap(self):
        engine = TaskScoringEngine()
        ctx = _make_context(active_entity_names=["Alice"])
        task = _make_task(entity_names=["Bob"])
        score = engine._score_context_relevance(task, ctx)
        assert score.value == pytest.approx(0.1)

    def test_case_insensitive_entity_match(self):
        engine = TaskScoringEngine()
        ctx = _make_context(active_entity_names=["alice"])
        task = _make_task(entity_names=["Alice"])
        score = engine._score_context_relevance(task, ctx)
        assert score.value > 0.3


# ---------------------------------------------------------------------------
# Emotional alignment scoring
# ---------------------------------------------------------------------------

class TestEmotionalAlignment:
    def test_low_mood_avoids_high_intensity(self):
        engine = TaskScoringEngine()
        ctx = _make_context(current_valence=-0.5, current_arousal=0.3)
        task = _make_task(capture_valence=0.0, capture_arousal=0.9)
        score = engine._score_emotional_alignment(task, ctx)
        assert score.value <= 0.3

    def test_low_mood_prefers_positive_task(self):
        engine = TaskScoringEngine()
        ctx = _make_context(current_valence=-0.5)
        task = _make_task(capture_valence=0.3, capture_arousal=0.3)
        score = engine._score_emotional_alignment(task, ctx)
        assert score.value >= 0.6

    def test_positive_mood_all_welcome(self):
        engine = TaskScoringEngine()
        ctx = _make_context(current_valence=0.7)
        task = _make_task(capture_valence=0.0)
        score = engine._score_emotional_alignment(task, ctx)
        assert score.value >= 0.5

    def test_neutral_mood_moderate(self):
        engine = TaskScoringEngine()
        ctx = _make_context(current_valence=0.0)
        task = _make_task()
        score = engine._score_emotional_alignment(task, ctx)
        assert 0.4 <= score.value <= 0.7

    def test_arousal_match_bonus(self):
        engine = TaskScoringEngine()
        ctx = _make_context(current_valence=0.0, current_arousal=0.5)
        task = _make_task(capture_arousal=0.5)
        score = engine._score_emotional_alignment(task, ctx)
        assert "arousal match" in score.reason


# ---------------------------------------------------------------------------
# Momentum scoring
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_no_recent_history(self):
        engine = TaskScoringEngine()
        ctx = _make_context(recent_task_ids=[])
        task = _make_task()
        score = engine._score_momentum(task, ctx)
        assert score.value == pytest.approx(0.3)

    def test_recently_worked_on_moderate(self):
        """Don't spam the same task — moderate score for recent work."""
        engine = TaskScoringEngine()
        task = _make_task(id="task-123")
        ctx = _make_context(recent_task_ids=["task-123"])
        score = engine._score_momentum(task, ctx)
        assert score.value == pytest.approx(0.4)

    def test_project_momentum(self):
        engine = TaskScoringEngine()
        ctx = _make_context(
            recent_task_ids=["other-task"],
            active_project="Blurt",
            active_entity_names=["x"],
        )
        task = _make_task(project="Blurt")
        score = engine._score_momentum(task, ctx)
        assert score.value >= 0.7

    def test_entity_momentum(self):
        engine = TaskScoringEngine()
        ctx = _make_context(
            recent_task_ids=["other"],
            active_entity_names=["Alice"]
        )
        task = _make_task(entity_names=["Alice"])
        score = engine._score_momentum(task, ctx)
        assert score.value >= 0.5


# ---------------------------------------------------------------------------
# Freshness scoring
# ---------------------------------------------------------------------------

class TestFreshness:
    def test_very_fresh(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(created_at=_now() - timedelta(minutes=30))
        score = engine._score_freshness(task, ctx)
        assert score.value == pytest.approx(1.0)

    def test_day_old(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(created_at=_now() - timedelta(hours=24))
        score = engine._score_freshness(task, ctx)
        assert 0.2 < score.value < 0.6

    def test_week_old(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(created_at=_now() - timedelta(days=7))
        score = engine._score_freshness(task, ctx)
        assert score.value < 0.5

    def test_old_task_nonzero(self):
        """Old tasks never disappear completely — just low freshness."""
        engine = TaskScoringEngine()
        ctx = _make_context()
        task = _make_task(created_at=_now() - timedelta(days=60))
        score = engine._score_freshness(task, ctx)
        assert score.value > 0.0

    def test_last_mentioned_resets_freshness(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        old_task = _make_task(
            created_at=_now() - timedelta(days=30),
            last_mentioned_at=_now() - timedelta(minutes=10),
        )
        score = engine._score_freshness(old_task, ctx)
        assert score.value >= 0.9


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

class TestCompositeScoring:
    def test_all_zeros_yields_zero(self):
        engine = TaskScoringEngine()
        signals = [
            SignalScore(signal=s, value=0.0)
            for s in SignalType
        ]
        assert engine._composite_score(signals) == pytest.approx(0.0)

    def test_all_ones_yields_one(self):
        engine = TaskScoringEngine()
        signals = [
            SignalScore(signal=s, value=1.0)
            for s in SignalType
        ]
        assert engine._composite_score(signals) == pytest.approx(1.0)

    def test_weighted_correctly(self):
        """If only time_relevance has value, composite = weight ratio."""
        weights = SurfacingWeights(
            time_relevance=0.5, energy_match=0.0,
            context_relevance=0.0, emotional_alignment=0.0,
            momentum=0.0, freshness=0.0,
        )
        engine = TaskScoringEngine(weights=weights)
        signals = [
            SignalScore(signal=SignalType.TIME_RELEVANCE, value=1.0),
            SignalScore(signal=SignalType.ENERGY_MATCH, value=0.0),
            SignalScore(signal=SignalType.CONTEXT_RELEVANCE, value=0.0),
            SignalScore(signal=SignalType.EMOTIONAL_ALIGNMENT, value=0.0),
            SignalScore(signal=SignalType.MOMENTUM, value=0.0),
            SignalScore(signal=SignalType.FRESHNESS, value=0.0),
        ]
        # Only time_relevance has weight, and it's 1.0
        assert engine._composite_score(signals) == pytest.approx(1.0)

    def test_zero_weights_yields_zero(self):
        weights = SurfacingWeights(
            time_relevance=0.0, energy_match=0.0,
            context_relevance=0.0, emotional_alignment=0.0,
            momentum=0.0, freshness=0.0,
        )
        engine = TaskScoringEngine(weights=weights)
        signals = [
            SignalScore(signal=s, value=1.0)
            for s in SignalType
        ]
        assert engine._composite_score(signals) == pytest.approx(0.0)

    def test_score_clamped_to_01(self):
        engine = TaskScoringEngine()
        signals = [
            SignalScore(signal=s, value=0.8)
            for s in SignalType
        ]
        score = engine._composite_score(signals)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Score and rank end-to-end
# ---------------------------------------------------------------------------

class TestScoreAndRank:
    def test_empty_tasks(self):
        engine = TaskScoringEngine()
        result = engine.score_and_rank([], _make_context())
        assert not result.has_tasks
        assert result.total_eligible == 0

    def test_filters_non_active(self):
        engine = TaskScoringEngine()
        tasks = [
            _make_task(status=TaskStatus.COMPLETED),
            _make_task(status=TaskStatus.DEFERRED),
            _make_task(status=TaskStatus.DROPPED),
        ]
        result = engine.score_and_rank(tasks, _make_context())
        assert result.total_eligible == 0
        assert not result.has_tasks

    def test_ranks_by_composite_score_descending(self):
        engine = TaskScoringEngine(min_score=0.0)
        ctx = _make_context(energy=EnergyLevel.HIGH)

        # Task 1: high energy match, due soon
        task1 = _make_task(
            content="high energy match",
            estimated_energy=EnergyLevel.HIGH,
            due_at=_now() + timedelta(hours=1),
            created_at=_now() - timedelta(minutes=10),
        )
        # Task 2: low energy match, far future
        task2 = _make_task(
            content="low energy match",
            estimated_energy=EnergyLevel.LOW,
            due_at=_now() + timedelta(days=30),
            created_at=_now() - timedelta(days=5),
        )

        result = engine.score_and_rank([task2, task1], ctx)
        assert result.has_tasks
        assert result.tasks[0].task.content == "high energy match"
        assert result.tasks[0].composite_score > result.tasks[1].composite_score

    def test_max_results_limit(self):
        engine = TaskScoringEngine(max_results=2, min_score=0.0)
        tasks = [_make_task(content=f"task-{i}") for i in range(10)]
        result = engine.score_and_rank(tasks, _make_context())
        assert len(result.tasks) <= 2
        assert result.total_filtered > 0

    def test_min_score_filter(self):
        engine = TaskScoringEngine(min_score=0.99)
        tasks = [_make_task()]
        result = engine.score_and_rank(tasks, _make_context())
        # Extremely high threshold — likely no tasks pass
        assert len(result.tasks) == 0

    def test_result_has_context_snapshot(self):
        engine = TaskScoringEngine()
        ctx = _make_context()
        result = engine.score_and_rank([], ctx)
        assert result.context_snapshot is ctx

    def test_top_task_returns_highest(self):
        engine = TaskScoringEngine(min_score=0.0)
        ctx = _make_context()
        tasks = [
            _make_task(content="a"),
            _make_task(content="b", due_at=_now() + timedelta(hours=1)),
        ]
        result = engine.score_and_rank(tasks, ctx)
        if result.has_tasks:
            assert result.top_task is result.tasks[0]


# ---------------------------------------------------------------------------
# Score single
# ---------------------------------------------------------------------------

class TestScoreSingle:
    def test_returns_scored_task(self):
        engine = TaskScoringEngine()
        task = _make_task()
        ctx = _make_context()
        scored = engine.score_single(task, ctx)
        assert isinstance(scored, ScoredTask)
        assert 0.0 <= scored.composite_score <= 1.0
        assert len(scored.signal_scores) == 6

    def test_signal_breakdown(self):
        engine = TaskScoringEngine()
        task = _make_task()
        ctx = _make_context()
        scored = engine.score_single(task, ctx)
        breakdown = scored.signal_breakdown
        assert set(breakdown.keys()) == {s.value for s in SignalType}


# ---------------------------------------------------------------------------
# Anti-shame design
# ---------------------------------------------------------------------------

class TestAntiShame:
    def test_no_increasing_penalty_for_old_tasks(self):
        """Tasks should not score WORSE the longer they go undone."""
        engine = TaskScoringEngine(min_score=0.0)
        ctx = _make_context()

        task_1w = _make_task(
            due_at=_now() - timedelta(weeks=1),
            created_at=_now() - timedelta(weeks=2),
        )
        task_1m = _make_task(
            due_at=_now() - timedelta(weeks=4),
            created_at=_now() - timedelta(weeks=5),
        )

        score_1w = engine.score_single(task_1w, ctx)
        score_1m = engine.score_single(task_1m, ctx)

        # Time relevance should be the same (constant for past due)
        tr_1w = next(s for s in score_1w.signal_scores
                     if s.signal == SignalType.TIME_RELEVANCE)
        tr_1m = next(s for s in score_1m.signal_scores
                     if s.signal == SignalType.TIME_RELEVANCE)
        assert tr_1w.value == tr_1m.value

    def test_deferred_tasks_not_surfaced(self):
        """Deferred tasks should never appear — user chose to defer."""
        engine = TaskScoringEngine(min_score=0.0)
        tasks = [_make_task(status=TaskStatus.DEFERRED)]
        result = engine.score_and_rank(tasks, _make_context())
        assert not result.has_tasks

    def test_dropped_tasks_not_surfaced(self):
        """Dropped tasks are gone — no guilt resurface."""
        engine = TaskScoringEngine(min_score=0.0)
        tasks = [_make_task(status=TaskStatus.DROPPED)]
        result = engine.score_and_rank(tasks, _make_context())
        assert not result.has_tasks

    def test_empty_is_valid(self):
        """No-tasks-pending is a valid state."""
        engine = TaskScoringEngine()
        result = engine.score_and_rank([], _make_context())
        assert not result.has_tasks
        assert result.top_task is None

    def test_no_shame_words_in_reasons(self):
        """Surfacing reasons should never contain guilt/shame language."""
        shame_words = {
            "overdue", "late", "behind", "failing", "missed",
            "neglected", "forgot", "shame", "guilt", "penalty",
        }
        engine = TaskScoringEngine(min_score=0.0)
        ctx = _make_context()

        # Various task scenarios
        tasks = [
            _make_task(due_at=_now() - timedelta(days=7)),  # past due
            _make_task(due_at=_now() + timedelta(hours=1)),  # urgent
            _make_task(),  # normal
        ]

        result = engine.score_and_rank(tasks, ctx)
        for scored in result.tasks:
            reason_lower = scored.surfacing_reason.lower()
            for word in shame_words:
                assert word not in reason_lower, (
                    f"Shame word '{word}' found in reason: {scored.surfacing_reason}"
                )
            for signal in scored.signal_scores:
                signal_reason_lower = signal.reason.lower()
                for word in shame_words:
                    assert word not in signal_reason_lower, (
                        f"Shame word '{word}' found in signal reason: {signal.reason}"
                    )
