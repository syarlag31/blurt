"""Tests for task surfacing scorers and composite scoring engine (AC 9, Sub-AC 3).

Tests cover:
- Individual scorer functions (surfacing/scorers.py):
  score_energy, score_mood, score_time_of_day, score_entity_relevance,
  score_calendar_availability, score_behavioral
- Composite TaskScoringEngine (task_surfacing.py) with all signal scorers
- Anti-shame design: no forced surfacing, no guilt, empty results are valid
- Edge cases: no deadline, completed tasks, extreme values
- Integration: full pipeline scoring with realistic mock data
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

# ─── Individual scorer imports (surfacing package) ───────────────────────────
from blurt.services.surfacing.models import (
    BehavioralProfile,
    CalendarSlot,
    ScoredTask,
    SurfacingContext,
    TaskItem,
    TimePreference,
    time_to_preference,
)
from blurt.services.surfacing.scorers import (
    score_behavioral,
    score_calendar_availability,
    score_energy,
    score_entity_relevance,
    score_mood,
    score_time_of_day,
)

# ─── Composite engine imports (task_surfacing.py) ────────────────────────────
from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask as EngineScoredTask,
    SignalScore,
    SignalType,
    SurfaceableTask,
    SurfacingResult,
    SurfacingWeights,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

NOW = datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)


def _ctx(**kwargs) -> SurfacingContext:
    """Helper to build SurfacingContext with defaults."""
    defaults = dict(
        current_time=NOW,
        mood_valence=0.0,
        energy_level=0.5,
    )
    defaults.update(kwargs)
    return SurfacingContext(**defaults)  # type: ignore[arg-type]


def _task(**kwargs) -> TaskItem:
    """Helper to build TaskItem with defaults."""
    defaults = dict(
        content="Test task",
        cognitive_load=0.5,
        created_at=NOW - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return TaskItem(**defaults)  # type: ignore[arg-type]


def _uctx(**kwargs) -> UserContext:
    """Helper to build UserContext for composite engine."""
    defaults = dict(
        energy=EnergyLevel.MEDIUM,
        current_valence=0.0,
        current_arousal=0.5,
        now=NOW,
    )
    defaults.update(kwargs)
    return UserContext(**defaults)  # type: ignore[arg-type]


def _stask(**kwargs) -> SurfaceableTask:
    """Helper to build SurfaceableTask for composite engine."""
    defaults = dict(
        content="Test task",
        status=TaskStatus.ACTIVE,
        estimated_energy=EnergyLevel.MEDIUM,
        created_at=NOW - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return SurfaceableTask(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: score_energy (individual scorer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreEnergy:
    """Tests for the Gaussian energy-cognitive load matcher."""

    def test_perfect_match_high_energy_high_load(self):
        ctx = _ctx(energy_level=0.9)
        task = _task(cognitive_load=0.9)
        s = score_energy(task, ctx)
        assert s > 0.9, f"Perfect high-high match should score >0.9, got {s}"

    def test_perfect_match_low_energy_low_load(self):
        ctx = _ctx(energy_level=0.1)
        task = _task(cognitive_load=0.1)
        s = score_energy(task, ctx)
        assert s > 0.9

    def test_perfect_match_medium(self):
        ctx = _ctx(energy_level=0.5)
        task = _task(cognitive_load=0.5)
        s = score_energy(task, ctx)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_mismatch_low_energy_high_load(self):
        ctx = _ctx(energy_level=0.1)
        task = _task(cognitive_load=0.9)
        s = score_energy(task, ctx)
        assert s < 0.6, f"Low energy + high load should score low, got {s}"

    def test_mismatch_high_energy_low_load(self):
        ctx = _ctx(energy_level=0.9)
        task = _task(cognitive_load=0.1)
        s = score_energy(task, ctx)
        assert s < 0.6

    def test_close_match_better_than_far(self):
        ctx = _ctx(energy_level=0.5)
        close = _task(cognitive_load=0.6)
        far = _task(cognitive_load=0.1)
        assert score_energy(close, ctx) > score_energy(far, ctx)

    def test_score_never_zero(self):
        """Baseline prevents zero scores (anti-shame: nothing is impossible)."""
        ctx = _ctx(energy_level=0.0)
        task = _task(cognitive_load=1.0)
        s = score_energy(task, ctx)
        assert s > 0.0, "Score should never be zero"

    def test_always_in_range(self):
        for energy in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for load in [0.0, 0.25, 0.5, 0.75, 1.0]:
                ctx = _ctx(energy_level=energy)
                task = _task(cognitive_load=load)
                s = score_energy(task, ctx)
                assert 0.0 <= s <= 1.0, f"Out of range: {s}"

    def test_symmetry(self):
        """score_energy(0.3 energy, 0.7 load) ≈ score_energy(0.7 energy, 0.3 load)."""
        s1 = score_energy(_task(cognitive_load=0.7), _ctx(energy_level=0.3))
        s2 = score_energy(_task(cognitive_load=0.3), _ctx(energy_level=0.7))
        assert s1 == pytest.approx(s2, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: score_mood (individual scorer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMood:
    def test_neutral_mood_baseline(self):
        ctx = _ctx(mood_valence=0.0)
        task = _task(cognitive_load=0.5)
        s = score_mood(task, ctx)
        assert 0.4 <= s <= 0.6, f"Neutral mood should give baseline, got {s}"

    def test_positive_mood_boosts_all(self):
        ctx = _ctx(mood_valence=0.8)
        task = _task(cognitive_load=0.5)
        s = score_mood(task, ctx)
        assert s > 0.6, "Positive mood should boost score"

    def test_positive_mood_prefers_challenging(self):
        ctx = _ctx(mood_valence=0.8)
        hard = _task(cognitive_load=0.9)
        easy = _task(cognitive_load=0.1)
        s_hard = score_mood(hard, ctx)
        s_easy = score_mood(easy, ctx)
        assert s_hard > s_easy, "Positive mood should prefer challenging tasks"

    def test_negative_mood_prefers_easy(self):
        ctx = _ctx(mood_valence=-0.8)
        easy = _task(cognitive_load=0.1)
        hard = _task(cognitive_load=0.9)
        s_easy = score_mood(easy, ctx)
        s_hard = score_mood(hard, ctx)
        assert s_easy > s_hard, "Negative mood should prefer easy tasks"

    def test_anti_shame_floor(self):
        """Even worst case should never score below 0.05."""
        ctx = _ctx(mood_valence=-1.0)
        task = _task(cognitive_load=1.0)
        s = score_mood(task, ctx)
        assert s >= 0.05, f"Anti-shame floor violated: {s}"

    def test_always_in_range(self):
        for valence in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for load in [0.0, 0.25, 0.5, 0.75, 1.0]:
                ctx = _ctx(mood_valence=valence)
                task = _task(cognitive_load=load)
                s = score_mood(task, ctx)
                assert 0.0 <= s <= 1.0, f"Out of range: {s}"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: score_time_of_day (individual scorer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreTimeOfDay:
    def test_preferred_time_match_high(self):
        ctx = _ctx(current_time=NOW)
        task = _task(preferred_time=TimePreference.AFTERNOON)
        # NOW is 14:00 UTC → afternoon
        s = score_time_of_day(task, ctx)
        assert s >= 0.8

    def test_preferred_time_mismatch_low(self):
        ctx = _ctx(current_time=NOW)  # 14:00 = afternoon
        task = _task(preferred_time=TimePreference.NIGHT)
        s = score_time_of_day(task, ctx)
        assert s <= 0.4

    def test_no_preferred_time_baseline(self):
        ctx = _ctx(current_time=NOW)
        task = _task(preferred_time=None, due_at=None)
        s = score_time_of_day(task, ctx)
        assert 0.3 <= s <= 0.7, "No time hint should give moderate score"

    def test_due_within_2_hours_boost(self):
        ctx = _ctx(current_time=NOW)
        task = _task(due_at=NOW + timedelta(hours=1))
        s = score_time_of_day(task, ctx)
        assert s >= 0.8

    def test_past_due_gentle_boost(self):
        """Past due should NOT penalize — anti-shame design."""
        ctx = _ctx(current_time=NOW)
        task = _task(due_at=NOW - timedelta(hours=3))
        s = score_time_of_day(task, ctx)
        assert s >= 0.6, f"Past due should get gentle boost, got {s}"

    def test_due_tomorrow_moderate(self):
        ctx = _ctx(current_time=NOW)
        task = _task(due_at=NOW + timedelta(hours=20))
        s = score_time_of_day(task, ctx)
        assert 0.45 <= s <= 0.7

    def test_behavioral_hourly_activity_blends(self):
        profile = BehavioralProfile(hourly_activity={14: 0.9})
        ctx = _ctx(current_time=NOW, behavioral=profile)
        task = _task()
        s = score_time_of_day(task, ctx)
        # Behavioral signal should blend in
        assert s >= 0.5

    def test_always_in_range(self):
        for hours_offset in [-24, -1, 0, 1, 6, 24, 168]:
            ctx = _ctx(current_time=NOW)
            task = _task(due_at=NOW + timedelta(hours=hours_offset))
            s = score_time_of_day(task, ctx)
            assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: score_entity_relevance (individual scorer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreEntityRelevance:
    def test_no_entities_neutral(self):
        ctx = _ctx(active_entity_ids=[])
        task = _task(entity_ids=[])
        s = score_entity_relevance(task, ctx)
        assert 0.3 <= s <= 0.5, "No entity data should give neutral"

    def test_overlap_boosts(self):
        ctx = _ctx(active_entity_ids=["e1", "e2", "e3"])
        matching = _task(entity_ids=["e1", "e2"])
        no_match = _task(entity_ids=["e4", "e5"])
        assert score_entity_relevance(matching, ctx) > score_entity_relevance(no_match, ctx)

    def test_no_overlap_low(self):
        ctx = _ctx(active_entity_ids=["e1"])
        task = _task(entity_ids=["e99"])
        s = score_entity_relevance(task, ctx)
        assert s <= 0.35

    def test_full_overlap_high(self):
        ctx = _ctx(active_entity_ids=["e1", "e2"])
        task = _task(entity_ids=["e1", "e2"])
        s = score_entity_relevance(task, ctx)
        assert s >= 0.7

    def test_more_overlap_higher(self):
        ctx = _ctx(active_entity_ids=["e1", "e2", "e3"])
        one = _task(entity_ids=["e1", "e99"])
        two = _task(entity_ids=["e1", "e2"])
        assert score_entity_relevance(two, ctx) >= score_entity_relevance(one, ctx)

    def test_always_in_range(self):
        for n_active in [0, 1, 3, 10]:
            for n_task in [0, 1, 3, 10]:
                ctx = _ctx(active_entity_ids=[f"a{i}" for i in range(n_active)])
                task = _task(entity_ids=[f"a{i}" for i in range(n_task)])
                s = score_entity_relevance(task, ctx)
                assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: score_calendar_availability (individual scorer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreCalendarAvailability:
    def test_no_calendar_neutral(self):
        ctx = _ctx(calendar_slots=[])
        s = score_calendar_availability(_task(), ctx)
        assert s == 0.5, "No calendar data should be neutral"

    def test_in_meeting_low(self):
        ctx = _ctx(
            current_time=NOW,
            calendar_slots=[
                CalendarSlot(
                    start=NOW - timedelta(minutes=30),
                    end=NOW + timedelta(minutes=30),
                    is_busy=True,
                ),
            ],
        )
        s = score_calendar_availability(_task(), ctx)
        assert s <= 0.2, "In meeting should be low"

    def test_free_block_high(self):
        ctx = _ctx(
            current_time=NOW,
            calendar_slots=[
                CalendarSlot(
                    start=NOW + timedelta(hours=1),
                    end=NOW + timedelta(hours=2),
                    is_busy=True,
                ),
            ],
        )
        # Free for next hour, then busy
        task = _task(estimated_minutes=30)
        s = score_calendar_availability(task, ctx)
        assert s >= 0.5

    def test_no_upcoming_busy_high(self):
        ctx = _ctx(
            current_time=NOW,
            calendar_slots=[
                CalendarSlot(
                    start=NOW - timedelta(hours=2),
                    end=NOW - timedelta(hours=1),
                    is_busy=True,
                ),
            ],
        )
        s = score_calendar_availability(_task(), ctx)
        assert s >= 0.7, "No upcoming busy = assume free"

    def test_always_in_range(self):
        for in_meeting in [True, False]:
            if in_meeting:
                slots = [CalendarSlot(
                    start=NOW - timedelta(minutes=30),
                    end=NOW + timedelta(minutes=30),
                    is_busy=True,
                )]
            else:
                slots = []
            ctx = _ctx(calendar_slots=slots)
            s = score_calendar_availability(_task(), ctx)
            assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: score_behavioral (individual scorer)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreBehavioral:
    def test_no_profile_data_reasonable(self):
        """Empty profile should give a reasonable baseline score."""
        ctx = _ctx(behavioral=BehavioralProfile())
        s = score_behavioral(_task(), ctx)
        # With default profile (avg_daily=3, completed_today=0),
        # the sigmoid load balancer boosts score, so expect > 0.5
        assert 0.3 <= s <= 1.0, f"Empty profile should give reasonable score, got {s}"

    def test_high_completion_time_boosts(self):
        profile = BehavioralProfile(
            completion_by_time={"afternoon": 0.9}
        )
        ctx = _ctx(behavioral=profile, current_time=NOW)  # 14:00 = afternoon
        s = score_behavioral(_task(), ctx)
        assert s > 0.5

    def test_low_completion_time_reduces(self):
        profile = BehavioralProfile(
            completion_by_time={"afternoon": 0.1}
        )
        ctx = _ctx(behavioral=profile, current_time=NOW)
        s = score_behavioral(_task(), ctx)
        assert s < 0.5

    def test_high_completion_by_load(self):
        profile = BehavioralProfile(
            completion_by_load={"medium": 0.9}
        )
        ctx = _ctx(behavioral=profile)
        task = _task(cognitive_load=0.5)  # medium
        s = score_behavioral(task, ctx)
        assert s > 0.5

    def test_daily_load_balancing(self):
        """After completing many tasks, score should decrease (anti-overwhelm)."""
        fresh = BehavioralProfile(avg_daily_completions=3.0, tasks_completed_today=0)
        tired = BehavioralProfile(avg_daily_completions=3.0, tasks_completed_today=8)
        ctx_fresh = _ctx(behavioral=fresh)
        ctx_tired = _ctx(behavioral=tired)
        s_fresh = score_behavioral(_task(), ctx_fresh)
        s_tired = score_behavioral(_task(), ctx_tired)
        assert s_fresh > s_tired, "Should reduce score when over daily average"

    def test_surface_count_fatigue(self):
        """Tasks shown many times without action should score lower."""
        ctx = _ctx(behavioral=BehavioralProfile())
        fresh_task = _task(surface_count=0)
        stale_task = _task(surface_count=10)
        s_fresh = score_behavioral(fresh_task, ctx)
        s_stale = score_behavioral(stale_task, ctx)
        assert s_fresh >= s_stale, "Surfaced-many-times should score lower"

    def test_tag_affinity(self):
        profile = BehavioralProfile(completion_by_tag={"work": 0.9, "personal": 0.2})
        ctx = _ctx(behavioral=profile)
        work_task = _task(tags=["work"])
        personal_task = _task(tags=["personal"])
        s_work = score_behavioral(work_task, ctx)
        s_personal = score_behavioral(personal_task, ctx)
        assert s_work > s_personal

    def test_always_in_range(self):
        for completed in [0, 3, 10, 50]:
            for surface_count in [0, 1, 5, 20]:
                profile = BehavioralProfile(
                    avg_daily_completions=3.0,
                    tasks_completed_today=completed,
                )
                ctx = _ctx(behavioral=profile)
                task = _task(surface_count=surface_count)
                s = score_behavioral(task, ctx)
                assert 0.0 <= s <= 1.0, f"Out of range: {s}"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: SurfacingContext model
# ═══════════════════════════════════════════════════════════════════════════════


class TestSurfacingModels:
    def test_time_to_preference_morning(self):
        from datetime import time
        assert time_to_preference(time(9, 0)) == TimePreference.MORNING

    def test_time_to_preference_afternoon(self):
        from datetime import time
        assert time_to_preference(time(14, 0)) == TimePreference.AFTERNOON

    def test_time_to_preference_evening(self):
        from datetime import time
        assert time_to_preference(time(19, 0)) == TimePreference.EVENING

    def test_time_to_preference_night(self):
        from datetime import time
        assert time_to_preference(time(23, 0)) == TimePreference.NIGHT

    def test_time_to_preference_early_morning(self):
        from datetime import time
        assert time_to_preference(time(6, 0)) == TimePreference.EARLY_MORNING

    def test_scored_task_creation(self):
        task = _task()
        scored = ScoredTask(
            task=task,
            mood_score=0.6,
            energy_score=0.8,
            time_of_day_score=0.5,
            calendar_score=0.5,
            entity_relevance_score=0.4,
            behavioral_score=0.7,
            composite_score=0.6,
        )
        assert scored.composite_score == 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: TaskScoringEngine (composite engine from task_surfacing.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskScoringEngine:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_score_single_returns_scored_task(self):
        result = self.engine.score_single(_stask(), _uctx())
        assert isinstance(result, EngineScoredTask)
        assert 0.0 <= result.composite_score <= 1.0

    def test_score_single_has_all_signals(self):
        result = self.engine.score_single(_stask(), _uctx())
        signals = {s.signal for s in result.signal_scores}
        for st in SignalType:
            assert st in signals, f"Missing signal: {st}"

    def test_signal_breakdown_dict(self):
        result = self.engine.score_single(_stask(), _uctx())
        breakdown = result.signal_breakdown
        assert isinstance(breakdown, dict)
        assert len(breakdown) == len(SignalType)

    def test_non_active_task_excluded(self):
        completed = _stask(status=TaskStatus.COMPLETED)
        deferred = _stask(status=TaskStatus.DEFERRED)
        dropped = _stask(status=TaskStatus.DROPPED)
        result = self.engine.score_and_rank(
            [completed, deferred, dropped], _uctx()
        )
        assert result.total_eligible == 0
        assert not result.has_tasks

    def test_active_tasks_scored(self):
        tasks = [
            _stask(content="task 1"),
            _stask(content="task 2"),
        ]
        result = self.engine.score_and_rank(tasks, _uctx())
        assert result.total_eligible == 2

    def test_empty_tasks_valid(self):
        """No-tasks-pending is a valid state."""
        result = self.engine.score_and_rank([], _uctx())
        assert not result.has_tasks
        assert result.top_task is None
        assert result.tasks == []

    def test_results_sorted_descending(self):
        tasks = [
            _stask(content="low", estimated_energy=EnergyLevel.HIGH),
            _stask(content="high", estimated_energy=EnergyLevel.MEDIUM),
        ]
        ctx = _uctx(energy=EnergyLevel.MEDIUM)
        result = self.engine.score_and_rank(tasks, ctx)
        if len(result.tasks) >= 2:
            for i in range(len(result.tasks) - 1):
                assert result.tasks[i].composite_score >= result.tasks[i + 1].composite_score

    def test_max_results_limit(self):
        tasks = [_stask(content=f"task-{i}") for i in range(20)]
        engine = TaskScoringEngine(max_results=3)
        result = engine.score_and_rank(tasks, _uctx())
        assert len(result.tasks) <= 3

    def test_min_score_filters(self):
        engine = TaskScoringEngine(min_score=0.99)
        tasks = [_stask(content="mediocre")]
        result = engine.score_and_rank(tasks, _uctx())
        assert len(result.tasks) == 0

    def test_custom_weights(self):
        """Custom weights should change scoring behavior."""
        weights = SurfacingWeights(
            time_relevance=0.0,
            energy_match=1.0,
            context_relevance=0.0,
            emotional_alignment=0.0,
            momentum=0.0,
            freshness=0.0,
        )
        engine = TaskScoringEngine(weights=weights)
        ctx = _uctx(energy=EnergyLevel.LOW)
        low_task = _stask(estimated_energy=EnergyLevel.LOW)
        high_task = _stask(estimated_energy=EnergyLevel.HIGH)
        r_low = engine.score_single(low_task, ctx)
        r_high = engine.score_single(high_task, ctx)
        assert r_low.composite_score > r_high.composite_score

    def test_surfacing_reason_not_empty(self):
        result = self.engine.score_single(_stask(), _uctx())
        assert result.surfacing_reason != ""


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: individual signal scorers inside TaskScoringEngine
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineTimeRelevance:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_no_due_date_moderate(self):
        task = _stask(due_at=None)
        result = self.engine.score_single(task, _uctx())
        ts = result.signal_breakdown.get("time_relevance", 0)
        assert 0.3 <= ts <= 0.5

    def test_within_2_hours_high(self):
        task = _stask(due_at=NOW + timedelta(hours=1))
        result = self.engine.score_single(task, _uctx())
        assert result.signal_breakdown["time_relevance"] >= 0.8

    def test_far_future_low(self):
        task = _stask(due_at=NOW + timedelta(days=30))
        result = self.engine.score_single(task, _uctx())
        assert result.signal_breakdown["time_relevance"] <= 0.2

    def test_past_due_gentle_not_punitive(self):
        """Anti-shame: past-due should NOT be penalized — constant gentle score."""
        task = _stask(due_at=NOW - timedelta(hours=5))
        result = self.engine.score_single(task, _uctx())
        assert result.signal_breakdown["time_relevance"] >= 0.5

    def test_urgency_increases_approaching(self):
        far = _stask(due_at=NOW + timedelta(days=5))
        mid = _stask(due_at=NOW + timedelta(hours=12))
        near = _stask(due_at=NOW + timedelta(hours=1))
        r_far = self.engine.score_single(far, _uctx())
        r_mid = self.engine.score_single(mid, _uctx())
        r_near = self.engine.score_single(near, _uctx())
        assert (
            r_near.signal_breakdown["time_relevance"]
            > r_mid.signal_breakdown["time_relevance"]
            > r_far.signal_breakdown["time_relevance"]
        )


class TestEngineEnergyMatch:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_perfect_match(self):
        task = _stask(estimated_energy=EnergyLevel.MEDIUM)
        ctx = _uctx(energy=EnergyLevel.MEDIUM)
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["energy_match"] == 1.0

    def test_one_off(self):
        task = _stask(estimated_energy=EnergyLevel.HIGH)
        ctx = _uctx(energy=EnergyLevel.MEDIUM)
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["energy_match"] == 0.5

    def test_two_off(self):
        task = _stask(estimated_energy=EnergyLevel.HIGH)
        ctx = _uctx(energy=EnergyLevel.LOW)
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["energy_match"] == 0.2


class TestEngineContextRelevance:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_no_context_moderate(self):
        task = _stask()
        ctx = _uctx(active_entity_ids=[], active_entity_names=[])
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["context_relevance"] == 0.3

    def test_entity_overlap_boosts(self):
        task = _stask(entity_names=["Alice", "Bob"])
        ctx = _uctx(active_entity_names=["Alice", "Charlie"])
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["context_relevance"] > 0.3

    def test_project_match(self):
        task = _stask(project="Blurt", entity_names=[])
        ctx = _uctx(active_project="Blurt", active_entity_names=["x"])
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["context_relevance"] >= 0.7


class TestEngineEmotionalAlignment:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_neutral_mood_moderate(self):
        task = _stask()
        ctx = _uctx(current_valence=0.0, current_arousal=0.5)
        result = self.engine.score_single(task, ctx)
        assert 0.4 <= result.signal_breakdown["emotional_alignment"] <= 0.7

    def test_low_mood_avoids_high_intensity(self):
        high_task = _stask(capture_arousal=0.9)
        low_task = _stask(capture_arousal=0.2, capture_valence=0.3)
        ctx = _uctx(current_valence=-0.5)
        r_high = self.engine.score_single(high_task, ctx)
        r_low = self.engine.score_single(low_task, ctx)
        assert (
            r_low.signal_breakdown["emotional_alignment"]
            > r_high.signal_breakdown["emotional_alignment"]
        )


class TestEngineMomentum:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_no_recent_tasks_moderate(self):
        task = _stask()
        ctx = _uctx(recent_task_ids=[])
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["momentum"] == 0.3

    def test_same_project_momentum(self):
        task = _stask(project="Blurt")
        ctx = _uctx(active_project="Blurt", recent_task_ids=["other"])
        result = self.engine.score_single(task, ctx)
        assert result.signal_breakdown["momentum"] >= 0.7


class TestEngineFreshness:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_very_fresh_high(self):
        task = _stask(created_at=NOW - timedelta(minutes=30))
        result = self.engine.score_single(task, _uctx())
        assert result.signal_breakdown["freshness"] >= 0.9

    def test_old_task_low(self):
        task = _stask(created_at=NOW - timedelta(days=14))
        result = self.engine.score_single(task, _uctx())
        assert result.signal_breakdown["freshness"] <= 0.25

    def test_recent_mention_refreshes(self):
        task = _stask(
            created_at=NOW - timedelta(days=14),
            last_mentioned_at=NOW - timedelta(minutes=10),
        )
        result = self.engine.score_single(task, _uctx())
        assert result.signal_breakdown["freshness"] >= 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-shame design tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAntiShameDesign:
    """Verify anti-shame principles across both scoring layers."""

    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_no_overdue_penalty_time_relevance(self):
        """Past-due tasks get gentle constant score, NOT increasing penalty."""
        recent_past = _stask(due_at=NOW - timedelta(hours=2))
        old_past = _stask(due_at=NOW - timedelta(days=7))
        r1 = self.engine.score_single(recent_past, _uctx())
        r2 = self.engine.score_single(old_past, _uctx())
        # Both should be the same constant — not penalized more over time
        assert r1.signal_breakdown["time_relevance"] == r2.signal_breakdown["time_relevance"]

    def test_dropped_tasks_not_surfaced(self):
        dropped = _stask(status=TaskStatus.DROPPED)
        result = self.engine.score_and_rank([dropped], _uctx())
        assert not result.has_tasks

    def test_no_forced_surfacing(self):
        engine = TaskScoringEngine(min_score=0.99)
        result = engine.score_and_rank([_stask()], _uctx())
        assert len(result.tasks) == 0

    def test_empty_is_valid(self):
        result = self.engine.score_and_rank([], _uctx())
        assert result.tasks == []
        assert not result.has_tasks
        assert result.top_task is None

    def test_deferred_not_surfaced(self):
        deferred = _stask(status=TaskStatus.DEFERRED, times_deferred=5)
        result = self.engine.score_and_rank([deferred], _uctx())
        assert not result.has_tasks

    def test_past_due_no_guilt_language(self):
        task = _stask(due_at=NOW - timedelta(hours=3))
        result = self.engine.score_single(task, _uctx())
        time_signal = next(
            s for s in result.signal_scores if s.signal == SignalType.TIME_RELEVANCE
        )
        reason_lower = time_signal.reason.lower()
        assert "overdue" not in reason_lower
        assert "late" not in reason_lower
        assert "missed" not in reason_lower

    def test_individual_mood_anti_shame_floor(self):
        """Mood scorer should never return zero even in worst case."""
        ctx = _ctx(mood_valence=-1.0)
        task = _task(cognitive_load=1.0)
        s = score_mood(task, ctx)
        assert s >= 0.05

    def test_individual_energy_never_zero(self):
        """Energy scorer baseline prevents zero."""
        ctx = _ctx(energy_level=0.0)
        task = _task(cognitive_load=1.0)
        s = score_energy(task, ctx)
        assert s > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests: realistic scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoringIntegration:
    """Integration tests with realistic mock user context and task data."""

    def setup_method(self):
        self.engine = TaskScoringEngine()

    def _morning_high_energy_tasks(self) -> tuple[list[SurfaceableTask], UserContext]:
        ctx = UserContext(
            energy=EnergyLevel.HIGH,
            current_valence=0.6,
            current_arousal=0.7,
            active_entity_names=["scoring", "pipeline"],
            active_project="Blurt",
            recent_task_ids=[],
            now=NOW,
        )
        tasks = [
            SurfaceableTask(
                content="Implement semantic search pipeline",
                estimated_energy=EnergyLevel.HIGH,
                project="Blurt",
                entity_names=["scoring", "pipeline"],
                due_at=NOW + timedelta(days=2),
                created_at=NOW - timedelta(days=1),
            ),
            SurfaceableTask(
                content="Update README typo",
                estimated_energy=EnergyLevel.LOW,
                project="Other",
                created_at=NOW - timedelta(days=5),
            ),
            SurfaceableTask(
                content="Review PR for scoring module",
                estimated_energy=EnergyLevel.MEDIUM,
                project="Blurt",
                entity_names=["scoring"],
                due_at=NOW + timedelta(hours=8),
                created_at=NOW - timedelta(hours=4),
            ),
        ]
        return tasks, ctx

    def test_morning_high_energy_ranks_complex_first(self):
        tasks, ctx = self._morning_high_energy_tasks()
        result = self.engine.score_and_rank(tasks, ctx)
        assert result.has_tasks
        top = result.top_task
        assert top is not None
        assert top.task.estimated_energy == EnergyLevel.HIGH

    def test_morning_complex_beats_trivial(self):
        tasks, ctx = self._morning_high_energy_tasks()
        result = self.engine.score_and_rank(tasks, ctx)
        scores = {t.task.content: t.composite_score for t in result.tasks}
        assert scores.get("Implement semantic search pipeline", 0) > scores.get(
            "Update README typo", 0
        )

    def _evening_low_energy_tasks(self) -> tuple[list[SurfaceableTask], UserContext]:
        ctx = UserContext(
            energy=EnergyLevel.LOW,
            current_valence=-0.4,
            current_arousal=0.2,
            active_entity_names=["dinner"],
            now=NOW,
        )
        tasks = [
            SurfaceableTask(
                content="Architect new microservice",
                estimated_energy=EnergyLevel.HIGH,
                due_at=NOW + timedelta(days=7),
                created_at=NOW - timedelta(days=3),
            ),
            SurfaceableTask(
                content="Reply to Alice's email",
                estimated_energy=EnergyLevel.LOW,
                entity_names=["Alice"],
                created_at=NOW - timedelta(hours=6),
            ),
        ]
        return tasks, ctx

    def test_evening_low_energy_prefers_easy(self):
        tasks, ctx = self._evening_low_energy_tasks()
        result = self.engine.score_and_rank(tasks, ctx)
        if result.has_tasks:
            top = result.top_task
            assert top is not None
            assert top.task.estimated_energy == EnergyLevel.LOW

    def _deadline_scenario(self) -> tuple[list[SurfaceableTask], UserContext]:
        ctx = UserContext(
            energy=EnergyLevel.MEDIUM,
            current_valence=0.0,
            current_arousal=0.5,
            now=NOW,
        )
        tasks = [
            SurfaceableTask(
                content="Submit report",
                estimated_energy=EnergyLevel.MEDIUM,
                due_at=NOW + timedelta(hours=1),
                created_at=NOW - timedelta(days=2),
            ),
            SurfaceableTask(
                content="Prepare presentation",
                estimated_energy=EnergyLevel.HIGH,
                due_at=NOW + timedelta(days=3),
                created_at=NOW - timedelta(days=5),
            ),
            SurfaceableTask(
                content="Explore new framework",
                estimated_energy=EnergyLevel.MEDIUM,
                created_at=NOW - timedelta(days=1),
            ),
        ]
        return tasks, ctx

    def test_imminent_deadline_surfaces_first(self):
        tasks, ctx = self._deadline_scenario()
        result = self.engine.score_and_rank(tasks, ctx)
        assert result.has_tasks
        assert result.top_task is not None
        assert result.top_task.task.content == "Submit report"

    def test_full_pipeline_all_scores_valid(self):
        scenarios = [
            self._morning_high_energy_tasks,
            self._evening_low_energy_tasks,
            self._deadline_scenario,
        ]
        for scenario_fn in scenarios:
            tasks, ctx = scenario_fn()
            for task in tasks:
                scored = self.engine.score_single(task, ctx)
                assert 0.0 <= scored.composite_score <= 1.0
                for signal in scored.signal_scores:
                    assert 0.0 <= signal.value <= 1.0
                    assert signal.signal in SignalType

    def test_mixed_status_filtering(self):
        tasks = [
            _stask(content="active", status=TaskStatus.ACTIVE),
            _stask(content="completed", status=TaskStatus.COMPLETED),
            _stask(content="deferred", status=TaskStatus.DEFERRED),
            _stask(content="dropped", status=TaskStatus.DROPPED),
        ]
        result = self.engine.score_and_rank(tasks, _uctx())
        assert result.total_eligible == 1
        if result.has_tasks:
            assert result.top_task is not None
            assert result.top_task.task.content == "active"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: individual scorers + composite engine cross-validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossValidation:
    """Ensure individual scorer functions and composite engine agree on direction."""

    def test_energy_direction_matches(self):
        """Both layers should prefer matched energy/load."""
        # Individual scorer: high energy + high load = good match
        ctx_ind = _ctx(energy_level=0.9)
        task_ind_match = _task(cognitive_load=0.9)
        task_ind_miss = _task(cognitive_load=0.1)
        s_match = score_energy(task_ind_match, ctx_ind)
        s_miss = score_energy(task_ind_miss, ctx_ind)
        assert s_match > s_miss

        # Engine: same direction
        ctx_eng = _uctx(energy=EnergyLevel.HIGH)
        task_eng_match = _stask(estimated_energy=EnergyLevel.HIGH)
        task_eng_miss = _stask(estimated_energy=EnergyLevel.LOW)
        engine = TaskScoringEngine()
        r_match = engine.score_single(task_eng_match, ctx_eng)
        r_miss = engine.score_single(task_eng_miss, ctx_eng)
        assert (
            r_match.signal_breakdown["energy_match"]
            > r_miss.signal_breakdown["energy_match"]
        )

    def test_mood_direction_matches(self):
        """Both layers should handle negative mood similarly."""
        # Individual: negative mood + high load = lower
        ctx_neg = _ctx(mood_valence=-0.8)
        hard = _task(cognitive_load=0.9)
        easy = _task(cognitive_load=0.1)
        assert score_mood(easy, ctx_neg) > score_mood(hard, ctx_neg)

        # Engine: negative valence + high arousal capture = lower
        ctx_eng = _uctx(current_valence=-0.5)
        engine = TaskScoringEngine()
        high_a = _stask(capture_arousal=0.9)
        low_a = _stask(capture_arousal=0.2, capture_valence=0.3)
        r_high = engine.score_single(high_a, ctx_eng)
        r_low = engine.score_single(low_a, ctx_eng)
        assert (
            r_low.signal_breakdown["emotional_alignment"]
            > r_high.signal_breakdown["emotional_alignment"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Edge case tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def setup_method(self):
        self.engine = TaskScoringEngine()

    def test_zero_weight_engine(self):
        weights = SurfacingWeights(
            time_relevance=0.0,
            energy_match=0.0,
            context_relevance=0.0,
            emotional_alignment=0.0,
            momentum=0.0,
            freshness=0.0,
        )
        engine = TaskScoringEngine(weights=weights)
        result = engine.score_single(_stask(), _uctx())
        assert result.composite_score == 0.0

    def test_extreme_values(self):
        ctx = _uctx(current_valence=-1.0, current_arousal=1.0)
        task = _stask(
            due_at=NOW - timedelta(days=365),
            created_at=NOW - timedelta(days=365),
            times_surfaced=999,
            times_deferred=999,
        )
        result = self.engine.score_single(task, ctx)
        assert 0.0 <= result.composite_score <= 1.0

    def test_future_created_at(self):
        task = _stask(created_at=NOW + timedelta(days=1))
        result = self.engine.score_single(task, _uctx())
        assert 0.0 <= result.composite_score <= 1.0

    def test_deadline_exactly_now(self):
        task = _stask(due_at=NOW)
        result = self.engine.score_single(task, _uctx())
        assert 0.0 <= result.composite_score <= 1.0

    def test_signal_score_validation(self):
        with pytest.raises(ValueError):
            SignalScore(signal=SignalType.FRESHNESS, value=1.5)
        with pytest.raises(ValueError):
            SignalScore(signal=SignalType.FRESHNESS, value=-0.1)

    def test_weights_total(self):
        w = SurfacingWeights()
        assert w.total == pytest.approx(1.0)

    def test_surfacing_result_properties(self):
        result = self.engine.score_and_rank([_stask()], _uctx())
        assert isinstance(result, SurfacingResult)
        assert isinstance(result.has_tasks, bool)
        assert result.total_eligible == 1

    def test_individual_scorer_extreme_energy(self):
        for e in [0.0, 1.0]:
            for load in [0.0, 1.0]:
                ctx = _ctx(energy_level=e)
                task = _task(cognitive_load=load)
                s = score_energy(task, ctx)
                assert 0.0 <= s <= 1.0

    def test_individual_scorer_extreme_mood(self):
        for v in [-1.0, 1.0]:
            for load in [0.0, 1.0]:
                ctx = _ctx(mood_valence=v)
                task = _task(cognitive_load=load)
                s = score_mood(task, ctx)
                assert 0.0 <= s <= 1.0
