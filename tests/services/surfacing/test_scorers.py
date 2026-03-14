"""Tests for task surfacing scoring functions.

Each scorer is a pure function: (TaskItem, SurfacingContext) → float in [0, 1].
Tests verify normalization, directional correctness, and anti-shame guarantees.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone


from blurt.services.surfacing.models import (
    BehavioralProfile,
    CalendarSlot,
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
    _clamp,
    _adjacent_buckets,
    _load_bucket,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)


def _task(**kwargs) -> TaskItem:
    defaults = dict(content="Test task", cognitive_load=0.5)
    defaults.update(kwargs)
    return TaskItem(**defaults)  # type: ignore[arg-type]


def _ctx(**kwargs) -> SurfacingContext:
    defaults = dict(current_time=_now())
    defaults.update(kwargs)
    return SurfacingContext(**defaults)  # type: ignore[arg-type]


# ── Helper Tests ─────────────────────────────────────────────────────


class TestHelpers:
    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_time_to_preference_early_morning(self):
        assert time_to_preference(time(6, 0)) == TimePreference.EARLY_MORNING

    def test_time_to_preference_morning(self):
        assert time_to_preference(time(10, 0)) == TimePreference.MORNING

    def test_time_to_preference_afternoon(self):
        assert time_to_preference(time(14, 0)) == TimePreference.AFTERNOON

    def test_time_to_preference_evening(self):
        assert time_to_preference(time(19, 0)) == TimePreference.EVENING

    def test_time_to_preference_night(self):
        assert time_to_preference(time(23, 0)) == TimePreference.NIGHT

    def test_time_to_preference_midnight(self):
        assert time_to_preference(time(0, 0)) == TimePreference.NIGHT

    def test_adjacent_buckets_true(self):
        assert _adjacent_buckets(TimePreference.MORNING, TimePreference.AFTERNOON)

    def test_adjacent_buckets_false(self):
        assert not _adjacent_buckets(TimePreference.MORNING, TimePreference.EVENING)

    def test_adjacent_buckets_wrap(self):
        assert _adjacent_buckets(TimePreference.NIGHT, TimePreference.EARLY_MORNING)

    def test_load_bucket(self):
        assert _load_bucket(0.1) == "low"
        assert _load_bucket(0.5) == "medium"
        assert _load_bucket(0.8) == "high"


# ── Mood Score Tests ─────────────────────────────────────────────────


class TestMoodScore:
    """Mood scorer: matches task difficulty to current emotional state."""

    def test_returns_normalized_range(self):
        for valence in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for load in [0.0, 0.5, 1.0]:
                s = score_mood(_task(cognitive_load=load), _ctx(mood_valence=valence))
                assert 0.0 <= s <= 1.0, f"v={valence}, l={load} → {s}"

    def test_positive_mood_prefers_hard_tasks(self):
        ctx = _ctx(mood_valence=0.8)
        hard = score_mood(_task(cognitive_load=0.9), ctx)
        easy = score_mood(_task(cognitive_load=0.1), ctx)
        assert hard > easy

    def test_negative_mood_prefers_easy_tasks(self):
        ctx = _ctx(mood_valence=-0.8)
        hard = score_mood(_task(cognitive_load=0.9), ctx)
        easy = score_mood(_task(cognitive_load=0.1), ctx)
        assert easy > hard

    def test_neutral_mood_is_baseline(self):
        ctx = _ctx(mood_valence=0.0)
        s = score_mood(_task(cognitive_load=0.5), ctx)
        assert 0.4 <= s <= 0.6, f"Neutral mood should be ~0.5, got {s}"

    def test_negative_mood_never_zero(self):
        """Anti-shame: even worst mood never zeroes out a task."""
        ctx = _ctx(mood_valence=-1.0)
        s = score_mood(_task(cognitive_load=1.0), ctx)
        assert s > 0.0


# ── Energy Score Tests ───────────────────────────────────────────────


class TestEnergyScore:
    """Energy scorer: matches task cognitive load to energy level."""

    def test_returns_normalized_range(self):
        for energy in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for load in [0.0, 0.25, 0.5, 0.75, 1.0]:
                s = score_energy(_task(cognitive_load=load), _ctx(energy_level=energy))
                assert 0.0 <= s <= 1.0

    def test_perfect_match_scores_high(self):
        """When energy matches cognitive load, score should be near 1.0."""
        s = score_energy(_task(cognitive_load=0.7), _ctx(energy_level=0.7))
        assert s > 0.9

    def test_mismatch_scores_lower(self):
        """Big mismatch between energy and load → lower score."""
        matched = score_energy(_task(cognitive_load=0.5), _ctx(energy_level=0.5))
        mismatched = score_energy(_task(cognitive_load=0.0), _ctx(energy_level=1.0))
        assert matched > mismatched

    def test_minimum_baseline(self):
        """Score never drops below baseline even with max mismatch."""
        s = score_energy(_task(cognitive_load=0.0), _ctx(energy_level=1.0))
        assert s >= 0.1


# ── Time-of-Day Score Tests ──────────────────────────────────────────


class TestTimeOfDayScore:
    """Time-of-day scorer: preferred time match + due date proximity."""

    def test_returns_normalized_range(self):
        for pref in list(TimePreference):
            s = score_time_of_day(
                _task(preferred_time=pref),
                _ctx(current_time=_now()),
            )
            assert 0.0 <= s <= 1.0

    def test_matching_preference_scores_high(self):
        # 14:00 UTC → afternoon
        ctx = _ctx(current_time=_now(), timezone_offset_hours=0)
        task = _task(preferred_time=TimePreference.AFTERNOON)
        s = score_time_of_day(task, ctx)
        assert s >= 0.85

    def test_mismatched_preference_scores_low(self):
        ctx = _ctx(current_time=_now(), timezone_offset_hours=0)
        task = _task(preferred_time=TimePreference.NIGHT)
        s = score_time_of_day(task, ctx)
        assert s <= 0.4

    def test_adjacent_preference_scores_mid(self):
        # 14:00 → afternoon; morning is adjacent
        ctx = _ctx(current_time=_now(), timezone_offset_hours=0)
        task = _task(preferred_time=TimePreference.MORNING)
        s = score_time_of_day(task, ctx)
        assert 0.5 <= s <= 0.8

    def test_due_soon_boosts_score(self):
        due = _now() + timedelta(hours=1)
        task = _task(due_at=due)
        s = score_time_of_day(task, _ctx())
        assert s >= 0.8

    def test_due_far_away_neutral(self):
        due = _now() + timedelta(days=7)
        task = _task(due_at=due)
        s = score_time_of_day(task, _ctx())
        assert s <= 0.6

    def test_overdue_still_positive(self):
        """Anti-shame: overdue tasks get boosted, not penalized."""
        due = _now() - timedelta(hours=2)
        task = _task(due_at=due)
        s = score_time_of_day(task, _ctx())
        assert s >= 0.6

    def test_behavioral_hourly_activity_blends(self):
        # 14:00 UTC → hour 14
        profile = BehavioralProfile(hourly_activity={14: 0.9})
        ctx = _ctx(behavioral=profile, timezone_offset_hours=0)
        task = _task()
        s = score_time_of_day(task, ctx)
        assert s >= 0.5  # blended upward

    def test_no_preference_no_due_neutral(self):
        task = _task(preferred_time=None, due_at=None)
        ctx = _ctx(behavioral=BehavioralProfile())
        s = score_time_of_day(task, ctx)
        assert 0.3 <= s <= 0.7


# ── Calendar Availability Score Tests ────────────────────────────────


class TestCalendarAvailabilityScore:
    """Calendar scorer: free time matching for task surfacing."""

    def test_no_calendar_data_neutral(self):
        s = score_calendar_availability(_task(), _ctx(calendar_slots=[]))
        assert s == 0.5

    def test_in_meeting_scores_low(self):
        now = _now()
        slot = CalendarSlot(
            start=now - timedelta(minutes=15),
            end=now + timedelta(minutes=45),
            is_busy=True,
        )
        s = score_calendar_availability(_task(), _ctx(calendar_slots=[slot]))
        assert s <= 0.2

    def test_free_with_enough_time_scores_high(self):
        now = _now()
        busy_later = CalendarSlot(
            start=now + timedelta(hours=2),
            end=now + timedelta(hours=3),
            is_busy=True,
        )
        task = _task(estimated_minutes=30)
        s = score_calendar_availability(task, _ctx(calendar_slots=[busy_later]))
        assert s >= 0.85

    def test_tight_window_scores_medium(self):
        now = _now()
        busy_soon = CalendarSlot(
            start=now + timedelta(minutes=20),
            end=now + timedelta(hours=1),
            is_busy=True,
        )
        task = _task(estimated_minutes=30)
        s = score_calendar_availability(task, _ctx(calendar_slots=[busy_soon]))
        assert 0.5 <= s <= 0.7

    def test_no_upcoming_busy_assumes_free(self):
        now = _now()
        past_slot = CalendarSlot(
            start=now - timedelta(hours=2),
            end=now - timedelta(hours=1),
            is_busy=True,
        )
        s = score_calendar_availability(_task(), _ctx(calendar_slots=[past_slot]))
        assert s >= 0.7

    def test_returns_normalized(self):
        for mins in [5, 15, 30, 60, 120]:
            task = _task(estimated_minutes=mins)
            s = score_calendar_availability(task, _ctx())
            assert 0.0 <= s <= 1.0


# ── Entity Relevance Score Tests ─────────────────────────────────────


class TestEntityRelevanceScore:
    """Entity relevance: contextual match between task and active entities."""

    def test_no_entities_returns_below_neutral(self):
        s = score_entity_relevance(_task(entity_ids=[]), _ctx(active_entity_ids=[]))
        assert s == 0.4

    def test_no_overlap_low_score(self):
        task = _task(entity_ids=["a", "b"])
        ctx = _ctx(active_entity_ids=["c", "d"])
        s = score_entity_relevance(task, ctx)
        assert s <= 0.4

    def test_full_overlap_high_score(self):
        task = _task(entity_ids=["a", "b"])
        ctx = _ctx(active_entity_ids=["a", "b"])
        s = score_entity_relevance(task, ctx)
        assert s >= 0.7

    def test_partial_overlap_mid_score(self):
        task = _task(entity_ids=["a", "b", "c"])
        ctx = _ctx(active_entity_ids=["a"])
        s = score_entity_relevance(task, ctx)
        full = score_entity_relevance(
            _task(entity_ids=["a", "b", "c"]),
            _ctx(active_entity_ids=["a", "b", "c"]),
        )
        assert s < full

    def test_diminishing_returns(self):
        """More overlaps help, but with diminishing returns."""
        task2 = _task(entity_ids=["a", "b"])
        task5 = _task(entity_ids=["a", "b", "c", "d", "e"])
        ctx = _ctx(active_entity_ids=["a", "b", "c", "d", "e"])
        s2 = score_entity_relevance(task2, ctx)
        s5 = score_entity_relevance(task5, ctx)
        # Both should be high, but not 5x different
        assert s5 >= s2
        assert s5 / max(s2, 0.01) < 2.0

    def test_returns_normalized(self):
        for n in range(6):
            ids = [str(i) for i in range(n)]
            s = score_entity_relevance(
                _task(entity_ids=ids), _ctx(active_entity_ids=ids)
            )
            assert 0.0 <= s <= 1.0


# ── Behavioral Score Tests ───────────────────────────────────────────


class TestBehavioralScore:
    """Behavioral scorer: learned patterns + anti-overwhelm signals."""

    def test_no_behavioral_data_neutral(self):
        s = score_behavioral(_task(), _ctx(behavioral=BehavioralProfile()))
        # Should still work with defaults
        assert 0.0 <= s <= 1.0

    def test_high_completion_rate_boosts(self):
        profile = BehavioralProfile(
            completion_by_time={"afternoon": 0.9},
            completion_by_load={"medium": 0.85},
        )
        ctx = _ctx(behavioral=profile, timezone_offset_hours=0)
        task = _task(cognitive_load=0.5)
        s = score_behavioral(task, ctx)
        assert s >= 0.6

    def test_low_completion_rate_reduces(self):
        profile = BehavioralProfile(
            completion_by_time={"afternoon": 0.1},
            completion_by_load={"medium": 0.1},
        )
        ctx = _ctx(behavioral=profile, timezone_offset_hours=0)
        task = _task(cognitive_load=0.5)
        s = score_behavioral(task, ctx)
        assert s <= 0.5

    def test_tag_affinity(self):
        profile = BehavioralProfile(
            completion_by_tag={"work": 0.9, "personal": 0.3}
        )
        work_task = _task(tags=["work"])
        personal_task = _task(tags=["personal"])
        ctx = _ctx(behavioral=profile)
        assert score_behavioral(work_task, ctx) > score_behavioral(personal_task, ctx)

    def test_daily_load_balancing(self):
        """If already done many tasks today, score decreases (anti-overwhelm)."""
        profile_light = BehavioralProfile(
            avg_daily_completions=5, tasks_completed_today=1
        )
        profile_heavy = BehavioralProfile(
            avg_daily_completions=5, tasks_completed_today=8
        )
        task = _task()
        s_light = score_behavioral(task, _ctx(behavioral=profile_light))
        s_heavy = score_behavioral(task, _ctx(behavioral=profile_heavy))
        assert s_light > s_heavy

    def test_surface_count_fatigue(self):
        """Tasks shown many times without action get gently deprioritized."""
        fresh = _task(surface_count=0)
        stale = _task(surface_count=10)
        ctx = _ctx()
        s_fresh = score_behavioral(fresh, ctx)
        s_stale = score_behavioral(stale, ctx)
        assert s_fresh > s_stale

    def test_surface_count_never_zero(self):
        """Anti-shame: even heavily surfaced tasks never zero out."""
        task = _task(surface_count=100)
        s = score_behavioral(task, _ctx())
        assert s > 0.0

    def test_returns_normalized(self):
        profile = BehavioralProfile(
            completion_by_time={"afternoon": 0.5},
            completion_by_load={"low": 0.5, "medium": 0.5, "high": 0.5},
            completion_by_tag={"test": 0.5},
            avg_daily_completions=5,
            tasks_completed_today=3,
        )
        for count in [0, 1, 5, 20]:
            task = _task(surface_count=count, tags=["test"], cognitive_load=0.5)
            s = score_behavioral(task, _ctx(behavioral=profile))
            assert 0.0 <= s <= 1.0


# ── Anti-Shame Design Tests ─────────────────────────────────────────


class TestAntiShameGuarantees:
    """Cross-cutting: no scorer should ever produce guilt-inducing behavior."""

    ALL_SCORERS = [
        score_mood,
        score_energy,
        score_time_of_day,
        score_calendar_availability,
        score_entity_relevance,
        score_behavioral,
    ]

    def test_all_scores_positive(self):
        """No scorer returns negative values."""
        task = _task()
        ctx = _ctx()
        for scorer in self.ALL_SCORERS:
            s = scorer(task, ctx)
            assert s >= 0.0, f"{scorer.__name__} returned {s}"

    def test_all_scores_at_most_one(self):
        """No scorer exceeds 1.0."""
        task = _task()
        ctx = _ctx()
        for scorer in self.ALL_SCORERS:
            s = scorer(task, ctx)
            assert s <= 1.0, f"{scorer.__name__} returned {s}"

    def test_worst_case_never_zero(self):
        """Even in the worst user state, scores stay above zero."""
        task = _task(
            cognitive_load=1.0,
            surface_count=50,
            entity_ids=["x"],
        )
        ctx = _ctx(
            mood_valence=-1.0,
            energy_level=0.0,
            active_entity_ids=["y"],
            behavioral=BehavioralProfile(
                completion_by_time={"afternoon": 0.0},
                completion_by_load={"high": 0.0},
                avg_daily_completions=3,
                tasks_completed_today=20,
            ),
        )
        for scorer in self.ALL_SCORERS:
            s = scorer(task, ctx)
            assert s > 0.0, f"{scorer.__name__} returned 0 in worst case"

    def test_completed_today_many_still_surfaces(self):
        """Even after many completions, behavioral score stays positive."""
        profile = BehavioralProfile(
            avg_daily_completions=3,
            tasks_completed_today=100,
        )
        s = score_behavioral(_task(), _ctx(behavioral=profile))
        assert s > 0.0
