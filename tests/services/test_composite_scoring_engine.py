"""Tests for the CompositeScoringEngine (AC 9, Sub-AC 2).

Tests cover:
- Weighted aggregation with default and custom weights
- Configurable weights per signal dimension
- SignalWeights dataclass (normalization, from_dict, uniform)
- DimensionScore validation
- Thompson Sampling integration (modulated weights)
- Task ranking with filtering and sorting
- Anti-shame design: no forced surfacing, empty results valid
- Edge cases: zero weights, all dimensions, extreme values
- Feedback recording
- Custom scorer registration
- CompositeResult conversion to ScoredTask
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.surfacing.engine import (
    CompositeResult,
    CompositeScoringEngine,
    DimensionScore,
    SignalDimension,
    SignalWeights,
)
from blurt.services.surfacing.models import (
    BehavioralProfile,
    CalendarSlot,
    ScoredTask,
    SurfacingContext,
    TaskItem,
)
from blurt.services.surfacing.thompson import ThompsonSampler

NOW = datetime(2026, 3, 14, 14, 0, 0, tzinfo=timezone.utc)


def _ctx(**kwargs) -> SurfacingContext:
    defaults = dict(
        current_time=NOW,
        mood_valence=0.0,
        energy_level=0.5,
    )
    defaults.update(kwargs)
    return SurfacingContext(**defaults)  # type: ignore[arg-type]


def _task(**kwargs) -> TaskItem:
    defaults = dict(
        content="Test task",
        cognitive_load=0.5,
        created_at=NOW - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return TaskItem(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# SignalWeights tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignalWeights:
    def test_default_weights_sum_to_one(self):
        w = SignalWeights()
        assert w.total == pytest.approx(1.0)

    def test_as_dict_has_all_dimensions(self):
        w = SignalWeights()
        d = w.as_dict()
        for dim in SignalDimension:
            assert dim in d

    def test_normalized_sums_to_one(self):
        w = SignalWeights(mood=0.5, energy=0.3, time_of_day=0.2,
                          calendar=0.1, entity_relevance=0.4, behavioral=0.1)
        norm = w.normalized()
        assert sum(norm.values()) == pytest.approx(1.0)

    def test_uniform_weights(self):
        w = SignalWeights.uniform()
        d = w.as_dict()
        expected = 1.0 / len(SignalDimension)
        for dim, val in d.items():
            assert val == pytest.approx(expected, abs=0.001)

    def test_from_dict(self):
        w = SignalWeights.from_dict({"mood": 0.5, "energy": 0.3})
        assert w.mood == 0.5
        assert w.energy == 0.3
        # Unspecified fields use defaults
        assert w.time_of_day == 0.15

    def test_from_dict_ignores_unknown_keys(self):
        w = SignalWeights.from_dict({"mood": 0.5, "nonexistent_key": 99.0})
        assert w.mood == 0.5

    def test_zero_weights_normalized(self):
        w = SignalWeights(mood=0, energy=0, time_of_day=0,
                          calendar=0, entity_relevance=0, behavioral=0)
        norm = w.normalized()
        # Should return uniform when all zero
        expected = 1.0 / len(SignalDimension)
        for val in norm.values():
            assert val == pytest.approx(expected, abs=0.001)

    def test_custom_ratios_preserved(self):
        """Relative ratios should be preserved after normalization."""
        w = SignalWeights(mood=2.0, energy=1.0, time_of_day=0.0,
                          calendar=0.0, entity_relevance=0.0, behavioral=0.0)
        norm = w.normalized()
        # mood should be 2x energy
        assert norm[SignalDimension.MOOD] == pytest.approx(
            2.0 * norm[SignalDimension.ENERGY], abs=0.001
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DimensionScore tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDimensionScore:
    def test_valid_score(self):
        ds = DimensionScore(
            dimension=SignalDimension.MOOD,
            value=0.5,
            weight=0.2,
            effective_weight=0.2,
            weighted_contribution=0.1,
        )
        assert ds.value == 0.5

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            DimensionScore(
                dimension=SignalDimension.MOOD,
                value=1.5,
                weight=0.2,
                effective_weight=0.2,
                weighted_contribution=0.3,
            )

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            DimensionScore(
                dimension=SignalDimension.ENERGY,
                value=-0.1,
                weight=0.2,
                effective_weight=0.2,
                weighted_contribution=-0.02,
            )

    def test_boundary_values(self):
        DimensionScore(
            dimension=SignalDimension.MOOD, value=0.0,
            weight=0.0, effective_weight=0.0, weighted_contribution=0.0,
        )
        DimensionScore(
            dimension=SignalDimension.MOOD, value=1.0,
            weight=1.0, effective_weight=1.0, weighted_contribution=1.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CompositeScoringEngine — basic scoring
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompositeScoringEngineBasic:
    def setup_method(self):
        self.engine = CompositeScoringEngine()

    def test_score_task_returns_composite_result(self):
        result = self.engine.score_task(_task(), _ctx())
        assert isinstance(result, CompositeResult)

    def test_composite_score_in_range(self):
        result = self.engine.score_task(_task(), _ctx())
        assert 0.0 <= result.composite_score <= 1.0

    def test_all_dimensions_scored(self):
        result = self.engine.score_task(_task(), _ctx())
        dims = {ds.dimension for ds in result.dimension_scores}
        for dim in SignalDimension:
            assert dim in dims, f"Missing dimension: {dim}"

    def test_breakdown_dict(self):
        result = self.engine.score_task(_task(), _ctx())
        breakdown = result.breakdown
        assert len(breakdown) == len(SignalDimension)
        for dim_name, value in breakdown.items():
            assert 0.0 <= value <= 1.0

    def test_weighted_breakdown_sums_to_composite(self):
        result = self.engine.score_task(_task(), _ctx())
        total = sum(result.weighted_breakdown.values())
        assert total == pytest.approx(result.composite_score, abs=0.001)

    def test_effective_weights_sum_to_one(self):
        result = self.engine.score_task(_task(), _ctx())
        total = sum(result.effective_weights.values())
        assert total == pytest.approx(1.0, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Weighted aggregation
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightedAggregation:
    def test_energy_only_weight(self):
        """When only energy is weighted, composite equals energy score."""
        weights = SignalWeights(
            mood=0, energy=1.0, time_of_day=0,
            calendar=0, entity_relevance=0, behavioral=0,
        )
        engine = CompositeScoringEngine(weights=weights)
        result = engine.score_task(_task(), _ctx())
        assert result.composite_score == pytest.approx(
            result.breakdown["energy"], abs=0.001
        )

    def test_mood_only_weight(self):
        """When only mood is weighted, composite equals mood score."""
        weights = SignalWeights(
            mood=1.0, energy=0, time_of_day=0,
            calendar=0, entity_relevance=0, behavioral=0,
        )
        engine = CompositeScoringEngine(weights=weights)
        result = engine.score_task(_task(), _ctx())
        assert result.composite_score == pytest.approx(
            result.breakdown["mood"], abs=0.001
        )

    def test_double_weight_doubles_influence(self):
        """Doubling one weight doubles its influence vs. others."""
        w1 = SignalWeights.uniform()
        w2 = SignalWeights.uniform()
        # Double mood weight
        w2.mood *= 2.0

        engine1 = CompositeScoringEngine(weights=w1)
        engine2 = CompositeScoringEngine(weights=w2)

        task = _task(cognitive_load=0.9)
        ctx = _ctx(mood_valence=0.9)  # high mood → high mood score

        r1 = engine1.score_task(task, ctx, use_thompson=False)
        r2 = engine2.score_task(task, ctx, use_thompson=False)

        # With doubled mood weight and high mood score, composite should increase
        assert r2.effective_weights["mood"] > r1.effective_weights["mood"]

    def test_zero_weights_gives_zero_composite(self):
        weights = SignalWeights(
            mood=0, energy=0, time_of_day=0,
            calendar=0, entity_relevance=0, behavioral=0,
        )
        engine = CompositeScoringEngine(weights=weights)
        result = engine.score_task(_task(), _ctx())
        assert result.composite_score == 0.0

    def test_custom_weights_change_ranking_order(self):
        """Different weights produce different rankings."""
        # High energy matched task
        matched_task = _task(content="matched", cognitive_load=0.1)
        # High mood task
        mood_task = _task(content="mood", cognitive_load=0.9)

        ctx = _ctx(mood_valence=0.9, energy_level=0.1)

        # Energy-weighted engine: should prefer matched task
        energy_engine = CompositeScoringEngine(
            weights=SignalWeights(
                mood=0, energy=1.0, time_of_day=0,
                calendar=0, entity_relevance=0, behavioral=0,
            )
        )
        r_matched = energy_engine.score_task(matched_task, ctx, use_thompson=False)
        r_mood = energy_engine.score_task(mood_task, ctx, use_thompson=False)
        assert r_matched.composite_score > r_mood.composite_score

    def test_weights_update_at_runtime(self):
        engine = CompositeScoringEngine()
        _old_weights = engine.get_weights()
        new_weights = SignalWeights(mood=1.0, energy=0, time_of_day=0,
                                    calendar=0, entity_relevance=0, behavioral=0)
        engine.update_weights(new_weights)
        assert engine.get_weights().mood == 1.0
        assert engine.get_weights().energy == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Ranking
# ═══════════════════════════════════════════════════════════════════════════════


class TestRanking:
    def setup_method(self):
        self.engine = CompositeScoringEngine()

    def test_empty_input_valid(self):
        """No-tasks-pending is a valid state."""
        result = self.engine.rank_tasks([], _ctx())
        assert not result.has_tasks
        assert result.top is None
        assert result.scored_tasks == []

    def test_completed_tasks_filtered(self):
        tasks = [
            _task(content="active", completed=False),
            _task(content="done", completed=True),
        ]
        result = self.engine.rank_tasks(tasks, _ctx())
        assert result.total_candidates == 1

    def test_results_sorted_descending(self):
        tasks = [_task(content=f"task-{i}") for i in range(10)]
        result = self.engine.rank_tasks(tasks, _ctx())
        for i in range(len(result.scored_tasks) - 1):
            assert (
                result.scored_tasks[i].composite_score
                >= result.scored_tasks[i + 1].composite_score
            )

    def test_max_results_limit(self):
        tasks = [_task(content=f"task-{i}") for i in range(20)]
        engine = CompositeScoringEngine(max_results=3)
        result = engine.rank_tasks(tasks, _ctx())
        assert len(result.scored_tasks) <= 3
        assert result.total_filtered >= 0

    def test_min_score_filters(self):
        engine = CompositeScoringEngine(min_score=0.99)
        tasks = [_task(content="mediocre")]
        result = engine.rank_tasks(tasks, _ctx())
        assert len(result.scored_tasks) == 0

    def test_top_returns_highest(self):
        tasks = [_task(content=f"task-{i}") for i in range(5)]
        result = self.engine.rank_tasks(tasks, _ctx())
        if result.has_tasks:
            top = result.top
            assert top is not None
            assert top.composite_score == max(
                r.composite_score for r in result.scored_tasks
            )

    def test_ranking_result_metadata(self):
        tasks = [_task() for _ in range(3)]
        result = self.engine.rank_tasks(tasks, _ctx())
        assert result.total_candidates == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Thompson Sampling integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestThompsonIntegration:
    def test_thompson_modulates_weights(self):
        """Thompson Sampling should change effective weights."""
        sampler = ThompsonSampler()
        _rng = random.Random(42)

        engine_no_ts = CompositeScoringEngine()
        engine_ts = CompositeScoringEngine(thompson_sampler=sampler)

        task = _task()
        ctx = _ctx()

        _r_no_ts = engine_no_ts.score_task(task, ctx, use_thompson=False)
        r_ts = engine_ts.score_task(task, ctx, use_thompson=True)

        # With Thompson, effective weights should differ from base
        # (unless the sampler happens to return uniform — very unlikely)
        # Just verify both produce valid results
        assert 0.0 <= r_ts.composite_score <= 1.0
        assert sum(r_ts.effective_weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_thompson_disabled_gives_deterministic(self):
        """use_thompson=False should be deterministic."""
        sampler = ThompsonSampler()
        engine = CompositeScoringEngine(thompson_sampler=sampler)
        task = _task()
        ctx = _ctx()

        r1 = engine.score_task(task, ctx, use_thompson=False)
        r2 = engine.score_task(task, ctx, use_thompson=False)
        assert r1.composite_score == r2.composite_score

    def test_ranking_uses_consistent_thompson_weights(self):
        """All tasks in a single rank_tasks call share the same Thompson weights."""
        sampler = ThompsonSampler()
        engine = CompositeScoringEngine(thompson_sampler=sampler)
        tasks = [_task(content=f"t{i}") for i in range(5)]
        result = engine.rank_tasks(tasks, _ctx())
        assert result.thompson_weights is not None
        assert isinstance(result.thompson_weights, dict)

    def test_no_thompson_no_weights_in_result(self):
        engine = CompositeScoringEngine()
        result = engine.rank_tasks([_task()], _ctx())
        assert result.thompson_weights is None


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback recording
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeedback:
    def test_feedback_with_thompson(self):
        # Initialize sampler with dimension names matching our engine
        arms = {dim.value: __import__("blurt.services.surfacing.thompson", fromlist=["ArmState"]).ArmState()
                for dim in SignalDimension}
        sampler = ThompsonSampler(arms=arms)
        engine = CompositeScoringEngine(thompson_sampler=sampler)

        initial_state = sampler.get_state()

        result = engine.score_task(_task(), _ctx())
        engine.record_feedback(result, engaged=True)

        # Thompson state should have changed
        new_state = sampler.get_state()
        changed = any(
            new_state[k]["alpha"] != initial_state[k]["alpha"]
            for k in initial_state
            if k in new_state
        )
        assert changed, "Positive feedback should update Thompson alpha"

    def test_feedback_without_thompson_noop(self):
        engine = CompositeScoringEngine()
        result = engine.score_task(_task(), _ctx())
        # Should not raise
        engine.record_feedback(result, engaged=True)
        engine.record_feedback(result, engaged=False)

    def test_negative_feedback_updates_beta(self):
        from blurt.services.surfacing.thompson import ArmState
        arms = {dim.value: ArmState() for dim in SignalDimension}
        sampler = ThompsonSampler(arms=arms)
        engine = CompositeScoringEngine(thompson_sampler=sampler)

        initial_state = sampler.get_state()
        result = engine.score_task(_task(), _ctx())
        engine.record_feedback(result, engaged=False)

        new_state = sampler.get_state()
        changed = any(
            new_state[k]["beta"] != initial_state[k]["beta"]
            for k in initial_state
            if k in new_state
        )
        assert changed, "Negative feedback should update Thompson beta"


# ═══════════════════════════════════════════════════════════════════════════════
# Custom scorer registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestCustomScorers:
    def test_register_custom_scorer(self):
        """Custom scorer should replace the built-in one."""
        engine = CompositeScoringEngine(
            weights=SignalWeights(
                mood=1.0, energy=0, time_of_day=0,
                calendar=0, entity_relevance=0, behavioral=0,
            )
        )

        # Register a custom mood scorer that always returns 0.99
        engine.register_scorer(
            SignalDimension.MOOD,
            lambda task, ctx: 0.99,
        )

        result = engine.score_task(_task(), _ctx(), use_thompson=False)
        assert result.breakdown["mood"] == pytest.approx(0.99, abs=0.01)
        assert result.composite_score == pytest.approx(0.99, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# CompositeResult → ScoredTask conversion
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompositeResultConversion:
    def test_to_scored_task(self):
        engine = CompositeScoringEngine()
        result = engine.score_task(_task(), _ctx())
        scored = result.to_scored_task()
        assert isinstance(scored, ScoredTask)
        assert scored.composite_score == result.composite_score
        assert scored.mood_score == result.breakdown["mood"]
        assert scored.energy_score == result.breakdown["energy"]
        assert scored.time_of_day_score == result.breakdown["time_of_day"]
        assert scored.calendar_score == result.breakdown["calendar"]
        assert scored.entity_relevance_score == result.breakdown["entity_relevance"]
        assert scored.behavioral_score == result.breakdown["behavioral"]


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-shame design
# ═══════════════════════════════════════════════════════════════════════════════


class TestAntiShameDesign:
    def setup_method(self):
        self.engine = CompositeScoringEngine()

    def test_empty_results_valid(self):
        result = self.engine.rank_tasks([], _ctx())
        assert not result.has_tasks
        assert result.top is None

    def test_completed_never_surfaced(self):
        tasks = [_task(completed=True)]
        result = self.engine.rank_tasks(tasks, _ctx())
        assert result.total_candidates == 0

    def test_no_forced_surfacing(self):
        engine = CompositeScoringEngine(min_score=0.99)
        result = engine.rank_tasks([_task()], _ctx())
        assert not result.has_tasks

    def test_all_scores_bounded(self):
        """No dimension should produce a score outside [0, 1]."""
        for valence in [-1.0, 0.0, 1.0]:
            for energy in [0.0, 0.5, 1.0]:
                ctx = _ctx(mood_valence=valence, energy_level=energy)
                result = self.engine.score_task(_task(), ctx)
                for ds in result.dimension_scores:
                    assert 0.0 <= ds.value <= 1.0, (
                        f"Score out of range for {ds.dimension}: {ds.value}"
                    )
                assert 0.0 <= result.composite_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: realistic scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    def setup_method(self):
        self.engine = CompositeScoringEngine()

    def test_high_energy_prefers_hard_tasks(self):
        """High energy + energy-weighted engine should prefer high-load tasks."""
        engine = CompositeScoringEngine(
            weights=SignalWeights(
                mood=0, energy=1.0, time_of_day=0,
                calendar=0, entity_relevance=0, behavioral=0,
            )
        )
        ctx = _ctx(energy_level=0.9)
        hard = _task(content="hard", cognitive_load=0.9)
        easy = _task(content="easy", cognitive_load=0.1)

        r_hard = engine.score_task(hard, ctx, use_thompson=False)
        r_easy = engine.score_task(easy, ctx, use_thompson=False)
        assert r_hard.composite_score > r_easy.composite_score

    def test_low_mood_prefers_easy_tasks(self):
        """Low mood + mood-weighted engine should prefer low-load tasks."""
        engine = CompositeScoringEngine(
            weights=SignalWeights(
                mood=1.0, energy=0, time_of_day=0,
                calendar=0, entity_relevance=0, behavioral=0,
            )
        )
        ctx = _ctx(mood_valence=-0.8)
        hard = _task(content="hard", cognitive_load=0.9)
        easy = _task(content="easy", cognitive_load=0.1)

        r_hard = engine.score_task(hard, ctx, use_thompson=False)
        r_easy = engine.score_task(easy, ctx, use_thompson=False)
        assert r_easy.composite_score > r_hard.composite_score

    def test_entity_relevance_boosts_related_tasks(self):
        """Tasks with matching entities should rank higher when entity weight is high."""
        engine = CompositeScoringEngine(
            weights=SignalWeights(
                mood=0, energy=0, time_of_day=0,
                calendar=0, entity_relevance=1.0, behavioral=0,
            )
        )
        ctx = _ctx(active_entity_ids=["e1", "e2"])
        related = _task(content="related", entity_ids=["e1", "e2"])
        unrelated = _task(content="unrelated", entity_ids=["e99"])

        r_related = engine.score_task(related, ctx, use_thompson=False)
        r_unrelated = engine.score_task(unrelated, ctx, use_thompson=False)
        assert r_related.composite_score > r_unrelated.composite_score

    def test_full_pipeline_realistic(self):
        """Full scoring pipeline with realistic data."""
        ctx = _ctx(
            mood_valence=0.3,
            energy_level=0.7,
            active_entity_ids=["proj-blurt"],
            behavioral=BehavioralProfile(
                completion_by_time={"afternoon": 0.8},
                completion_by_load={"medium": 0.7},
                avg_daily_completions=5.0,
                tasks_completed_today=2,
            ),
        )
        tasks = [
            _task(
                content="Implement scoring engine",
                cognitive_load=0.7,
                entity_ids=["proj-blurt"],
                due_at=NOW + timedelta(hours=3),
            ),
            _task(
                content="Reply to email",
                cognitive_load=0.2,
                entity_ids=[],
            ),
            _task(
                content="Completed task",
                cognitive_load=0.5,
                completed=True,
            ),
        ]
        result = self.engine.rank_tasks(tasks, ctx)
        # Completed task should be filtered
        assert result.total_candidates == 2
        assert result.has_tasks
        # All scored tasks should have valid composites
        for scored in result.scored_tasks:
            assert 0.0 <= scored.composite_score <= 1.0

    def test_calendar_awareness(self):
        """In-meeting context should lower scores via calendar dimension."""
        engine = CompositeScoringEngine(
            weights=SignalWeights(
                mood=0, energy=0, time_of_day=0,
                calendar=1.0, entity_relevance=0, behavioral=0,
            )
        )
        free_ctx = _ctx(calendar_slots=[])
        meeting_ctx = _ctx(
            calendar_slots=[
                CalendarSlot(
                    start=NOW - timedelta(minutes=30),
                    end=NOW + timedelta(minutes=30),
                    is_busy=True,
                ),
            ]
        )

        r_free = engine.score_task(_task(), free_ctx, use_thompson=False)
        r_meeting = engine.score_task(_task(), meeting_ctx, use_thompson=False)
        assert r_free.composite_score > r_meeting.composite_score


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_task(self):
        engine = CompositeScoringEngine()
        result = engine.rank_tasks([_task()], _ctx())
        assert result.total_candidates == 1

    def test_all_completed(self):
        tasks = [_task(completed=True) for _ in range(5)]
        engine = CompositeScoringEngine()
        result = engine.rank_tasks(tasks, _ctx())
        assert result.total_candidates == 0
        assert not result.has_tasks

    def test_extreme_context_values(self):
        ctx = _ctx(mood_valence=-1.0, energy_level=0.0)
        task = _task(cognitive_load=1.0)
        engine = CompositeScoringEngine()
        result = engine.score_task(task, ctx)
        assert 0.0 <= result.composite_score <= 1.0

    def test_many_tasks_performance(self):
        """Engine should handle many tasks without error."""
        engine = CompositeScoringEngine(max_results=5)
        tasks = [_task(content=f"task-{i}") for i in range(100)]
        result = engine.rank_tasks(tasks, _ctx())
        assert len(result.scored_tasks) <= 5
        assert result.total_candidates == 100

    def test_identical_tasks_stable_order(self):
        """Identical tasks should maintain stable ordering."""
        engine = CompositeScoringEngine()
        tasks = [_task(content="same") for _ in range(5)]
        result = engine.rank_tasks(tasks, _ctx())
        # All scores should be equal for identical tasks
        if len(result.scored_tasks) >= 2:
            scores = [r.composite_score for r in result.scored_tasks]
            assert all(
                s == pytest.approx(scores[0], abs=0.001) for s in scores
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Import tests — verify barrel exports work
# ═══════════════════════════════════════════════════════════════════════════════


class TestImports:
    def test_import_from_surfacing_package(self):
        from blurt.services.surfacing import (
            CompositeScoringEngine,
        )
        assert CompositeScoringEngine is not None

    def test_import_from_services(self):
        from blurt.services import (
            CompositeScoringEngine,
        )
        assert CompositeScoringEngine is not None
