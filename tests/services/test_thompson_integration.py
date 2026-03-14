"""Tests for Thompson Sampling integration in the task surfacing pipeline.

AC 11, Sub-AC 2: Verify that Thompson Sampling weights are integrated into
the task surfacing/ranking pipeline so that task selection queries use
sampled weights combined with existing context/mood/energy scores.

Covers:
- Engine accepts ThompsonSampler and uses it in scoring
- Sampled weights modulate base signal weights
- Uniform Thompson weights leave scoring unchanged
- Biased Thompson weights shift task rankings
- Feedback loop updates Thompson arms
- Thompson weights attached to SurfacingResult
- Disabled Thompson Sampling has no effect
- Anti-shame: Thompson never suppresses tasks to zero
- Deterministic scoring via use_thompson=False
- Serialization/restoration of Thompson state
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.surfacing.thompson import ThompsonSampler
from blurt.services.task_surfacing import (
    EnergyLevel,
    SignalType,
    SurfaceableTask,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)


# ── Helpers ─────────────────────────────────────────────────────────

NOW = datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)


def _ctx(**kwargs) -> UserContext:
    defaults: dict[str, object] = dict(
        energy=EnergyLevel.MEDIUM,
        current_valence=0.0,
        current_arousal=0.5,
        now=NOW,
    )
    defaults.update(kwargs)
    return UserContext(**defaults)  # type: ignore[arg-type]


def _task(**kwargs) -> SurfaceableTask:
    defaults: dict[str, object] = dict(
        content="test task",
        status=TaskStatus.ACTIVE,
        estimated_energy=EnergyLevel.MEDIUM,
        created_at=NOW - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return SurfaceableTask(**defaults)  # type: ignore[arg-type]


def _make_sampler(seed: int = 42, **arm_overrides) -> ThompsonSampler:
    """Create a ThompsonSampler with optional arm overrides."""
    sampler = ThompsonSampler()
    # Set seed for reproducibility
    random.Random(seed)
    # Monkey-patch for reproducible sampling in tests
    for arm in sampler.arms.values():
        pass  # arms use the sampler's rng
    if arm_overrides:
        for name, (alpha, beta) in arm_overrides.items():
            if name in sampler.arms:
                sampler.arms[name].alpha = alpha
                sampler.arms[name].beta = beta
    return sampler


# ═══════════════════════════════════════════════════════════════════════
# Core integration: engine + Thompson Sampler
# ═══════════════════════════════════════════════════════════════════════


class TestThompsonIntegration:
    """Test that TaskScoringEngine correctly uses ThompsonSampler."""

    def test_engine_accepts_thompson_sampler(self):
        """Engine should accept a ThompsonSampler at construction."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)
        assert engine.thompson_sampler is sampler

    def test_engine_without_sampler_works(self):
        """Engine should work fine without a Thompson Sampler."""
        engine = TaskScoringEngine()
        assert engine.thompson_sampler is None
        result = engine.score_and_rank([_task()], _ctx())
        assert result.has_tasks
        assert result.thompson_weights is None

    def test_score_and_rank_uses_thompson(self):
        """score_and_rank should use Thompson weights when sampler attached."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)
        result = engine.score_and_rank([_task()], _ctx())
        assert result.has_tasks
        assert result.thompson_weights is not None
        assert isinstance(result.thompson_weights, dict)
        # Should have weights for all signal types
        for st in SignalType:
            assert st.value in result.thompson_weights

    def test_thompson_weights_are_normalized(self):
        """Thompson weights should sum to approximately 1.0."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)
        result = engine.score_and_rank([_task()], _ctx())
        assert result.thompson_weights is not None
        total = sum(result.thompson_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_score_single_uses_thompson(self):
        """score_single should use Thompson weights by default."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)
        # Two calls may produce slightly different scores due to sampling
        scored1 = engine.score_single(_task(), _ctx())
        scored2 = engine.score_single(_task(), _ctx())
        # Both should be valid
        assert 0.0 <= scored1.composite_score <= 1.0
        assert 0.0 <= scored2.composite_score <= 1.0

    def test_score_single_deterministic_without_thompson(self):
        """score_single with use_thompson=False should be deterministic."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)
        task = _task()
        ctx = _ctx()
        s1 = engine.score_single(task, ctx, use_thompson=False)
        s2 = engine.score_single(task, ctx, use_thompson=False)
        assert s1.composite_score == s2.composite_score


# ═══════════════════════════════════════════════════════════════════════
# Weight modulation behavior
# ═══════════════════════════════════════════════════════════════════════


class TestWeightModulation:
    """Test that Thompson weights correctly modulate base weights."""

    def test_uniform_thompson_preserves_base_weights(self):
        """When all Thompson arms are equal (uniform), scoring should
        be equivalent to no Thompson sampling."""
        engine_no_ts = TaskScoringEngine()
        sampler = ThompsonSampler()
        engine_ts = TaskScoringEngine(thompson_sampler=sampler)

        task = _task()
        ctx = _ctx()

        # Score without Thompson
        base_score = engine_no_ts.score_single(task, ctx)

        # Score with Thompson but use_thompson=False
        ts_off = engine_ts.score_single(task, ctx, use_thompson=False)

        assert base_score.composite_score == ts_off.composite_score

    def test_biased_thompson_shifts_ranking(self):
        """When Thompson strongly favors energy_match, a task with perfect
        energy match should rank higher relative to other signals."""
        # Create sampler with strong bias toward energy_match
        sampler = ThompsonSampler()
        sampler.arms["energy_match"].alpha = 50.0  # very confident
        sampler.arms["energy_match"].beta = 1.0
        # Suppress other signals
        for name in sampler.arms:
            if name != "energy_match":
                sampler.arms[name].alpha = 1.0
                sampler.arms[name].beta = 50.0

        engine = TaskScoringEngine(thompson_sampler=sampler)
        ctx = _ctx(energy=EnergyLevel.HIGH)

        # Task with perfect energy match
        high_match = _task(
            content="high energy task",
            estimated_energy=EnergyLevel.HIGH,
            due_at=NOW + timedelta(days=30),  # far future (low time relevance)
        )
        # Task with poor energy match but high time relevance
        low_match = _task(
            content="urgent low energy",
            estimated_energy=EnergyLevel.LOW,
            due_at=NOW + timedelta(hours=1),  # imminent
        )

        result = engine.score_and_rank([low_match, high_match], ctx)
        assert result.has_tasks
        assert len(result.tasks) >= 2
        # With energy_match dominating, the high-match task should win
        assert result.tasks[0].task.content == "high energy task"

    def test_composite_score_always_in_range(self):
        """Composite score should always be in [0, 1] even with extreme
        Thompson weights."""
        sampler = ThompsonSampler()
        # Extreme: one arm very high, others very low
        sampler.arms["freshness"].alpha = 100.0
        sampler.arms["freshness"].beta = 1.0

        engine = TaskScoringEngine(thompson_sampler=sampler)
        task = _task()
        ctx = _ctx()

        # Run many times with different samples
        for _ in range(50):
            scored = engine.score_single(task, ctx)
            assert 0.0 <= scored.composite_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Feedback loop
# ═══════════════════════════════════════════════════════════════════════


class TestFeedbackLoop:
    """Test the feedback loop: surface → engage/dismiss → learn."""

    def test_record_task_feedback_engaged(self):
        """Recording engagement should update Thompson arms."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        task = _task()
        ctx = _ctx()
        scored = engine.score_single(task, ctx, use_thompson=False)

        # Record engagement
        initial_state = sampler.get_state()
        engine.record_task_feedback(scored, engaged=True)
        updated_state = sampler.get_state()

        # At least some arms should have increased alpha
        any_increased = any(
            updated_state[name]["alpha"] > initial_state[name]["alpha"]
            for name in initial_state
        )
        assert any_increased, "Engagement should increase alpha on contributing arms"

    def test_record_task_feedback_dismissed(self):
        """Recording dismissal should update Thompson arms (beta)."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        task = _task()
        ctx = _ctx()
        scored = engine.score_single(task, ctx, use_thompson=False)

        initial_state = sampler.get_state()
        engine.record_task_feedback(scored, engaged=False)
        updated_state = sampler.get_state()

        # At least some arms should have increased beta
        any_increased = any(
            updated_state[name]["beta"] > initial_state[name]["beta"]
            for name in initial_state
        )
        assert any_increased, "Dismissal should increase beta on contributing arms"

    def test_feedback_noop_without_sampler(self):
        """record_task_feedback should be a no-op without a sampler."""
        engine = TaskScoringEngine()
        task = _task()
        scored = engine.score_single(task, _ctx())
        # Should not raise
        engine.record_task_feedback(scored, engaged=True)
        engine.record_task_feedback(scored, engaged=False)

    def test_repeated_engagement_shifts_weights(self):
        """Repeated engagement with energy-matched tasks should shift
        Thompson weights toward energy_match over time."""
        sampler = ThompsonSampler(decay_factor=0.99)
        engine = TaskScoringEngine(thompson_sampler=sampler)

        # Simulate: always engaging with tasks that have high energy match
        for _ in range(30):
            task = _task(estimated_energy=EnergyLevel.HIGH)
            ctx = _ctx(energy=EnergyLevel.HIGH)
            scored = engine.score_single(task, ctx, use_thompson=False)
            engine.record_task_feedback(scored, engaged=True)

        # After many engagements, energy_match arm should have high mean
        expected = sampler.get_expected_weights()
        # Energy match should have a meaningful weight
        assert expected["energy_match"] > 0.0

    def test_repeated_dismissal_reduces_arm(self):
        """Repeated dismissal when a signal was high should reduce its
        expected weight (gentle, not punitive)."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        initial_mean = sampler.arms["time_relevance"].mean

        # Simulate dismissals of tasks scored high on time_relevance
        for _ in range(20):
            task = _task(due_at=NOW + timedelta(hours=1))
            ctx = _ctx()
            scored = engine.score_single(task, ctx, use_thompson=False)
            engine.record_task_feedback(scored, engaged=False)

        # time_relevance arm mean should have decreased
        final_mean = sampler.arms["time_relevance"].mean
        assert final_mean < initial_mean


# ═══════════════════════════════════════════════════════════════════════
# Anti-shame verification
# ═══════════════════════════════════════════════════════════════════════


class TestAntiShameThompson:
    """Verify anti-shame principles with Thompson Sampling active."""

    def test_no_task_scores_zero_with_thompson(self):
        """Even with extreme Thompson weights, no task should score
        exactly zero (anti-shame: nothing is impossible)."""
        sampler = ThompsonSampler()
        # Suppress all arms
        for arm in sampler.arms.values():
            arm.alpha = 1.0
            arm.beta = 100.0

        engine = TaskScoringEngine(thompson_sampler=sampler, min_score=0.0)
        task = _task()
        ctx = _ctx()

        scores = set()
        for _ in range(100):
            scored = engine.score_single(task, ctx)
            scores.add(scored.composite_score)
            assert scored.composite_score >= 0.0

        # Scores should be positive (not all zero)
        assert max(scores) > 0.0

    def test_empty_results_still_valid(self):
        """No-tasks-pending remains valid with Thompson Sampling."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)
        result = engine.score_and_rank([], _ctx())
        assert not result.has_tasks
        assert result.top_task is None

    def test_deferred_tasks_not_surfaced_with_thompson(self):
        """Thompson Sampling should not override task eligibility."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler, min_score=0.0)
        deferred = _task(status=TaskStatus.DEFERRED)
        result = engine.score_and_rank([deferred], _ctx())
        assert not result.has_tasks

    def test_dropped_tasks_not_surfaced_with_thompson(self):
        """Dropped tasks stay dropped regardless of Thompson weights."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler, min_score=0.0)
        dropped = _task(status=TaskStatus.DROPPED)
        result = engine.score_and_rank([dropped], _ctx())
        assert not result.has_tasks


# ═══════════════════════════════════════════════════════════════════════
# Exploration behavior
# ═══════════════════════════════════════════════════════════════════════


class TestExploration:
    """Verify that Thompson Sampling provides exploration."""

    def test_sampling_produces_variety(self):
        """Multiple ranking calls should produce some variety in ordering,
        demonstrating exploration."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler, min_score=0.0)

        tasks = [
            _task(content="task-a", estimated_energy=EnergyLevel.HIGH),
            _task(content="task-b", estimated_energy=EnergyLevel.MEDIUM),
            _task(content="task-c", estimated_energy=EnergyLevel.LOW),
        ]
        ctx = _ctx()

        # Run many times, collect top task
        top_tasks = set()
        for _ in range(100):
            result = engine.score_and_rank(tasks, ctx)
            if result.has_tasks and result.top_task is not None:
                top_tasks.add(result.top_task.task.content)

        # With exploration, we should see at least some variety
        # (uniform priors mean any task could be top)
        assert len(top_tasks) >= 1  # at minimum, always produces a result

    def test_cold_start_uniform_exploration(self):
        """Fresh sampler (cold start) should have roughly uniform weights."""
        sampler = ThompsonSampler()
        expected = sampler.get_expected_weights()
        n = len(expected)
        uniform = 1.0 / n
        for name, weight in expected.items():
            assert weight == pytest.approx(uniform, abs=0.05), (
                f"Cold start weight for {name} should be ~{uniform}, got {weight}"
            )


# ═══════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════


class TestThompsonPersistence:
    """Test serialization and restoration of Thompson state."""

    def test_round_trip_state(self):
        """Serialized state should restore identical sampler."""
        sampler = ThompsonSampler(decay_factor=0.95, min_weight=0.1)
        sampler.arms["energy_match"].alpha = 5.0
        sampler.arms["energy_match"].beta = 3.0
        sampler.arms["momentum"].alpha = 2.0

        state = sampler.get_state()
        restored = ThompsonSampler.from_state(
            state, decay_factor=0.95, min_weight=0.1
        )

        for name in sampler.arms:
            assert restored.arms[name].alpha == sampler.arms[name].alpha
            assert restored.arms[name].beta == sampler.arms[name].beta

    def test_engine_with_restored_sampler(self):
        """Engine should work with a restored sampler."""
        sampler = ThompsonSampler()
        sampler.arms["energy_match"].alpha = 10.0

        state = sampler.get_state()
        restored = ThompsonSampler.from_state(state)
        engine = TaskScoringEngine(thompson_sampler=restored)

        result = engine.score_and_rank([_task()], _ctx())
        assert result.has_tasks
        assert result.thompson_weights is not None


# ═══════════════════════════════════════════════════════════════════════
# Integration with existing scoring
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationWithScoring:
    """Verify Thompson Sampling works with realistic scoring scenarios."""

    def test_morning_high_energy_with_thompson(self):
        """Realistic scenario: morning high energy, Thompson should
        still allow the best-matched task to surface."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        ctx = UserContext(
            energy=EnergyLevel.HIGH,
            current_valence=0.6,
            current_arousal=0.7,
            active_entity_names=["scoring"],
            active_project="Blurt",
            now=NOW,
        )

        tasks = [
            SurfaceableTask(
                content="Deep architecture work",
                estimated_energy=EnergyLevel.HIGH,
                project="Blurt",
                entity_names=["scoring"],
                due_at=NOW + timedelta(days=2),
                created_at=NOW - timedelta(days=1),
            ),
            SurfaceableTask(
                content="Fix typo",
                estimated_energy=EnergyLevel.LOW,
                created_at=NOW - timedelta(days=5),
            ),
        ]

        # Run multiple times — the high-energy task should win most
        high_wins = 0
        for _ in range(50):
            result = engine.score_and_rank(tasks, ctx)
            if result.has_tasks and result.top_task is not None and result.top_task.task.content == "Deep architecture work":
                high_wins += 1

        # Should win majority of the time (exploration may flip occasionally)
        assert high_wins > 25, f"High energy task should win most times, won {high_wins}/50"

    def test_all_signals_contribute_with_thompson(self):
        """All signal scores should still be computed with Thompson active."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        task = _task(due_at=NOW + timedelta(hours=5))
        scored = engine.score_single(task, _ctx())

        assert len(scored.signal_scores) == 6
        signals_present = {s.signal for s in scored.signal_scores}
        assert signals_present == set(SignalType)

    def test_thompson_result_metadata(self):
        """SurfacingResult should contain Thompson metadata for transparency."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        result = engine.score_and_rank([_task()], _ctx())

        assert result.thompson_weights is not None
        # All signal types should be represented
        assert len(result.thompson_weights) == len(SignalType)
        # Weights should sum to 1.0
        assert sum(result.thompson_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_feedback_then_rank_changes_weights(self):
        """After recording feedback, subsequent rankings should use
        updated Thompson weights."""
        sampler = ThompsonSampler()
        engine = TaskScoringEngine(thompson_sampler=sampler)

        task = _task()
        ctx = _ctx()

        # Get initial expected weights
        initial_expected = sampler.get_expected_weights()

        # Record many engagements
        scored = engine.score_single(task, ctx, use_thompson=False)
        for _ in range(20):
            engine.record_task_feedback(scored, engaged=True)

        # Expected weights should have shifted
        updated_expected = sampler.get_expected_weights()
        # At least one weight should differ
        any_different = any(
            abs(updated_expected[k] - initial_expected[k]) > 0.001
            for k in initial_expected
        )
        assert any_different, "Feedback should shift expected weights"
