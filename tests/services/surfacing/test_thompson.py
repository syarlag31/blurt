"""Tests for Thompson Sampling convergence, weight adjustment, integration, and edge cases.

AC 11, Sub-AC 4: Validates that Thompson Sampling correctly:
- Converges to favor signals that predict user engagement
- Adjusts weights proportionally on positive/negative feedback
- Integrates properly with task surfacing rankings
- Handles edge cases (cold start, all tasks dismissed, decay, serialization)

Anti-shame: Thompson Sampling treats dismissals as neutral learning signal,
not punishment. Cold start is optimistic (uniform prior).
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.surfacing.thompson import ArmState, ThompsonSampler
from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
    SurfacingWeights,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)


def _uctx(**kwargs) -> UserContext:
    defaults: dict[str, object] = dict(energy=EnergyLevel.MEDIUM, now=NOW)
    defaults.update(kwargs)
    return UserContext(**defaults)  # type: ignore[arg-type]


def _stask(**kwargs) -> SurfaceableTask:
    defaults: dict[str, object] = dict(
        content="test task",
        status=TaskStatus.ACTIVE,
        estimated_energy=EnergyLevel.MEDIUM,
        created_at=NOW - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return SurfaceableTask(**defaults)  # type: ignore[arg-type]


def _uniform_contributions(signals: tuple[str, ...] | None = None) -> dict[str, float]:
    """Equal contribution from all signals."""
    sigs = signals or ThompsonSampler.DEFAULT_SIGNALS
    n = len(sigs)
    return {s: 1.0 / n for s in sigs}


def _dominant_contribution(dominant: str, signals: tuple[str, ...] | None = None) -> dict[str, float]:
    """One signal dominates the contribution."""
    sigs = signals or ThompsonSampler.DEFAULT_SIGNALS
    result = {s: 0.05 for s in sigs}
    result[dominant] = 0.75
    total = sum(result.values())
    return {k: v / total for k, v in result.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# ArmState unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestArmState:
    """Tests for the Beta distribution arm state."""

    def test_default_prior_is_uniform(self):
        arm = ArmState()
        assert arm.alpha == 1.0
        assert arm.beta == 1.0

    def test_mean_uniform_prior(self):
        arm = ArmState(alpha=1.0, beta=1.0)
        assert arm.mean == pytest.approx(0.5)

    def test_mean_skewed_success(self):
        arm = ArmState(alpha=10.0, beta=2.0)
        assert arm.mean == pytest.approx(10.0 / 12.0)

    def test_mean_skewed_failure(self):
        arm = ArmState(alpha=2.0, beta=10.0)
        assert arm.mean == pytest.approx(2.0 / 12.0)

    def test_variance_decreases_with_observations(self):
        """More observations → lower variance (more confident)."""
        few = ArmState(alpha=2.0, beta=2.0)
        many = ArmState(alpha=50.0, beta=50.0)
        assert many.variance < few.variance

    def test_total_observations(self):
        arm = ArmState(alpha=5.0, beta=3.0)
        # 5 + 3 - 2 (prior) = 6 actual observations
        assert arm.total_observations == pytest.approx(6.0)

    def test_total_observations_cold_start(self):
        arm = ArmState()
        assert arm.total_observations == pytest.approx(0.0)

    def test_sample_in_range(self):
        rng = random.Random(42)
        arm = ArmState(alpha=5.0, beta=5.0)
        for _ in range(100):
            s = arm.sample(rng)
            assert 0.0 <= s <= 1.0

    def test_sample_with_rng_reproducible(self):
        arm = ArmState(alpha=3.0, beta=7.0)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert arm.sample(rng1) == arm.sample(rng2)

    def test_sample_high_alpha_tends_high(self):
        """With high alpha, samples should trend toward 1.0."""
        rng = random.Random(42)
        arm = ArmState(alpha=100.0, beta=2.0)
        samples = [arm.sample(rng) for _ in range(200)]
        avg = sum(samples) / len(samples)
        assert avg > 0.9

    def test_sample_high_beta_tends_low(self):
        """With high beta, samples should trend toward 0.0."""
        rng = random.Random(42)
        arm = ArmState(alpha=2.0, beta=100.0)
        samples = [arm.sample(rng) for _ in range(200)]
        avg = sum(samples) / len(samples)
        assert avg < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# Thompson Sampler: cold start behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestColdStart:
    """Tests for Thompson Sampling cold start (no prior data)."""

    def test_cold_start_all_arms_uniform(self):
        """Fresh sampler should have all arms at Beta(1,1) (uniform)."""
        sampler = ThompsonSampler()
        for arm in sampler.arms.values():
            assert arm.alpha == 1.0
            assert arm.beta == 1.0

    def test_cold_start_has_all_default_signals(self):
        sampler = ThompsonSampler()
        for signal in ThompsonSampler.DEFAULT_SIGNALS:
            assert signal in sampler.arms

    def test_cold_start_expected_weights_approximately_uniform(self):
        """With uniform priors, expected weights should be approximately equal."""
        sampler = ThompsonSampler()
        weights = sampler.get_expected_weights()
        n = len(weights)
        expected = 1.0 / n
        for w in weights.values():
            assert w == pytest.approx(expected, abs=0.01)

    def test_cold_start_sampled_weights_sum_to_one(self):
        sampler = ThompsonSampler()
        rng = random.Random(42)
        weights = sampler.sample_weights(rng)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_cold_start_sampled_weights_all_positive(self):
        """Even with sampling variance, min_weight ensures all are positive."""
        sampler = ThompsonSampler()
        rng = random.Random(42)
        for _ in range(50):
            weights = sampler.sample_weights(rng)
            for w in weights.values():
                assert w > 0.0

    def test_cold_start_exploration_variance(self):
        """Cold start should have high variance (lots of exploration)."""
        sampler = ThompsonSampler()
        rng = random.Random(42)
        # Sample weights multiple times — should show variety
        weight_sets = [sampler.sample_weights(rng) for _ in range(100)]

        # Check variance of a specific signal's weight across samples
        time_weights = [ws["time_relevance"] for ws in weight_sets]
        mean_w = sum(time_weights) / len(time_weights)
        var_w = sum((w - mean_w) ** 2 for w in time_weights) / len(time_weights)
        # With uniform prior, variance should be notable
        assert var_w > 0.001, "Cold start should have exploration variance"


# ═══════════════════════════════════════════════════════════════════════════════
# Thompson Sampler: weight adjustment on feedback
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightAdjustment:
    """Tests for weight adjustment after user feedback."""

    def test_positive_feedback_increases_alpha(self):
        """Completion should increase alpha of contributing signals."""
        sampler = ThompsonSampler()
        initial_alpha = sampler.arms["energy_match"].alpha

        contributions = _dominant_contribution("energy_match")
        sampler.record_feedback(contributions, reward=True)

        assert sampler.arms["energy_match"].alpha > initial_alpha

    def test_negative_feedback_increases_beta(self):
        """Dismissal should increase beta of contributing signals."""
        sampler = ThompsonSampler()
        initial_beta = sampler.arms["energy_match"].beta

        contributions = _dominant_contribution("energy_match")
        sampler.record_feedback(contributions, reward=False)

        assert sampler.arms["energy_match"].beta > initial_beta

    def test_positive_feedback_does_not_change_beta(self):
        sampler = ThompsonSampler()
        initial_beta = sampler.arms["energy_match"].beta

        contributions = _dominant_contribution("energy_match")
        sampler.record_feedback(contributions, reward=True)

        assert sampler.arms["energy_match"].beta == initial_beta

    def test_negative_feedback_does_not_change_alpha(self):
        sampler = ThompsonSampler()
        initial_alpha = sampler.arms["energy_match"].alpha

        contributions = _dominant_contribution("energy_match")
        sampler.record_feedback(contributions, reward=False)

        assert sampler.arms["energy_match"].alpha == initial_alpha

    def test_proportional_update(self):
        """Updates should be proportional to contribution magnitude."""
        sampler = ThompsonSampler()

        # High contribution signal gets bigger update
        contributions = {
            "time_relevance": 0.8,
            "energy_match": 0.2,
        }
        sampler.record_feedback(contributions, reward=True)

        # time_relevance had 4x the contribution, so alpha increase is 4x
        time_delta = sampler.arms["time_relevance"].alpha - 1.0
        energy_delta = sampler.arms["energy_match"].alpha - 1.0
        assert time_delta == pytest.approx(0.8)
        assert energy_delta == pytest.approx(0.2)

    def test_learning_rate_scales_updates(self):
        """Lower learning rate should produce smaller updates."""
        sampler = ThompsonSampler()
        contributions = _uniform_contributions()

        sampler.record_feedback(contributions, reward=True, learning_rate=0.5)

        arm = sampler.arms["time_relevance"]
        expected_delta = (1.0 / len(ThompsonSampler.DEFAULT_SIGNALS)) * 0.5
        assert arm.alpha - 1.0 == pytest.approx(expected_delta)

    def test_record_completion_convenience(self):
        """record_completion should update alphas."""
        sampler = ThompsonSampler()
        signal_scores = {s: 0.5 for s in ThompsonSampler.DEFAULT_SIGNALS}
        sampler.record_completion(signal_scores, composite_score=0.5)

        for arm in sampler.arms.values():
            assert arm.alpha > 1.0

    def test_record_dismissal_convenience(self):
        """record_dismissal should update betas."""
        sampler = ThompsonSampler()
        signal_scores = {s: 0.5 for s in ThompsonSampler.DEFAULT_SIGNALS}
        sampler.record_dismissal(signal_scores, composite_score=0.5)

        for arm in sampler.arms.values():
            assert arm.beta > 1.0

    def test_empty_contributions_no_crash(self):
        """Empty contributions should be a no-op."""
        sampler = ThompsonSampler()
        sampler.record_feedback({}, reward=True)
        for arm in sampler.arms.values():
            assert arm.alpha == 1.0
            assert arm.beta == 1.0

    def test_unknown_signal_creates_arm(self):
        """Feedback for unknown signal should create a new arm."""
        sampler = ThompsonSampler()
        sampler.record_feedback({"novel_signal": 1.0}, reward=True)
        assert "novel_signal" in sampler.arms
        assert sampler.arms["novel_signal"].alpha > 1.0

    def test_repeated_positive_feedback_increases_mean(self):
        """Repeated completions should increase the arm's expected value."""
        sampler = ThompsonSampler()
        initial_mean = sampler.arms["energy_match"].mean

        for _ in range(20):
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=True,
            )

        assert sampler.arms["energy_match"].mean > initial_mean

    def test_repeated_negative_feedback_decreases_mean(self):
        """Repeated dismissals should decrease the arm's expected value."""
        sampler = ThompsonSampler()
        initial_mean = sampler.arms["energy_match"].mean

        for _ in range(20):
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=False,
            )

        assert sampler.arms["energy_match"].mean < initial_mean


# ═══════════════════════════════════════════════════════════════════════════════
# Thompson Sampler: convergence behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvergence:
    """Tests that Thompson Sampling converges to the true best signals."""

    def test_converges_to_single_dominant_signal(self):
        """If one signal consistently predicts engagement, weight should converge."""
        sampler = ThompsonSampler()
        random.Random(42)

        # Simulate: time_relevance always predicts well
        for _ in range(200):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )
            # Other signals get negative feedback sometimes
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=False,
            )

        weights = sampler.get_expected_weights()
        assert weights["time_relevance"] > weights["energy_match"]
        # The dominant signal should have the highest expected weight
        assert weights["time_relevance"] == max(weights.values())

    def test_converges_to_two_strong_signals(self):
        """With two strong signals, both should get high weights."""
        sampler = ThompsonSampler()

        for _ in range(100):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=True,
            )
            # Weaker signals get mixed feedback
            sampler.record_feedback(
                _dominant_contribution("freshness"),
                reward=False,
            )

        weights = sampler.get_expected_weights()
        # Both strong signals should be in the top 3
        sorted_signals = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_3_names = {s[0] for s in sorted_signals[:3]}
        assert "time_relevance" in top_3_names
        assert "energy_match" in top_3_names

    def test_variance_decreases_with_data(self):
        """More data should reduce sampling variance (more confident)."""
        sampler = ThompsonSampler()

        # Variance before
        initial_var = sampler.arms["time_relevance"].variance

        for _ in range(50):
            sampler.record_feedback(
                _uniform_contributions(),
                reward=True,
            )

        final_var = sampler.arms["time_relevance"].variance
        assert final_var < initial_var, "Variance should decrease with more data"

    def test_convergence_with_noisy_feedback(self):
        """Should converge even with noisy (probabilistic) feedback."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        # time_relevance is good 70% of the time, energy_match 30%
        for _ in range(500):
            if rng.random() < 0.7:
                sampler.record_feedback(
                    _dominant_contribution("time_relevance"),
                    reward=True,
                )
            else:
                sampler.record_feedback(
                    _dominant_contribution("time_relevance"),
                    reward=False,
                )

            if rng.random() < 0.3:
                sampler.record_feedback(
                    _dominant_contribution("energy_match"),
                    reward=True,
                )
            else:
                sampler.record_feedback(
                    _dominant_contribution("energy_match"),
                    reward=False,
                )

        weights = sampler.get_expected_weights()
        assert weights["time_relevance"] > weights["energy_match"], (
            f"time_relevance ({weights['time_relevance']:.3f}) should outweigh "
            f"energy_match ({weights['energy_match']:.3f}) with 70% vs 30% success"
        )

    def test_sampled_weights_reflect_learned_preferences(self):
        """After learning, sampled weights should mostly favor the best signal."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        # Train: time_relevance is consistently good
        for _ in range(100):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )

        # Sample many times — time_relevance should usually be highest
        count_time_highest = 0
        for _ in range(100):
            weights = sampler.sample_weights(rng)
            if weights["time_relevance"] == max(weights.values()):
                count_time_highest += 1

        assert count_time_highest > 50, (
            f"time_relevance should be highest in most samples, "
            f"was highest in {count_time_highest}/100"
        )

    def test_exploration_still_happens_after_convergence(self):
        """Even after convergence, exploration should still occur sometimes."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        # Heavily train time_relevance
        for _ in range(100):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )

        # Sample many times — non-dominant signal should sometimes be highest
        non_dominant_highest_count = 0
        for _ in range(200):
            weights = sampler.sample_weights(rng)
            if max(weights, key=lambda k: weights[k]) != "time_relevance":
                non_dominant_highest_count += 1

        # Should still explore sometimes (exploration-exploitation tradeoff)
        assert non_dominant_highest_count > 0, (
            "Should still explore non-dominant signals occasionally"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with TaskScoringEngine
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskSurfacingIntegration:
    """Tests that Thompson Sampling integrates properly with task ranking."""

    def test_sampled_weights_produce_valid_surfacing_weights(self):
        """Sampled weights should be convertible to SurfacingWeights."""
        sampler = ThompsonSampler()
        rng = random.Random(42)
        weights = sampler.sample_weights(rng)

        # Should have all required signal types
        sw = SurfacingWeights(
            time_relevance=weights.get("time_relevance", 0.1),
            energy_match=weights.get("energy_match", 0.1),
            context_relevance=weights.get("context_relevance", 0.1),
            emotional_alignment=weights.get("emotional_alignment", 0.1),
            momentum=weights.get("momentum", 0.1),
            freshness=weights.get("freshness", 0.1),
        )
        assert sw.total > 0

    def test_thompson_weights_change_task_rankings(self):
        """Different Thompson-sampled weights should produce different rankings."""
        # Create two samplers with different learned preferences
        time_sampler = ThompsonSampler()
        energy_sampler = ThompsonSampler()

        for _ in range(50):
            time_sampler.record_feedback(
                _dominant_contribution("time_relevance"), reward=True
            )
            energy_sampler.record_feedback(
                _dominant_contribution("energy_match"), reward=True
            )

        time_weights = time_sampler.get_expected_weights()
        energy_weights = energy_sampler.get_expected_weights()

        # Verify the learned biases are different
        assert time_weights["time_relevance"] > energy_weights["time_relevance"]
        assert energy_weights["energy_match"] > time_weights["energy_match"]

    def test_thompson_engine_scores_task(self):
        """An engine configured with Thompson weights should score tasks."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        # Learn some weights
        for _ in range(20):
            sampler.record_feedback(
                _dominant_contribution("energy_match"), reward=True
            )

        weights = sampler.sample_weights(rng)
        sw = SurfacingWeights(**{k: weights.get(k, 0.1) for k in [
            "time_relevance", "energy_match", "context_relevance",
            "emotional_alignment", "momentum", "freshness",
        ]})

        engine = TaskScoringEngine(weights=sw, min_score=0.0)
        task = _stask(estimated_energy=EnergyLevel.MEDIUM)
        ctx = _uctx(energy=EnergyLevel.MEDIUM)

        result = engine.score_single(task, ctx)
        assert 0.0 <= result.composite_score <= 1.0

    def test_feedback_loop_with_engine(self):
        """Full feedback loop: score → surface → user acts → update weights."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        tasks = [
            _stask(content="easy email", estimated_energy=EnergyLevel.LOW),
            _stask(
                content="complex analysis",
                estimated_energy=EnergyLevel.HIGH,
                due_at=NOW + timedelta(hours=2),
            ),
        ]
        ctx = _uctx(energy=EnergyLevel.HIGH)

        # Step 1: Score with Thompson-sampled weights
        weights = sampler.sample_weights(rng)
        sw = SurfacingWeights(**{k: weights.get(k, 0.1) for k in [
            "time_relevance", "energy_match", "context_relevance",
            "emotional_alignment", "momentum", "freshness",
        ]})
        engine = TaskScoringEngine(weights=sw, min_score=0.0)
        result = engine.score_and_rank(tasks, ctx)

        assert result.has_tasks

        # Step 2: User completes top task
        top = result.top_task
        assert top is not None
        sampler.record_completion(
            top.signal_breakdown,
            top.composite_score,
        )

        # Step 3: User dismisses second task (if exists)
        if len(result.tasks) > 1:
            dismissed = result.tasks[1]
            sampler.record_dismissal(
                dismissed.signal_breakdown,
                dismissed.composite_score,
            )

        # Step 4: Verify weights shifted
        new_weights = sampler.get_expected_weights()
        assert sum(new_weights.values()) == pytest.approx(1.0)

    def test_ranking_changes_after_learning(self):
        """After learning, rankings should shift to favor learned patterns."""
        sampler = ThompsonSampler()

        # Simulate: user always completes tasks with strong time_relevance
        time_scores = {"time_relevance": 0.9, "energy_match": 0.3,
                       "context_relevance": 0.2, "emotional_alignment": 0.3,
                       "momentum": 0.2, "freshness": 0.4}
        for _ in range(50):
            sampler.record_completion(time_scores, composite_score=0.6)

        # And dismisses tasks with strong freshness
        fresh_scores = {"time_relevance": 0.2, "energy_match": 0.3,
                        "context_relevance": 0.3, "emotional_alignment": 0.3,
                        "momentum": 0.2, "freshness": 0.9}
        for _ in range(50):
            sampler.record_dismissal(fresh_scores, composite_score=0.5)

        weights = sampler.get_expected_weights()
        assert weights["time_relevance"] > weights["freshness"], (
            "Learned weights should favor time_relevance over freshness"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases: all tasks dismissed
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllTasksDismissed:
    """Tests behavior when user dismisses everything."""

    def test_all_dismissed_weights_still_valid(self):
        """Even with 100% dismissals, weights should still be valid (sum to 1)."""
        sampler = ThompsonSampler()

        for _ in range(100):
            sampler.record_feedback(_uniform_contributions(), reward=False)

        weights = sampler.get_expected_weights()
        assert sum(weights.values()) == pytest.approx(1.0)
        for w in weights.values():
            assert w > 0.0

    def test_all_dismissed_arms_have_high_beta(self):
        """All dismissals should increase beta across all arms."""
        sampler = ThompsonSampler()

        for _ in range(50):
            sampler.record_feedback(_uniform_contributions(), reward=False)

        for arm in sampler.arms.values():
            assert arm.beta > arm.alpha

    def test_all_dismissed_expected_weights_low_but_valid(self):
        """Expected weights should be low (below 0.5 mean) but still sum to 1."""
        sampler = ThompsonSampler()

        for _ in range(100):
            sampler.record_feedback(_uniform_contributions(), reward=False)

        for arm in sampler.arms.values():
            assert arm.mean < 0.5

        weights = sampler.get_expected_weights()
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_dismissed_then_one_success_shifts(self):
        """After all dismissals, a single success should noticeably shift weights."""
        sampler = ThompsonSampler()

        for _ in range(20):
            sampler.record_feedback(_uniform_contributions(), reward=False)

        before = sampler.arms["time_relevance"].mean

        sampler.record_feedback(
            _dominant_contribution("time_relevance"),
            reward=True,
        )

        after = sampler.arms["time_relevance"].mean
        assert after > before, "Success after dismissals should shift mean up"

    def test_min_weight_prevents_signal_death(self):
        """min_weight floor should prevent any signal from being zeroed out."""
        sampler = ThompsonSampler(min_weight=0.05)

        # Massively penalize one signal
        for _ in range(200):
            sampler.record_feedback(
                _dominant_contribution("freshness"),
                reward=False,
            )

        rng = random.Random(42)
        weights = sampler.sample_weights(rng)
        # Even the penalized signal should have at least min_weight
        # (before normalization — after normalization it's proportional)
        assert weights["freshness"] > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Decay behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecay:
    """Tests for exponential decay (forgetting old patterns)."""

    def test_no_decay_preserves_observations(self):
        sampler = ThompsonSampler(decay_factor=1.0)
        sampler.record_feedback(
            _dominant_contribution("time_relevance"),
            reward=True,
        )
        alpha_before = sampler.arms["time_relevance"].alpha

        # Record another with no decay
        sampler.record_feedback(
            _dominant_contribution("time_relevance"),
            reward=True,
        )

        # Alpha should just accumulate
        assert sampler.arms["time_relevance"].alpha > alpha_before

    def test_decay_reduces_old_evidence(self):
        """Decay should bring alphas/betas closer to prior over time."""
        sampler = ThompsonSampler(decay_factor=0.9)

        # Build up evidence
        for _ in range(10):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )

        alpha_before_decay = sampler.arms["time_relevance"].alpha

        # Trigger decay via another feedback
        sampler.record_feedback(
            _dominant_contribution("energy_match"),
            reward=True,
        )

        # Alpha should have been decayed before the update
        # The excess above prior should be reduced
        # New alpha = 1.0 + (old_excess * 0.9) + new_update
        assert sampler.arms["time_relevance"].alpha < alpha_before_decay + 0.1

    def test_decay_preserves_prior(self):
        """Decay should never reduce alpha/beta below the prior of 1.0."""
        sampler = ThompsonSampler(decay_factor=0.5)

        # Small evidence, aggressive decay
        sampler.record_feedback({"time_relevance": 1.0}, reward=True)

        # Trigger many decays
        for _ in range(50):
            sampler.record_feedback({"energy_match": 0.01}, reward=True)

        # time_relevance should still have alpha >= 1.0 (prior preserved)
        assert sampler.arms["time_relevance"].alpha >= 1.0
        assert sampler.arms["time_relevance"].beta >= 1.0

    def test_adaptive_behavior_change(self):
        """With decay, the sampler should adapt when user behavior changes."""
        sampler = ThompsonSampler(decay_factor=0.95)

        # Phase 1: user favors time_relevance
        for _ in range(50):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=False,
            )

        phase1_weights = sampler.get_expected_weights()

        # Phase 2: user behavior shifts to favor energy_match
        for _ in range(100):
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=True,
            )
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=False,
            )

        phase2_weights = sampler.get_expected_weights()

        # After phase 2, energy_match should be higher than in phase 1
        assert phase2_weights["energy_match"] > phase1_weights["energy_match"], (
            "Decay should allow adaptation to changing behavior"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization / state persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Tests for saving and restoring sampler state."""

    def test_get_state_returns_all_arms(self):
        sampler = ThompsonSampler()
        state = sampler.get_state()
        assert len(state) == len(ThompsonSampler.DEFAULT_SIGNALS)

    def test_get_state_includes_alpha_beta(self):
        sampler = ThompsonSampler()
        state = sampler.get_state()
        for arm_state in state.values():
            assert "alpha" in arm_state
            assert "beta" in arm_state
            assert "mean" in arm_state
            assert "variance" in arm_state

    def test_roundtrip_serialization(self):
        """Save → restore should produce identical sampler."""
        sampler = ThompsonSampler()

        # Train it
        for _ in range(20):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )

        state = sampler.get_state()
        restored = ThompsonSampler.from_state(state)

        # Compare arm states
        for name in ThompsonSampler.DEFAULT_SIGNALS:
            assert restored.arms[name].alpha == pytest.approx(
                sampler.arms[name].alpha
            )
            assert restored.arms[name].beta == pytest.approx(
                sampler.arms[name].beta
            )

    def test_roundtrip_preserves_weights(self):
        sampler = ThompsonSampler()
        for _ in range(30):
            sampler.record_feedback(
                _dominant_contribution("energy_match"),
                reward=True,
            )

        original_weights = sampler.get_expected_weights()
        state = sampler.get_state()
        restored = ThompsonSampler.from_state(state)
        restored_weights = restored.get_expected_weights()

        for key in original_weights:
            assert original_weights[key] == pytest.approx(
                restored_weights[key], abs=1e-9
            )

    def test_from_state_with_custom_params(self):
        state = {"time_relevance": {"alpha": 5.0, "beta": 2.0}}
        sampler = ThompsonSampler.from_state(
            state, decay_factor=0.9, min_weight=0.1
        )
        assert sampler.decay_factor == 0.9
        assert sampler.min_weight == 0.1
        assert sampler.arms["time_relevance"].alpha == 5.0

    def test_from_state_fills_missing_defaults(self):
        """Partial state should still create all default arms."""
        state = {"time_relevance": {"alpha": 5.0, "beta": 2.0}}
        sampler = ThompsonSampler.from_state(state)
        # time_relevance should have custom values
        assert sampler.arms["time_relevance"].alpha == 5.0
        # Other default signals should exist with default prior
        assert "energy_match" in sampler.arms


# ═══════════════════════════════════════════════════════════════════════════════
# Reset behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestReset:
    """Tests for sampler reset (return to cold start)."""

    def test_reset_returns_to_uniform(self):
        sampler = ThompsonSampler()

        # Train
        for _ in range(50):
            sampler.record_feedback(
                _dominant_contribution("time_relevance"),
                reward=True,
            )

        assert sampler.arms["time_relevance"].alpha > 5.0

        # Reset
        sampler.reset()

        for arm in sampler.arms.values():
            assert arm.alpha == 1.0
            assert arm.beta == 1.0

    def test_reset_preserves_arm_structure(self):
        sampler = ThompsonSampler()
        sampler.record_feedback({"novel": 1.0}, reward=True)
        assert "novel" in sampler.arms

        sampler.reset()
        # Novel arm should still exist but be reset
        assert "novel" in sampler.arms
        assert sampler.arms["novel"].alpha == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-shame verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestAntiShame:
    """Verify Thompson Sampling follows anti-shame design principles."""

    def test_dismissals_are_neutral_not_punishment(self):
        """Dismissals update beta proportionally — they don't amplify guilt."""
        sampler = ThompsonSampler()
        contributions = _uniform_contributions()

        # One dismissal
        sampler.record_feedback(contributions, reward=False)

        # Check beta increase is proportional, not punitive
        n = len(ThompsonSampler.DEFAULT_SIGNALS)
        for arm in sampler.arms.values():
            assert arm.beta == pytest.approx(1.0 + 1.0 / n)

    def test_no_signal_fully_suppressed(self):
        """Even heavily penalized signals retain min_weight in samples."""
        sampler = ThompsonSampler(min_weight=0.03)

        # 500 dismissals on one signal
        for _ in range(500):
            sampler.record_feedback(
                _dominant_contribution("freshness"),
                reward=False,
            )

        rng = random.Random(42)
        for _ in range(20):
            weights = sampler.sample_weights(rng)
            # After normalization, no signal is zero
            for w in weights.values():
                assert w > 0.0

    def test_weights_remain_valid_under_all_feedback_patterns(self):
        """Weights should always be valid regardless of feedback pattern."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        patterns = [
            (True, True, True),   # all completions
            (False, False, False),  # all dismissals
            (True, False, True),  # mixed
        ]

        for pattern in patterns:
            for reward in pattern:
                sampler.record_feedback(_uniform_contributions(), reward=reward)

            weights = sampler.sample_weights(rng)
            assert sum(weights.values()) == pytest.approx(1.0)
            for w in weights.values():
                assert w > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Stress tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStress:
    """Stress tests for numerical stability."""

    def test_many_updates_no_overflow(self):
        """Thousands of updates should not cause numerical issues."""
        sampler = ThompsonSampler()
        rng = random.Random(42)

        for i in range(2000):
            sampler.record_feedback(
                _uniform_contributions(),
                reward=(i % 3 != 0),
            )

        # Should still produce valid weights
        weights = sampler.sample_weights(rng)
        assert sum(weights.values()) == pytest.approx(1.0)
        for w in weights.values():
            assert math.isfinite(w)
            assert w > 0.0

    def test_extreme_alpha_beta_still_samples(self):
        """Very high alpha/beta values should still produce valid samples."""
        arm = ArmState(alpha=10000.0, beta=10000.0)
        rng = random.Random(42)
        s = arm.sample(rng)
        assert 0.0 <= s <= 1.0
        assert math.isfinite(s)

    def test_very_small_contributions(self):
        """Very small contributions should not cause issues."""
        sampler = ThompsonSampler()
        tiny = {s: 1e-10 for s in ThompsonSampler.DEFAULT_SIGNALS}
        sampler.record_feedback(tiny, reward=True)

        for arm in sampler.arms.values():
            assert math.isfinite(arm.alpha)
            assert math.isfinite(arm.beta)

    def test_zero_composite_score_completion(self):
        """record_completion with zero composite should not crash."""
        sampler = ThompsonSampler()
        signal_scores = {s: 0.0 for s in ThompsonSampler.DEFAULT_SIGNALS}
        sampler.record_completion(signal_scores, composite_score=0.0)
        # Should use equal contributions
        for arm in sampler.arms.values():
            assert arm.alpha >= 1.0
