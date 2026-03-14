"""Tests for the Thompson Sampling engine.

Validates Beta distribution tracking, sampling, feedback updates,
temporal decay, serialization, and anti-shame design principles.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.thompson_sampling import (
    BetaParams,
    DecayConfig,
    DEFAULT_CATEGORIES,
    FeedbackType,
    SamplingResult,
    ThompsonSamplingEngine,
)


# ── BetaParams unit tests ──────────────────────────────────────────


class TestBetaParams:
    """Tests for BetaParams dataclass."""

    def test_default_uniform_prior(self) -> None:
        """Default Beta(1,1) is uniform — mean should be 0.5."""
        p = BetaParams()
        assert p.alpha == 1.0
        assert p.beta == 1.0
        assert p.mean == pytest.approx(0.5)

    def test_mean_calculation(self) -> None:
        """Mean = alpha / (alpha + beta)."""
        p = BetaParams(alpha=3.0, beta=1.0)
        assert p.mean == pytest.approx(0.75)

        p2 = BetaParams(alpha=1.0, beta=4.0)
        assert p2.mean == pytest.approx(0.2)

    def test_variance_calculation(self) -> None:
        """Variance = ab / ((a+b)^2 * (a+b+1))."""
        p = BetaParams(alpha=2.0, beta=2.0)
        expected = (2.0 * 2.0) / (16.0 * 5.0)
        assert p.variance == pytest.approx(expected)

    def test_confidence_increases_with_observations(self) -> None:
        """Confidence should increase with more observations."""
        p0 = BetaParams(total_observations=0)
        p5 = BetaParams(total_observations=5)
        p50 = BetaParams(total_observations=50)
        p500 = BetaParams(total_observations=500)

        assert p0.confidence == 0.0
        assert p5.confidence > p0.confidence
        assert p50.confidence > p5.confidence
        assert p500.confidence > p50.confidence
        assert p500.confidence < 1.0  # Never reaches 1.0 exactly

    def test_sample_returns_value_in_range(self) -> None:
        """Sample should return a value in [0, 1]."""
        import random
        rng = random.Random(42)
        p = BetaParams(alpha=2.0, beta=3.0)

        for _ in range(100):
            s = p.sample(rng)
            assert 0.0 <= s <= 1.0

    def test_sample_reproducibility_with_seed(self) -> None:
        """Same seed should produce same samples."""
        import random
        p = BetaParams(alpha=5.0, beta=2.0)

        rng1 = random.Random(123)
        samples1 = [p.sample(rng1) for _ in range(10)]

        rng2 = random.Random(123)
        samples2 = [p.sample(rng2) for _ in range(10)]

        assert samples1 == samples2

    def test_high_alpha_samples_high(self) -> None:
        """Beta with high alpha should mostly sample high values."""
        import random
        rng = random.Random(42)
        p = BetaParams(alpha=50.0, beta=2.0)

        samples = [p.sample(rng) for _ in range(100)]
        avg = sum(samples) / len(samples)
        assert avg > 0.8  # Should be close to mean of ~0.96

    def test_serialization_roundtrip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        p = BetaParams(alpha=3.5, beta=2.1, total_observations=17)
        d = p.to_dict()
        p2 = BetaParams.from_dict(d)

        assert p2.alpha == pytest.approx(p.alpha)
        assert p2.beta == pytest.approx(p.beta)
        assert p2.total_observations == p.total_observations

    def test_zero_params_mean_returns_half(self) -> None:
        """Edge case: both alpha and beta are 0 → mean should be 0.5."""
        p = BetaParams(alpha=0.0, beta=0.0)
        assert p.mean == 0.5


# ── ThompsonSamplingEngine tests ────────────────────────────────────


class TestThompsonSamplingEngine:
    """Tests for the Thompson Sampling engine."""

    def test_default_categories(self) -> None:
        """Engine initializes with all 7 Blurt intent categories."""
        engine = ThompsonSamplingEngine(seed=42)
        assert set(engine.categories) == set(DEFAULT_CATEGORIES)
        assert len(engine.categories) == 7

    def test_custom_categories(self) -> None:
        """Engine can be initialized with custom categories."""
        cats = ["work", "personal", "health"]
        engine = ThompsonSamplingEngine(categories=cats, seed=42)
        assert engine.categories == cats

    def test_initial_params_are_uniform(self) -> None:
        """All categories start with uniform Beta(1,1) prior."""
        engine = ThompsonSamplingEngine(seed=42)
        for cat in engine.categories:
            p = engine.get_params(cat)
            assert p.alpha == 1.0
            assert p.beta == 1.0
            assert p.mean == pytest.approx(0.5)

    def test_sample_returns_valid_result(self) -> None:
        """Sampling should return a valid SamplingResult."""
        engine = ThompsonSamplingEngine(seed=42)
        result = engine.sample(apply_decay=False)

        assert isinstance(result, SamplingResult)
        assert result.category in engine.categories
        assert 0.0 <= result.sampled_value <= 1.0
        assert len(result.all_samples) == 7
        assert all(0.0 <= v <= 1.0 for v in result.all_samples.values())

    def test_sample_selects_highest(self) -> None:
        """Sampling should select the category with the highest draw."""
        engine = ThompsonSamplingEngine(seed=42)
        result = engine.sample(apply_decay=False)

        max_cat = max(result.all_samples, key=lambda c: result.all_samples[c])
        assert result.category == max_cat

    def test_sample_with_eligible_subset(self) -> None:
        """Sampling with eligible_categories filters correctly."""
        engine = ThompsonSamplingEngine(seed=42)
        eligible = ["task", "event"]
        result = engine.sample(eligible_categories=eligible, apply_decay=False)

        assert result.category in eligible
        assert set(result.all_samples.keys()) == set(eligible)

    def test_sample_top_k(self) -> None:
        """sample_top_k returns k results sorted by sample value."""
        engine = ThompsonSamplingEngine(seed=42)
        results = engine.sample_top_k(k=3, apply_decay=False)

        assert len(results) == 3
        # Should be sorted descending by sampled_value
        values = [r.sampled_value for r in results]
        assert values == sorted(values, reverse=True)

    def test_sample_top_k_exceeds_categories(self) -> None:
        """sample_top_k with k > categories returns all categories."""
        cats = ["a", "b"]
        engine = ThompsonSamplingEngine(categories=cats, seed=42)
        results = engine.sample_top_k(k=5, apply_decay=False)
        assert len(results) == 2

    def test_sample_reproducibility(self) -> None:
        """Same seed produces same sampling results."""
        engine1 = ThompsonSamplingEngine(seed=42)
        engine2 = ThompsonSamplingEngine(seed=42)

        r1 = engine1.sample(apply_decay=False)
        r2 = engine2.sample(apply_decay=False)

        assert r1.category == r2.category
        assert r1.sampled_value == pytest.approx(r2.sampled_value)


class TestFeedbackUpdates:
    """Tests for feedback-driven parameter updates."""

    def test_accepted_increases_alpha(self) -> None:
        """Accepted feedback should increase alpha."""
        engine = ThompsonSamplingEngine(seed=42)
        before = engine.get_params("task").alpha

        engine.update("task", FeedbackType.ACCEPTED)

        after = engine.get_params("task").alpha
        assert after > before

    def test_completed_increases_alpha_more(self) -> None:
        """Completed feedback should increase alpha more than accepted."""
        engine1 = ThompsonSamplingEngine(seed=42)
        engine2 = ThompsonSamplingEngine(seed=42)

        engine1.update("task", FeedbackType.ACCEPTED)
        engine2.update("task", FeedbackType.COMPLETED)

        assert engine2.get_params("task").alpha > engine1.get_params("task").alpha

    def test_dismissed_increases_beta(self) -> None:
        """Dismissed feedback should increase beta."""
        engine = ThompsonSamplingEngine(seed=42)
        before = engine.get_params("task").beta

        engine.update("task", FeedbackType.DISMISSED)

        after = engine.get_params("task").beta
        assert after > before

    def test_snoozed_updates_both(self) -> None:
        """Snoozed feedback should increase both alpha and beta (partial positive)."""
        engine = ThompsonSamplingEngine(seed=42)
        alpha_before = engine.get_params("task").alpha
        beta_before = engine.get_params("task").beta

        engine.update("task", FeedbackType.SNOOZED)

        p = engine.get_params("task")
        assert p.alpha > alpha_before  # Partial positive
        assert p.beta > beta_before    # Slight negative (not now)

    def test_ignored_minimal_beta_increase(self) -> None:
        """Ignored feedback should have minimal beta increase."""
        engine = ThompsonSamplingEngine(seed=42)
        beta_before = engine.get_params("task").beta

        engine.update("task", FeedbackType.IGNORED)

        beta_after = engine.get_params("task").beta
        beta_delta = beta_after - beta_before
        assert 0 < beta_delta <= 0.15  # Very small

    def test_observation_count_increments(self) -> None:
        """Each update should increment total_observations."""
        engine = ThompsonSamplingEngine(seed=42)

        for i in range(5):
            engine.update("task", FeedbackType.ACCEPTED)

        assert engine.get_params("task").total_observations == 5

    def test_unknown_category_raises(self) -> None:
        """Updating an unknown category should raise KeyError."""
        engine = ThompsonSamplingEngine(seed=42)

        with pytest.raises(KeyError, match="Unknown category"):
            engine.update("nonexistent", FeedbackType.ACCEPTED)

    def test_weight_multiplier(self) -> None:
        """Weight multiplier should scale the update."""
        engine1 = ThompsonSamplingEngine(seed=42)
        engine2 = ThompsonSamplingEngine(seed=42)

        engine1.update("task", FeedbackType.ACCEPTED, weight_multiplier=1.0)
        engine2.update("task", FeedbackType.ACCEPTED, weight_multiplier=2.0)

        # engine2 should have larger alpha increase
        a1 = engine1.get_params("task").alpha
        a2 = engine2.get_params("task").alpha
        assert a2 > a1

    def test_batch_update(self) -> None:
        """Batch update should apply all updates."""
        engine = ThompsonSamplingEngine(seed=42)

        updates = [
            ("task", FeedbackType.ACCEPTED),
            ("task", FeedbackType.ACCEPTED),
            ("event", FeedbackType.DISMISSED),
        ]
        results = engine.batch_update(updates)

        assert "task" in results
        assert "event" in results
        assert engine.get_params("task").total_observations == 2
        assert engine.get_params("event").total_observations == 1

    def test_feedback_history_recorded(self) -> None:
        """Feedback history should be recorded."""
        engine = ThompsonSamplingEngine(seed=42)

        engine.update("task", FeedbackType.ACCEPTED)
        engine.update("event", FeedbackType.DISMISSED)

        history = engine.feedback_history
        assert len(history) == 2
        assert history[0]["category"] == "task"
        assert history[0]["feedback"] == "accepted"
        assert history[1]["category"] == "event"
        assert history[1]["feedback"] == "dismissed"

    def test_anti_shame_dismissed_weight_less_than_accepted(self) -> None:
        """Anti-shame: dismissed has less impact than accepted.

        This ensures the model doesn't over-penalize the user for
        not engaging with surfaced tasks.
        """
        engine = ThompsonSamplingEngine(seed=42)
        fw = engine._feedback_weights

        # Dismissed beta increase should be less than accepted alpha increase
        assert fw.dismissed_beta < fw.accepted_alpha
        # Ignored should be even less
        assert fw.ignored_beta < fw.dismissed_beta


class TestTemporalDecay:
    """Tests for temporal decay of Beta parameters."""

    def test_no_decay_when_recent(self) -> None:
        """Decay should not apply when last update was recent."""
        config = DecayConfig(
            half_life_hours=168.0,
            decay_interval_hours=1.0,
        )
        engine = ThompsonSamplingEngine(
            categories=["task"],
            decay_config=config,
            seed=42,
        )

        # Set some non-default params
        engine._params["task"].alpha = 10.0
        engine._params["task"].beta = 3.0
        engine._params["task"].last_updated = datetime.now(timezone.utc)

        # Apply decay immediately — should not change
        engine.apply_decay_all()

        p = engine.get_params("task")
        assert p.alpha == pytest.approx(10.0)
        assert p.beta == pytest.approx(3.0)

    def test_decay_after_half_life(self) -> None:
        """After one half-life, parameters should be halfway to prior."""
        config = DecayConfig(
            half_life_hours=24.0,
            min_alpha=1.0,
            min_beta=1.0,
            decay_interval_hours=0.0,  # Allow immediate decay
        )
        engine = ThompsonSamplingEngine(
            categories=["task"],
            decay_config=config,
            seed=42,
        )

        # Set params and backdate
        engine._params["task"].alpha = 11.0  # 10 above prior
        engine._params["task"].beta = 5.0    # 4 above prior
        past = datetime.now(timezone.utc) - timedelta(hours=24)
        engine._params["task"].last_updated = past

        # Apply decay — should be halfway to prior
        now = datetime.now(timezone.utc)
        engine.apply_decay_all(now)

        p = engine.get_params("task")
        # After one half-life: prior + (old - prior) * 0.5
        # alpha: 1.0 + (11.0 - 1.0) * 0.5 = 6.0
        # beta: 1.0 + (5.0 - 1.0) * 0.5 = 3.0
        assert p.alpha == pytest.approx(6.0, abs=0.1)
        assert p.beta == pytest.approx(3.0, abs=0.1)

    def test_decay_preserves_prior_minimum(self) -> None:
        """Decay should never reduce parameters below the prior."""
        config = DecayConfig(
            half_life_hours=1.0,  # Very fast decay
            min_alpha=1.0,
            min_beta=1.0,
            decay_interval_hours=0.0,
        )
        engine = ThompsonSamplingEngine(
            categories=["task"],
            decay_config=config,
            seed=42,
        )

        engine._params["task"].alpha = 2.0
        engine._params["task"].beta = 2.0
        far_past = datetime.now(timezone.utc) - timedelta(hours=1000)
        engine._params["task"].last_updated = far_past

        engine.apply_decay_all()

        p = engine.get_params("task")
        assert p.alpha >= config.min_alpha
        assert p.beta >= config.min_beta

    def test_decay_rate_proportional_to_elapsed(self) -> None:
        """More elapsed time should produce more decay."""
        config = DecayConfig(
            half_life_hours=48.0,
            decay_interval_hours=0.0,
        )

        now = datetime.now(timezone.utc)

        # Engine with 12h elapsed
        e1 = ThompsonSamplingEngine(
            categories=["task"], decay_config=config, seed=42
        )
        e1._params["task"].alpha = 10.0
        e1._params["task"].last_updated = now - timedelta(hours=12)
        e1.apply_decay_all(now)

        # Engine with 48h elapsed
        e2 = ThompsonSamplingEngine(
            categories=["task"], decay_config=config, seed=42
        )
        e2._params["task"].alpha = 10.0
        e2._params["task"].last_updated = now - timedelta(hours=48)
        e2.apply_decay_all(now)

        # More time = more decay (closer to prior of 1.0)
        assert e2.get_params("task").alpha < e1.get_params("task").alpha

    def test_decay_applied_during_sampling(self) -> None:
        """Decay should be applied during sample() when apply_decay=True."""
        config = DecayConfig(
            half_life_hours=1.0,
            decay_interval_hours=0.0,
        )
        engine = ThompsonSamplingEngine(
            categories=["task"],
            decay_config=config,
            seed=42,
        )

        engine._params["task"].alpha = 100.0
        engine._params["task"].beta = 1.0
        past = datetime.now(timezone.utc) - timedelta(hours=100)
        engine._params["task"].last_updated = past

        # Sample with decay
        engine.sample(apply_decay=True)

        # Alpha should have decayed significantly
        assert engine.get_params("task").alpha < 100.0


class TestEngineOperations:
    """Tests for engine management operations."""

    def test_add_category(self) -> None:
        """Adding a new category should work."""
        engine = ThompsonSamplingEngine(seed=42)
        engine.add_category("custom")

        assert "custom" in engine.categories
        p = engine.get_params("custom")
        assert p.alpha == 1.0
        assert p.beta == 1.0

    def test_add_category_with_params(self) -> None:
        """Adding a category with custom params should preserve them."""
        engine = ThompsonSamplingEngine(seed=42)
        engine.add_category("custom", BetaParams(alpha=5.0, beta=2.0))

        p = engine.get_params("custom")
        assert p.alpha == 5.0
        assert p.beta == 2.0

    def test_reset_category(self) -> None:
        """Resetting a category should restore uniform prior."""
        engine = ThompsonSamplingEngine(seed=42)
        engine.update("task", FeedbackType.ACCEPTED)
        engine.update("task", FeedbackType.ACCEPTED)

        engine.reset_category("task")

        p = engine.get_params("task")
        assert p.alpha == 1.0
        assert p.beta == 1.0
        assert p.total_observations == 0

    def test_reset_all(self) -> None:
        """Resetting all should restore all categories and clear history."""
        engine = ThompsonSamplingEngine(seed=42)
        engine.update("task", FeedbackType.ACCEPTED)
        engine.update("event", FeedbackType.DISMISSED)

        engine.reset_all()

        for cat in engine.categories:
            p = engine.get_params(cat)
            assert p.alpha == 1.0
            assert p.beta == 1.0
        assert len(engine.feedback_history) == 0

    def test_get_category_rankings(self) -> None:
        """Rankings should be sorted by mean descending."""
        engine = ThompsonSamplingEngine(seed=42)

        # Make "task" the best category
        for _ in range(10):
            engine.update("task", FeedbackType.ACCEPTED)

        # Make "event" moderately good
        for _ in range(5):
            engine.update("event", FeedbackType.ACCEPTED)

        rankings = engine.get_category_rankings()

        assert rankings[0][0] == "task"
        assert rankings[1][0] == "event"
        # All values should be in [0, 1]
        assert all(0.0 <= mean <= 1.0 for _, mean in rankings)

    def test_get_category_stats(self) -> None:
        """Stats should include all expected fields."""
        engine = ThompsonSamplingEngine(seed=42)
        engine.update("task", FeedbackType.ACCEPTED)

        stats = engine.get_category_stats()

        assert "task" in stats
        task_stats = stats["task"]
        assert "alpha" in task_stats
        assert "beta" in task_stats
        assert "mean" in task_stats
        assert "variance" in task_stats
        assert "confidence" in task_stats
        assert "total_observations" in task_stats

    def test_unknown_category_get_raises(self) -> None:
        """Getting an unknown category should raise KeyError."""
        engine = ThompsonSamplingEngine(seed=42)
        with pytest.raises(KeyError):
            engine.get_params("nonexistent")

    def test_reset_unknown_category_raises(self) -> None:
        """Resetting an unknown category should raise KeyError."""
        engine = ThompsonSamplingEngine(seed=42)
        with pytest.raises(KeyError):
            engine.reset_category("nonexistent")


class TestSerialization:
    """Tests for engine serialization/deserialization."""

    def test_roundtrip(self) -> None:
        """Engine state should survive serialization roundtrip."""
        engine = ThompsonSamplingEngine(seed=42)
        engine.update("task", FeedbackType.ACCEPTED)
        engine.update("task", FeedbackType.COMPLETED)
        engine.update("event", FeedbackType.DISMISSED)

        data = engine.to_dict()
        restored = ThompsonSamplingEngine.from_dict(data, seed=42)

        # Check categories preserved
        assert set(restored.categories) == set(engine.categories)

        # Check params preserved
        for cat in engine.categories:
            orig = engine.get_params(cat)
            rest = restored.get_params(cat)
            assert rest.alpha == pytest.approx(orig.alpha)
            assert rest.beta == pytest.approx(orig.beta)
            assert rest.total_observations == orig.total_observations

    def test_from_dict_with_extra_category(self) -> None:
        """from_dict should handle categories in params but not in category list."""
        data = {
            "categories": ["task"],
            "params": {
                "task": {"alpha": 3.0, "beta": 2.0, "total_observations": 5},
                "custom": {"alpha": 1.5, "beta": 1.5, "total_observations": 2},
            },
        }
        engine = ThompsonSamplingEngine.from_dict(data)

        # Both categories should exist
        assert "task" in engine.categories
        assert "custom" in engine.categories

    def test_from_dict_defaults(self) -> None:
        """from_dict with empty dict should use all defaults."""
        engine = ThompsonSamplingEngine.from_dict({})
        assert set(engine.categories) == set(DEFAULT_CATEGORIES)


class TestLearningBehavior:
    """Integration-level tests for learning behavior over time."""

    def test_repeated_acceptance_shifts_distribution(self) -> None:
        """Repeatedly accepting a category should shift its mean upward."""
        engine = ThompsonSamplingEngine(seed=42)

        initial_mean = engine.get_params("task").mean
        assert initial_mean == pytest.approx(0.5)

        for _ in range(20):
            engine.update("task", FeedbackType.ACCEPTED)

        final_mean = engine.get_params("task").mean
        assert final_mean > 0.8  # Should be significantly higher

    def test_repeated_dismissal_shifts_distribution(self) -> None:
        """Repeatedly dismissing a category should shift its mean downward."""
        engine = ThompsonSamplingEngine(seed=42)

        for _ in range(20):
            engine.update("idea", FeedbackType.DISMISSED)

        mean = engine.get_params("idea").mean
        assert mean < 0.3  # Should be lower

    def test_preferred_category_sampled_more_often(self) -> None:
        """A category with many acceptances should be sampled more often."""
        engine = ThompsonSamplingEngine(seed=42)

        # Strongly prefer "task"
        for _ in range(50):
            engine.update("task", FeedbackType.ACCEPTED)

        # Sample many times and count
        counts: dict[str, int] = {c: 0 for c in engine.categories}
        for _ in range(200):
            result = engine.sample(apply_decay=False)
            counts[result.category] += 1

        # "task" should be selected most often
        assert counts["task"] == max(counts.values())
        assert counts["task"] > 100  # Should win majority of the time

    def test_exploration_still_happens(self) -> None:
        """Even with a strong preference, other categories should still be sampled.

        This is the key property of Thompson Sampling: exploration happens
        naturally through variance in the sampling distribution.
        """
        engine = ThompsonSamplingEngine(seed=42)

        # Strongly prefer "task"
        for _ in range(20):
            engine.update("task", FeedbackType.ACCEPTED)

        # Sample many times
        sampled_categories: set[str] = set()
        for _ in range(500):
            result = engine.sample(apply_decay=False)
            sampled_categories.add(result.category)

        # Should have explored at least some other categories
        assert len(sampled_categories) > 1

    def test_decay_enables_recovery(self) -> None:
        """After heavy dismissals and time passing, a category should recover.

        Decay ensures old negative signals don't permanently suppress a
        category — anti-shame principle.
        """
        config = DecayConfig(
            half_life_hours=24.0,
            decay_interval_hours=0.0,
        )
        engine = ThompsonSamplingEngine(
            categories=["task"],
            decay_config=config,
            seed=42,
        )

        # Heavy dismissal
        for _ in range(50):
            engine.update("task", FeedbackType.DISMISSED)

        suppressed_mean = engine.get_params("task").mean
        assert suppressed_mean < 0.15  # Should be quite low

        # Simulate 2 weeks passing
        far_future = datetime.now(timezone.utc) + timedelta(days=14)
        engine._params["task"].last_updated = (
            datetime.now(timezone.utc) - timedelta(days=14)
        )
        engine.apply_decay_all(far_future)

        recovered_mean = engine.get_params("task").mean
        # Should have recovered toward 0.5 (the prior)
        assert recovered_mean > suppressed_mean
        assert recovered_mean > 0.4  # Mostly recovered after 14 half-lives
