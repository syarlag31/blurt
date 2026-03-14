"""Tests for the per-user Beta distribution preference persistence layer.

Validates:
- Per-user isolation of Beta parameters
- Temporal decay for non-stationary preferences
- Weight convergence under consistent feedback
- Adaptation when user preferences shift
- Serialization and import/export roundtrips
- Anti-shame design principles
- Multi-user concurrent access patterns
- Cold start behavior
- Batch feedback operations

AC 11 Sub-AC 4: Persistence layer for per-user Beta distribution parameters.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.preference_store import (
    InMemoryPreferenceBackend,
    UserPreferenceSnapshot,
    UserPreferenceStore,
)
from blurt.services.thompson_sampling import (
    BetaParams,
    DecayConfig,
    DEFAULT_CATEGORIES,
    FeedbackType,
    FeedbackWeights,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def store() -> UserPreferenceStore:
    """A fresh preference store with default config."""
    return UserPreferenceStore()


@pytest.fixture
def fast_decay_store() -> UserPreferenceStore:
    """A store with fast decay for testing adaptation."""
    return UserPreferenceStore(
        decay_config=DecayConfig(
            half_life_hours=24.0,
            min_alpha=1.0,
            min_beta=1.0,
            decay_interval_hours=0.0,  # Allow immediate decay
        ),
    )


@pytest.fixture
def no_decay_store() -> UserPreferenceStore:
    """A store with no decay for testing pure convergence."""
    return UserPreferenceStore(
        decay_config=DecayConfig(
            half_life_hours=1e12,  # Effectively no decay
            decay_interval_hours=1e12,
        ),
    )


# ── Cold start & initialization ──────────────────────────────────────


class TestColdStart:
    """Tests for new user initialization."""

    def test_new_user_gets_uniform_priors(self, store: UserPreferenceStore) -> None:
        """New users should start with Beta(1,1) uniform prior for all categories."""
        params = store.get_all_params("new-user", apply_decay=False)

        assert len(params) == 7  # All 7 Blurt intents
        for cat, p in params.items():
            assert p.alpha == 1.0, f"{cat} alpha should be 1.0"
            assert p.beta == 1.0, f"{cat} beta should be 1.0"
            assert p.mean == pytest.approx(0.5), f"{cat} mean should be 0.5"

    def test_new_user_all_categories_present(self, store: UserPreferenceStore) -> None:
        """New users should have all 7 default categories."""
        params = store.get_all_params("new-user", apply_decay=False)
        assert set(params.keys()) == set(DEFAULT_CATEGORIES)

    def test_new_user_snapshot_metadata(self, store: UserPreferenceStore) -> None:
        """Snapshot metadata should be initialized correctly."""
        snapshot = store.get_snapshot("new-user", apply_decay=False)

        assert snapshot.user_id == "new-user"
        assert snapshot.total_feedback_count == 0
        assert snapshot.version == 1

    def test_custom_categories(self) -> None:
        """Store with custom categories should initialize those for new users."""
        custom = ["work", "health", "creative"]
        store = UserPreferenceStore(categories=custom)
        params = store.get_all_params("user-1", apply_decay=False)

        assert set(params.keys()) == set(custom)

    def test_get_unknown_category_raises(self, store: UserPreferenceStore) -> None:
        """Requesting an untracked category should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown category"):
            store.get_params("user-1", "nonexistent")


# ── Per-user isolation ───────────────────────────────────────────────


class TestUserIsolation:
    """Tests for multi-user parameter isolation."""

    def test_different_users_independent(self, store: UserPreferenceStore) -> None:
        """Feedback for one user should not affect another."""
        store.record_feedback("alice", "task", FeedbackType.ACCEPTED)
        store.record_feedback("alice", "task", FeedbackType.ACCEPTED)

        alice_params = store.get_params("alice", "task", apply_decay=False)
        bob_params = store.get_params("bob", "task", apply_decay=False)

        assert alice_params.alpha > bob_params.alpha
        assert bob_params.alpha == 1.0  # Bob is unchanged

    def test_many_users_isolated(self, store: UserPreferenceStore) -> None:
        """Multiple users should maintain fully independent states."""
        users = [f"user-{i}" for i in range(10)]

        # Each user accepts a different category
        for i, user in enumerate(users):
            cat = DEFAULT_CATEGORIES[i % len(DEFAULT_CATEGORIES)]
            for _ in range(5):
                store.record_feedback(user, cat, FeedbackType.ACCEPTED)

        # Verify isolation
        for i, user in enumerate(users):
            cat = DEFAULT_CATEGORIES[i % len(DEFAULT_CATEGORIES)]
            p = store.get_params(user, cat, apply_decay=False)
            assert p.total_observations == 5

            # Other categories should be untouched
            other_cat = DEFAULT_CATEGORIES[(i + 1) % len(DEFAULT_CATEGORIES)]
            p_other = store.get_params(user, other_cat, apply_decay=False)
            assert p_other.total_observations == 0

    def test_user_deletion_is_isolated(self, store: UserPreferenceStore) -> None:
        """Deleting one user should not affect others."""
        store.record_feedback("alice", "task", FeedbackType.ACCEPTED)
        store.record_feedback("bob", "task", FeedbackType.ACCEPTED)

        store.delete_user("alice")

        # Bob should be unaffected
        bob_params = store.get_params("bob", "task", apply_decay=False)
        assert bob_params.alpha > 1.0

        # Alice should be reset to cold start on next access
        alice_params = store.get_params("alice", "task", apply_decay=False)
        assert alice_params.alpha == 1.0

    def test_list_users(self, store: UserPreferenceStore) -> None:
        """list_users should return all users with stored preferences."""
        store.record_feedback("alice", "task", FeedbackType.ACCEPTED)
        store.record_feedback("bob", "event", FeedbackType.COMPLETED)
        store.record_feedback("charlie", "idea", FeedbackType.DISMISSED)

        users = store.list_users()
        assert set(users) == {"alice", "bob", "charlie"}


# ── Feedback recording ───────────────────────────────────────────────


class TestFeedbackRecording:
    """Tests for feedback-driven parameter updates."""

    def test_accepted_increases_alpha(self, store: UserPreferenceStore) -> None:
        """Accepted feedback should increase alpha."""
        before = store.get_params("user-1", "task", apply_decay=False)
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)
        after = store.get_params("user-1", "task", apply_decay=False)

        assert after.alpha > before.alpha
        assert after.beta == before.beta  # Beta unchanged

    def test_completed_increases_alpha_more(self, store: UserPreferenceStore) -> None:
        """Completed should increase alpha more than accepted."""
        store.record_feedback("user-a", "task", FeedbackType.ACCEPTED)
        store.record_feedback("user-b", "task", FeedbackType.COMPLETED)

        a = store.get_params("user-a", "task", apply_decay=False)
        b = store.get_params("user-b", "task", apply_decay=False)
        assert b.alpha > a.alpha

    def test_dismissed_increases_beta(self, store: UserPreferenceStore) -> None:
        """Dismissed feedback should increase beta."""
        store.record_feedback("user-1", "task", FeedbackType.DISMISSED)
        p = store.get_params("user-1", "task", apply_decay=False)

        assert p.beta > 1.0
        assert p.alpha == 1.0  # Alpha unchanged

    def test_snoozed_updates_both(self, store: UserPreferenceStore) -> None:
        """Snoozed should increase both alpha and beta."""
        store.record_feedback("user-1", "task", FeedbackType.SNOOZED)
        p = store.get_params("user-1", "task", apply_decay=False)

        assert p.alpha > 1.0
        assert p.beta > 1.0

    def test_ignored_minimal_beta(self, store: UserPreferenceStore) -> None:
        """Ignored feedback should have minimal beta increase."""
        store.record_feedback("user-1", "task", FeedbackType.IGNORED)
        p = store.get_params("user-1", "task", apply_decay=False)

        assert p.beta > 1.0
        assert p.beta - 1.0 < 0.15  # Very small increase

    def test_observation_count_increments(self, store: UserPreferenceStore) -> None:
        """Each feedback should increment total_observations."""
        for _ in range(5):
            store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)

        p = store.get_params("user-1", "task", apply_decay=False)
        assert p.total_observations == 5

    def test_total_feedback_count(self, store: UserPreferenceStore) -> None:
        """Total feedback count should track across all categories."""
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)
        store.record_feedback("user-1", "event", FeedbackType.DISMISSED)
        store.record_feedback("user-1", "idea", FeedbackType.COMPLETED)

        snapshot = store.get_snapshot("user-1", apply_decay=False)
        assert snapshot.total_feedback_count == 3

    def test_unknown_category_feedback_raises(
        self, store: UserPreferenceStore
    ) -> None:
        """Feedback for unknown category should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown category"):
            store.record_feedback("user-1", "nonexistent", FeedbackType.ACCEPTED)

    def test_weight_multiplier(self, store: UserPreferenceStore) -> None:
        """Weight multiplier should scale the update."""
        store.record_feedback(
            "user-a", "task", FeedbackType.ACCEPTED, weight_multiplier=1.0
        )
        store.record_feedback(
            "user-b", "task", FeedbackType.ACCEPTED, weight_multiplier=3.0
        )

        a = store.get_params("user-a", "task", apply_decay=False)
        b = store.get_params("user-b", "task", apply_decay=False)
        assert b.alpha > a.alpha


# ── Batch feedback ───────────────────────────────────────────────────


class TestBatchFeedback:
    """Tests for batch feedback operations."""

    def test_batch_updates_multiple_categories(
        self, store: UserPreferenceStore
    ) -> None:
        """Batch feedback should update all specified categories."""
        feedbacks = [
            ("task", FeedbackType.ACCEPTED),
            ("task", FeedbackType.ACCEPTED),
            ("event", FeedbackType.DISMISSED),
            ("idea", FeedbackType.COMPLETED),
        ]
        results = store.batch_feedback("user-1", feedbacks)

        assert "task" in results
        assert "event" in results
        assert "idea" in results

        # Task got 2 acceptances
        assert results["task"].total_observations == 2
        # Event got 1 dismissal
        assert results["event"].total_observations == 1

    def test_batch_single_save(self, store: UserPreferenceStore) -> None:
        """Batch should result in consistent total_feedback_count."""
        feedbacks = [
            ("task", FeedbackType.ACCEPTED),
            ("event", FeedbackType.ACCEPTED),
            ("idea", FeedbackType.ACCEPTED),
        ]
        store.batch_feedback("user-1", feedbacks)

        snapshot = store.get_snapshot("user-1", apply_decay=False)
        assert snapshot.total_feedback_count == 3

    def test_batch_unknown_category_raises(
        self, store: UserPreferenceStore
    ) -> None:
        """Batch with unknown category should raise KeyError."""
        with pytest.raises(KeyError):
            store.batch_feedback(
                "user-1",
                [("nonexistent", FeedbackType.ACCEPTED)],
            )


# ── Temporal decay ───────────────────────────────────────────────────


class TestTemporalDecay:
    """Tests for decay mechanism on non-stationary preferences."""

    def test_no_decay_when_recent(self, fast_decay_store: UserPreferenceStore) -> None:
        """Decay should not apply when parameters were just updated."""
        now = datetime.now(timezone.utc)
        fast_decay_store.record_feedback(
            "user-1", "task", FeedbackType.ACCEPTED, now=now
        )

        # Read immediately — no decay expected
        p = fast_decay_store.get_params("user-1", "task", now=now)
        assert p.alpha == pytest.approx(2.0)

    def test_decay_after_half_life(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """After one half-life, excess params should halve."""
        now = datetime.now(timezone.utc)

        # Give strong preference
        for _ in range(10):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED, now=now
            )

        # Read well after half-life (24h)
        future = now + timedelta(hours=24)
        p = fast_decay_store.get_params("user-1", "task", now=future)

        # Alpha was 11.0 (1.0 + 10*1.0), excess = 10.0
        # After 24h half-life: 1.0 + 10.0 * 0.5 = 6.0
        assert p.alpha == pytest.approx(6.0, abs=0.1)

    def test_decay_preserves_minimum(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Decay should never reduce params below the prior minimum."""
        now = datetime.now(timezone.utc)
        fast_decay_store.record_feedback(
            "user-1", "task", FeedbackType.ACCEPTED, now=now
        )

        # Very far future — massive decay
        far_future = now + timedelta(days=365)
        p = fast_decay_store.get_params("user-1", "task", now=far_future)

        assert p.alpha >= 1.0  # Never below prior
        assert p.beta >= 1.0

    def test_decay_proportional_to_time(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """More time should produce more decay."""
        now = datetime.now(timezone.utc)

        # Two users with identical feedback
        for uid in ["user-a", "user-b"]:
            for _ in range(10):
                fast_decay_store.record_feedback(
                    uid, "task", FeedbackType.ACCEPTED, now=now
                )

        # Read at different future times
        p_12h = fast_decay_store.get_params(
            "user-a", "task", now=now + timedelta(hours=12)
        )
        p_48h = fast_decay_store.get_params(
            "user-b", "task", now=now + timedelta(hours=48)
        )

        assert p_48h.alpha < p_12h.alpha  # More decay with more time

    def test_decay_enables_recovery(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Heavy dismissals should recover over time (anti-shame).

        This is a key anti-shame property: a category that was heavily
        dismissed should gradually return toward the neutral prior,
        allowing future engagement.
        """
        now = datetime.now(timezone.utc)

        # Heavy dismissal of "idea" category
        for _ in range(50):
            fast_decay_store.record_feedback(
                "user-1", "idea", FeedbackType.DISMISSED, now=now
            )

        # Read immediately — should be quite low
        suppressed = fast_decay_store.get_params(
            "user-1", "idea", apply_decay=False
        )
        assert suppressed.mean < 0.15

        # Read after 14 half-lives (14 * 24h = 336h)
        far_future = now + timedelta(hours=336)
        recovered = fast_decay_store.get_params(
            "user-1", "idea", now=far_future
        )

        assert recovered.mean > suppressed.mean
        assert recovered.mean > 0.4  # Mostly recovered

    def test_decay_applied_on_read_not_write(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Decay should be lazy (applied on read), not persisted until feedback."""
        now = datetime.now(timezone.utc)
        fast_decay_store.record_feedback(
            "user-1", "task", FeedbackType.ACCEPTED, now=now
        )

        # Read with decay at future time
        future = now + timedelta(hours=24)
        p_decayed = fast_decay_store.get_params("user-1", "task", now=future)

        # Raw stored value should still be the original
        p_raw = fast_decay_store.get_params(
            "user-1", "task", apply_decay=False
        )
        assert p_raw.alpha == pytest.approx(2.0)
        assert p_decayed.alpha < p_raw.alpha

    def test_decay_before_feedback_update(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """When recording feedback, decay should be applied first."""
        now = datetime.now(timezone.utc)

        # Set high alpha
        for _ in range(10):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED, now=now
            )

        # Record new feedback after half-life
        future = now + timedelta(hours=24)
        p = fast_decay_store.record_feedback(
            "user-1", "task", FeedbackType.ACCEPTED, now=future
        )

        # Should be: decayed_alpha + 1.0 (accepted)
        # decayed_alpha = 1.0 + (11.0 - 1.0) * 0.5 = 6.0
        # final = 6.0 + 1.0 = 7.0
        assert p.alpha == pytest.approx(7.0, abs=0.2)


# ── Weight convergence ───────────────────────────────────────────────


class TestWeightConvergence:
    """Tests for Beta parameter convergence under consistent feedback.

    These tests validate that:
    1. Repeated positive feedback drives mean toward 1.0
    2. Repeated negative feedback drives mean toward 0.0
    3. Mixed feedback converges to a stable ratio
    4. Convergence rate is reasonable (not too fast, not too slow)
    """

    def test_positive_convergence(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Consistent acceptance should converge mean toward 1.0."""
        means: list[float] = []
        for i in range(100):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED
            )
            p = no_decay_store.get_params("user-1", "task", apply_decay=False)
            means.append(p.mean)

        # Mean should be monotonically increasing
        for i in range(1, len(means)):
            assert means[i] >= means[i - 1]

        # Should converge above 0.9 after 100 acceptances
        assert means[-1] > 0.9

    def test_negative_convergence(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Consistent dismissal should converge mean downward."""
        means: list[float] = []
        for i in range(100):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.DISMISSED
            )
            p = no_decay_store.get_params("user-1", "task", apply_decay=False)
            means.append(p.mean)

        # Mean should be monotonically decreasing
        for i in range(1, len(means)):
            assert means[i] <= means[i - 1]

        # Should converge below 0.15 after 100 dismissals
        assert means[-1] < 0.15

    def test_mixed_feedback_converges(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """70/30 accept/dismiss ratio should converge to ~0.7 mean."""
        rng = random.Random(42)

        for _ in range(200):
            if rng.random() < 0.7:
                no_decay_store.record_feedback(
                    "user-1", "task", FeedbackType.ACCEPTED
                )
            else:
                no_decay_store.record_feedback(
                    "user-1", "task", FeedbackType.DISMISSED
                )

        p = no_decay_store.get_params("user-1", "task", apply_decay=False)

        # With 70% accept, 30% dismiss, and the feedback weights:
        # alpha += 1.0 per accept, beta += 0.5 per dismiss
        # Expected: ~140 alpha increments + 1 prior, ~30 beta increments + 1 prior
        # Mean should be higher than 0.7 because dismiss weight is < accept weight
        assert p.mean > 0.65
        assert p.mean < 0.95

    def test_variance_decreases_with_observations(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Variance should decrease as more feedback accumulates."""
        variances: list[float] = []
        for _ in range(50):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED
            )
            p = no_decay_store.get_params("user-1", "task", apply_decay=False)
            variances.append(p.variance)

        # Overall trend should be decreasing
        assert variances[-1] < variances[0]
        # Final variance should be quite small
        assert variances[-1] < 0.01

    def test_convergence_speed_reasonable(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """After 10 observations, mean should have moved meaningfully."""
        for _ in range(10):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED
            )

        p = no_decay_store.get_params("user-1", "task", apply_decay=False)

        # Should be noticeably above 0.5 but not at 1.0 yet
        assert p.mean > 0.7
        assert p.mean < 0.98

    def test_category_rankings_reflect_feedback(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Rankings should reflect relative feedback patterns."""
        # Strongly prefer "task"
        for _ in range(20):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED
            )
        # Moderately prefer "event"
        for _ in range(10):
            no_decay_store.record_feedback(
                "user-1", "event", FeedbackType.ACCEPTED
            )
        # Dismiss "idea"
        for _ in range(10):
            no_decay_store.record_feedback(
                "user-1", "idea", FeedbackType.DISMISSED
            )

        rankings = no_decay_store.get_category_rankings(
            "user-1", apply_decay=False
        )

        # task should be first, event second
        categories = [cat for cat, _ in rankings]
        assert categories[0] == "task"
        assert categories[1] == "event"

        # idea should be lowest
        means = {cat: mean for cat, mean in rankings}
        assert means["idea"] < means["task"]
        assert means["idea"] < means["event"]


# ── Adaptation (non-stationarity) ────────────────────────────────────


class TestAdaptation:
    """Tests for adaptation to changing user preferences.

    The decay mechanism should allow the model to shift when user
    behavior changes, rather than being stuck on historical patterns.
    """

    def test_preference_shift_detection(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Model should adapt when user switches preferred category.

        Scenario: User initially prefers "task", then shifts to "idea".
        After enough time and new feedback, "idea" should overtake "task".
        """
        now = datetime.now(timezone.utc)

        # Phase 1: Heavy "task" preference
        for _ in range(30):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED, now=now
            )

        # Phase 2: Time passes, user shifts to "idea"
        phase2_time = now + timedelta(hours=48)  # 2 half-lives
        for _ in range(30):
            fast_decay_store.record_feedback(
                "user-1", "idea", FeedbackType.ACCEPTED, now=phase2_time
            )

        # Read at phase 2 time
        task_params = fast_decay_store.get_params(
            "user-1", "task", now=phase2_time
        )
        idea_params = fast_decay_store.get_params(
            "user-1", "idea", now=phase2_time
        )

        # After 2 half-lives, task's excess decayed to ~25%
        # Meanwhile idea just got 30 fresh acceptances
        assert idea_params.mean > task_params.mean

    def test_gradual_preference_evolution(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Preferences should evolve gradually, not flip instantly."""
        now = datetime.now(timezone.utc)

        # Build up "task" preference
        for _ in range(20):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED, now=now
            )

        _initial_mean = fast_decay_store.get_params(
            "user-1", "task", apply_decay=False
        ).mean

        # Check decay at increasing intervals
        means_over_time = []
        for hours in [6, 12, 24, 48, 96]:
            future = now + timedelta(hours=hours)
            p = fast_decay_store.get_params("user-1", "task", now=future)
            means_over_time.append(p.mean)

        # Should be monotonically decreasing (decay toward prior)
        for i in range(1, len(means_over_time)):
            assert means_over_time[i] < means_over_time[i - 1]

        # All should still be above 0.5 (prior) since there's only positive evidence
        for m in means_over_time[:-1]:  # Last one might be very close to 0.5
            assert m > 0.5

    def test_fresh_feedback_outweighs_stale(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Recent feedback should have more influence than old feedback."""
        now = datetime.now(timezone.utc)

        # Old positive feedback
        for _ in range(20):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED, now=now
            )

        # Time passes (3 half-lives = 72h)
        future = now + timedelta(hours=72)

        # New negative feedback
        for _ in range(10):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.DISMISSED, now=future
            )

        # Read immediately after new feedback
        p = fast_decay_store.get_params(
            "user-1", "task", apply_decay=False
        )

        # The old 20 acceptances decayed significantly (factor ~0.125)
        # New 10 dismissals are fresh
        # Mean should reflect the recent negative trend
        # After 3 half-lives: alpha = 1 + (20)*0.125 + 0 = 3.5 (approx)
        # beta = 1 + 0 + 10*0.5 = 6.0
        # But decay happens incrementally per record_feedback call,
        # so the exact value depends on implementation
        # Key assertion: mean should be below 0.5 (negative trend dominates)
        assert p.mean < 0.55

    def test_recovery_from_negative_phase(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """After a period of dismissals, the model should eventually allow recovery.

        Anti-shame design: temporary dismissal patterns shouldn't
        permanently suppress a category.
        """
        now = datetime.now(timezone.utc)

        # Phase 1: Lots of dismissals
        for _ in range(30):
            fast_decay_store.record_feedback(
                "user-1", "idea", FeedbackType.DISMISSED, now=now
            )

        suppressed = fast_decay_store.get_params(
            "user-1", "idea", apply_decay=False
        )
        assert suppressed.mean < 0.2  # Quite suppressed

        # Phase 2: 10 half-lives later, start accepting
        far_future = now + timedelta(hours=240)
        for _ in range(10):
            fast_decay_store.record_feedback(
                "user-1", "idea", FeedbackType.ACCEPTED, now=far_future
            )

        recovered = fast_decay_store.get_params(
            "user-1", "idea", apply_decay=False
        )

        # Should be well above the suppressed level
        assert recovered.mean > suppressed.mean
        # Should be above 0.5 (positive evidence now dominates)
        assert recovered.mean > 0.5


# ── Serialization & import/export ────────────────────────────────────


class TestSerialization:
    """Tests for export/import roundtrips."""

    def test_export_import_roundtrip(self, store: UserPreferenceStore) -> None:
        """Export → import should preserve all data."""
        # Build up some state
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)
        store.record_feedback("user-1", "task", FeedbackType.COMPLETED)
        store.record_feedback("user-1", "event", FeedbackType.DISMISSED)

        # Export
        exported = store.export_snapshot("user-1")

        # Import into new store
        new_store = UserPreferenceStore()
        imported_user_id = new_store.import_snapshot(exported)

        assert imported_user_id == "user-1"

        # Verify params match
        orig = store.get_all_params("user-1", apply_decay=False)
        restored = new_store.get_all_params("user-1", apply_decay=False)

        for cat in DEFAULT_CATEGORIES:
            assert restored[cat].alpha == pytest.approx(orig[cat].alpha)
            assert restored[cat].beta == pytest.approx(orig[cat].beta)
            assert (
                restored[cat].total_observations == orig[cat].total_observations
            )

    def test_snapshot_to_dict_roundtrip(self) -> None:
        """UserPreferenceSnapshot serialization should be lossless."""
        now = datetime.now(timezone.utc)
        snap = UserPreferenceSnapshot(
            user_id="test-user",
            params={
                "task": BetaParams(alpha=5.0, beta=2.0, total_observations=10),
                "idea": BetaParams(alpha=1.5, beta=3.0, total_observations=5),
            },
            created_at=now,
            last_interaction=now,
            total_feedback_count=15,
            version=1,
        )

        data = snap.to_dict()
        restored = UserPreferenceSnapshot.from_dict(data)

        assert restored.user_id == snap.user_id
        assert restored.total_feedback_count == snap.total_feedback_count
        assert restored.version == snap.version
        assert set(restored.params.keys()) == set(snap.params.keys())
        for cat in snap.params:
            assert restored.params[cat].alpha == pytest.approx(
                snap.params[cat].alpha
            )
            assert restored.params[cat].beta == pytest.approx(
                snap.params[cat].beta
            )


# ── Backend isolation ────────────────────────────────────────────────


class TestInMemoryBackend:
    """Tests for the InMemoryPreferenceBackend."""

    def test_deep_copy_on_load(self) -> None:
        """Load should return a deep copy, not a reference."""
        backend = InMemoryPreferenceBackend()
        snap = UserPreferenceSnapshot(
            user_id="user-1",
            params={"task": BetaParams(alpha=5.0, beta=2.0)},
        )
        backend.save_snapshot(snap)

        loaded = backend.load_snapshot("user-1")
        assert loaded is not None
        loaded.params["task"].alpha = 999.0

        # Original should be unchanged
        reloaded = backend.load_snapshot("user-1")
        assert reloaded is not None
        assert reloaded.params["task"].alpha == pytest.approx(5.0)

    def test_load_nonexistent_returns_none(self) -> None:
        """Loading a non-existent user should return None."""
        backend = InMemoryPreferenceBackend()
        assert backend.load_snapshot("nonexistent") is None

    def test_delete_returns_bool(self) -> None:
        """Delete should return True if found, False otherwise."""
        backend = InMemoryPreferenceBackend()
        snap = UserPreferenceSnapshot(user_id="user-1")
        backend.save_snapshot(snap)

        assert backend.delete_snapshot("user-1") is True
        assert backend.delete_snapshot("user-1") is False

    def test_list_users_empty(self) -> None:
        """Empty backend should return empty list."""
        backend = InMemoryPreferenceBackend()
        assert backend.list_users() == []


# ── Reset & delete ───────────────────────────────────────────────────


class TestResetAndDelete:
    """Tests for user reset and deletion."""

    def test_reset_restores_uniform_priors(
        self, store: UserPreferenceStore
    ) -> None:
        """Reset should restore all categories to Beta(1,1)."""
        for _ in range(10):
            store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)

        store.reset_user("user-1")

        p = store.get_params("user-1", "task", apply_decay=False)
        assert p.alpha == 1.0
        assert p.beta == 1.0

    def test_reset_clears_feedback_count(
        self, store: UserPreferenceStore
    ) -> None:
        """Reset should clear the total feedback count."""
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)
        store.reset_user("user-1")

        snapshot = store.get_snapshot("user-1", apply_decay=False)
        assert snapshot.total_feedback_count == 0

    def test_delete_user(self, store: UserPreferenceStore) -> None:
        """Delete should remove user from backend."""
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)
        assert store.delete_user("user-1") is True
        assert "user-1" not in store.list_users()

    def test_delete_nonexistent_returns_false(
        self, store: UserPreferenceStore
    ) -> None:
        """Deleting non-existent user should return False."""
        assert store.delete_user("ghost") is False


# ── Anti-shame design ────────────────────────────────────────────────


class TestAntiShameDesign:
    """Tests verifying anti-shame principles are upheld."""

    def test_dismissed_weight_less_than_accepted(self) -> None:
        """Dismissal impact should be less than acceptance impact."""
        fw = FeedbackWeights()
        assert fw.dismissed_beta < fw.accepted_alpha

    def test_ignored_weight_less_than_dismissed(self) -> None:
        """Ignored impact should be less than dismissal impact."""
        fw = FeedbackWeights()
        assert fw.ignored_beta < fw.dismissed_beta

    def test_no_category_fully_suppressed(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Even after many dismissals, a category should not reach mean=0."""
        for _ in range(100):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.DISMISSED
            )

        p = no_decay_store.get_params("user-1", "task", apply_decay=False)
        assert p.mean > 0.0  # Never fully suppressed

    def test_snoozed_partial_positive(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Snoozed should be treated as partial positive, not punishment."""
        for _ in range(20):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.SNOOZED
            )

        p = no_decay_store.get_params("user-1", "task", apply_decay=False)

        # Snoozed adds to both alpha and beta, but more to alpha
        # With default weights: alpha += 0.3, beta += 0.2 per snooze
        # After 20: alpha = 1 + 20*0.3 = 7.0, beta = 1 + 20*0.2 = 5.0
        # Mean = 7/12 ≈ 0.583 — slightly positive
        assert p.mean > 0.5  # Net positive

    def test_decay_prevents_permanent_suppression(
        self, fast_decay_store: UserPreferenceStore
    ) -> None:
        """Decay ensures old dismissals fade, preventing permanent category death."""
        now = datetime.now(timezone.utc)

        # Massive dismissal wave
        for _ in range(100):
            fast_decay_store.record_feedback(
                "user-1", "task", FeedbackType.DISMISSED, now=now
            )

        # Way in the future — all old data should have decayed
        far_future = now + timedelta(days=365)
        p = fast_decay_store.get_params("user-1", "task", now=far_future)

        # Should be very close to the prior (0.5)
        assert abs(p.mean - 0.5) < 0.05


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_empty_user_id(self, store: UserPreferenceStore) -> None:
        """Empty string user ID should work (not crash)."""
        store.record_feedback("", "task", FeedbackType.ACCEPTED)
        p = store.get_params("", "task", apply_decay=False)
        assert p.alpha > 1.0

    def test_very_large_alpha_beta(
        self, no_decay_store: UserPreferenceStore
    ) -> None:
        """Store should handle very large parameter values."""
        for _ in range(1000):
            no_decay_store.record_feedback(
                "user-1", "task", FeedbackType.ACCEPTED
            )

        p = no_decay_store.get_params("user-1", "task", apply_decay=False)
        assert p.alpha > 1000
        assert p.mean > 0.99

    def test_concurrent_category_access(
        self, store: UserPreferenceStore
    ) -> None:
        """Updating different categories for same user should work correctly."""
        for cat in DEFAULT_CATEGORIES:
            store.record_feedback("user-1", cat, FeedbackType.ACCEPTED)

        snapshot = store.get_snapshot("user-1", apply_decay=False)
        assert snapshot.total_feedback_count == 7

        for cat in DEFAULT_CATEGORIES:
            p = snapshot.params[cat]
            assert p.total_observations == 1

    def test_new_category_added_to_existing_user(self) -> None:
        """If a new category is added, existing users should get it."""
        initial_cats = ["task", "event"]
        store = UserPreferenceStore(categories=initial_cats)
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)

        # Create new store with additional category
        extended_cats = ["task", "event", "idea"]
        store2 = UserPreferenceStore(
            categories=extended_cats,
            backend=store._backend,
        )

        # Existing user should get the new category
        params = store2.get_all_params("user-1", apply_decay=False)
        assert "idea" in params
        assert params["idea"].alpha == 1.0  # Uniform prior for new category
        # Original category should be preserved
        assert params["task"].alpha > 1.0
