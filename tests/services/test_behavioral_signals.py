"""Tests for the behavioral signal collector.

AC 11 Sub-AC 2: Validates that user interactions (accept, dismiss, complete,
snooze) are correctly converted into reward/penalty signals for the Thompson
Sampler, with context-aware adjustments and behavioral profile building.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.behavioral_signals import (
    BehavioralSignalCollector,
    InMemorySignalStore,
    RewardConfig,
    SignalKind,
    _energy_bucket,
    _load_bucket,
)
from blurt.services.feedback import FeedbackAction, FeedbackEvent
from blurt.services.surfacing.models import BehavioralProfile
from blurt.services.surfacing.thompson import ThompsonSampler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sampler() -> ThompsonSampler:
    """Fresh Thompson Sampler with default signals."""
    return ThompsonSampler()


@pytest.fixture
def store() -> InMemorySignalStore:
    return InMemorySignalStore()


@pytest.fixture
def collector(sampler: ThompsonSampler, store: InMemorySignalStore) -> BehavioralSignalCollector:
    return BehavioralSignalCollector(sampler=sampler, store=store)


def _make_event(
    action: FeedbackAction = FeedbackAction.ACCEPT,
    task_id: str = "task-1",
    user_id: str = "user-1",
    energy: float = 0.5,
    mood: float = 0.0,
    time_of_day: str = "morning",
    snooze_minutes: int | None = None,
    timestamp: datetime | None = None,
) -> FeedbackEvent:
    return FeedbackEvent(
        id=str(uuid.uuid4()),
        task_id=task_id,
        user_id=user_id,
        action=action,
        energy_level=energy,
        mood_valence=mood,
        time_of_day=time_of_day,
        snooze_minutes=snooze_minutes,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Signal kind mapping tests
# ---------------------------------------------------------------------------


class TestSignalKindMapping:
    """Accept → REWARD, Complete → REWARD, Dismiss → PENALTY, Snooze → DEFERRAL."""

    def test_accept_produces_reward(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(action=FeedbackAction.ACCEPT)
        signal = collector.collect(event)
        assert signal.kind == SignalKind.REWARD
        assert signal.magnitude > 0
        assert signal.reward_value > 0

    def test_complete_produces_reward(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(action=FeedbackAction.COMPLETE)
        signal = collector.collect(event)
        assert signal.kind == SignalKind.REWARD
        assert signal.magnitude > 0
        assert signal.reward_value > 0

    def test_complete_stronger_than_accept(self, collector: BehavioralSignalCollector) -> None:
        accept_signal = collector.collect(_make_event(action=FeedbackAction.ACCEPT))
        complete_signal = collector.collect(_make_event(action=FeedbackAction.COMPLETE))
        assert complete_signal.magnitude > accept_signal.magnitude
        assert complete_signal.reward_value > accept_signal.reward_value

    def test_dismiss_produces_penalty(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(action=FeedbackAction.DISMISS)
        signal = collector.collect(event)
        assert signal.kind == SignalKind.PENALTY
        assert signal.magnitude > 0  # Magnitude always positive
        assert signal.reward_value < 0  # But reward is negative

    def test_snooze_produces_deferral(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(action=FeedbackAction.SNOOZE, snooze_minutes=30)
        signal = collector.collect(event)
        assert signal.kind == SignalKind.DEFERRAL


# ---------------------------------------------------------------------------
# Reward magnitude tests
# ---------------------------------------------------------------------------


class TestRewardMagnitudes:
    """Verify reward/penalty magnitudes match the RewardConfig."""

    def test_default_accept_magnitude(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(action=FeedbackAction.ACCEPT)
        signal = collector.collect(event)
        assert signal.magnitude == pytest.approx(1.0, abs=0.3)

    def test_default_complete_magnitude(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(action=FeedbackAction.COMPLETE)
        signal = collector.collect(event)
        assert signal.magnitude == pytest.approx(2.0, abs=0.5)

    def test_dismiss_penalty_less_than_accept_reward(
        self, collector: BehavioralSignalCollector
    ) -> None:
        """Anti-shame: penalties are smaller than rewards."""
        accept = collector.collect(_make_event(action=FeedbackAction.ACCEPT))
        dismiss = collector.collect(_make_event(action=FeedbackAction.DISMISS))
        assert dismiss.magnitude < accept.magnitude

    def test_snooze_penalty_smallest(self, collector: BehavioralSignalCollector) -> None:
        """Anti-shame: snooze has the gentlest penalty."""
        dismiss = collector.collect(_make_event(action=FeedbackAction.DISMISS))
        snooze = collector.collect(_make_event(action=FeedbackAction.SNOOZE))
        assert snooze.magnitude < dismiss.magnitude

    def test_custom_config(self) -> None:
        config = RewardConfig(accept_reward=5.0, complete_reward=10.0)
        collector = BehavioralSignalCollector(config=config)
        signal = collector.collect(_make_event(action=FeedbackAction.COMPLETE))
        assert signal.magnitude > 5.0


# ---------------------------------------------------------------------------
# Context-aware adjustments
# ---------------------------------------------------------------------------


class TestContextAdjustments:
    """Signals are adjusted based on energy and interaction history."""

    def test_high_energy_amplifies_signal(self) -> None:
        collector = BehavioralSignalCollector()
        normal = collector.collect(_make_event(energy=0.5))
        high = collector.collect(_make_event(energy=0.9))
        assert high.magnitude > normal.magnitude

    def test_low_energy_dampens_signal(self) -> None:
        collector = BehavioralSignalCollector()
        normal = collector.collect(_make_event(energy=0.5))
        low = collector.collect(_make_event(energy=0.1))
        assert low.magnitude < normal.magnitude

    def test_repeat_dismiss_decays(self) -> None:
        """Repeated dismissals of the same task get gentler penalties."""
        store = InMemorySignalStore()
        collector = BehavioralSignalCollector(store=store)

        first = collector.collect(_make_event(action=FeedbackAction.DISMISS))
        second = collector.collect(_make_event(action=FeedbackAction.DISMISS))
        third = collector.collect(_make_event(action=FeedbackAction.DISMISS))

        # Each subsequent dismissal should have smaller magnitude
        assert second.magnitude < first.magnitude
        assert third.magnitude < second.magnitude

    def test_repeat_dismiss_different_tasks_no_decay(self) -> None:
        """Dismissing different tasks doesn't trigger decay."""
        store = InMemorySignalStore()
        collector = BehavioralSignalCollector(store=store)

        first = collector.collect(
            _make_event(action=FeedbackAction.DISMISS, task_id="task-A")
        )
        second = collector.collect(
            _make_event(action=FeedbackAction.DISMISS, task_id="task-B")
        )

        # Different tasks — no decay applied
        assert second.magnitude == pytest.approx(first.magnitude, abs=0.01)


# ---------------------------------------------------------------------------
# Thompson Sampler integration
# ---------------------------------------------------------------------------


class TestSamplerIntegration:
    """Signals update the Thompson Sampler arms correctly."""

    def test_accept_updates_sampler_alpha(self) -> None:
        sampler = ThompsonSampler()
        collector = BehavioralSignalCollector(sampler=sampler)

        initial_alpha = sampler.arms["time_relevance"].alpha
        contributions = {"time_relevance": 0.8, "energy_match": 0.2}

        collector.collect(
            _make_event(action=FeedbackAction.ACCEPT),
            signal_contributions=contributions,
        )

        # Alpha should increase (positive reward)
        assert sampler.arms["time_relevance"].alpha > initial_alpha

    def test_dismiss_updates_sampler_beta(self) -> None:
        sampler = ThompsonSampler()
        collector = BehavioralSignalCollector(sampler=sampler)

        initial_beta = sampler.arms["time_relevance"].beta
        contributions = {"time_relevance": 0.8, "energy_match": 0.2}

        collector.collect(
            _make_event(action=FeedbackAction.DISMISS),
            signal_contributions=contributions,
        )

        # Beta should increase (negative reward)
        assert sampler.arms["time_relevance"].beta > initial_beta

    def test_no_contributions_no_sampler_update(self) -> None:
        sampler = ThompsonSampler()
        collector = BehavioralSignalCollector(sampler=sampler)

        initial_state = sampler.get_state()
        collector.collect(_make_event(action=FeedbackAction.ACCEPT))

        # No contributions → no update
        assert sampler.get_state() == initial_state

    def test_proportional_contributions(self) -> None:
        """Signals with higher contribution get larger updates."""
        sampler = ThompsonSampler()
        collector = BehavioralSignalCollector(sampler=sampler)

        contributions = {"time_relevance": 0.9, "energy_match": 0.1}
        collector.collect(
            _make_event(action=FeedbackAction.ACCEPT),
            signal_contributions=contributions,
        )

        time_delta = sampler.arms["time_relevance"].alpha - 1.0  # minus prior
        energy_delta = sampler.arms["energy_match"].alpha - 1.0

        assert time_delta > energy_delta

    def test_multiple_signals_compound(self) -> None:
        """Multiple positive signals compound in the sampler."""
        sampler = ThompsonSampler()
        collector = BehavioralSignalCollector(sampler=sampler)

        contributions = {"time_relevance": 0.5, "energy_match": 0.5}

        for _ in range(5):
            collector.collect(
                _make_event(action=FeedbackAction.COMPLETE),
                signal_contributions=contributions,
            )

        # After 5 completions, alpha should be well above prior
        assert sampler.arms["time_relevance"].alpha > 3.0


# ---------------------------------------------------------------------------
# Signal storage tests
# ---------------------------------------------------------------------------


class TestSignalStorage:
    def test_signals_stored(self, collector: BehavioralSignalCollector, store: InMemorySignalStore) -> None:
        collector.collect(_make_event(action=FeedbackAction.ACCEPT))
        collector.collect(_make_event(action=FeedbackAction.DISMISS))

        signals = store.get_signals(user_id="user-1")
        assert len(signals) == 2

    def test_signals_filtered_by_user(self, collector: BehavioralSignalCollector, store: InMemorySignalStore) -> None:
        collector.collect(_make_event(user_id="user-A"))
        collector.collect(_make_event(user_id="user-B"))

        a_signals = store.get_signals(user_id="user-A")
        b_signals = store.get_signals(user_id="user-B")
        assert len(a_signals) == 1
        assert len(b_signals) == 1

    def test_signals_filtered_by_time(self, collector: BehavioralSignalCollector, store: InMemorySignalStore) -> None:
        old = _make_event(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=48),
        )
        recent = _make_event(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        collector.collect(old)
        collector.collect(recent)

        since = datetime.now(timezone.utc) - timedelta(hours=24)
        signals = store.get_signals(user_id="user-1", since=since)
        assert len(signals) == 1

    def test_signal_has_correct_metadata(self, collector: BehavioralSignalCollector) -> None:
        event = _make_event(
            action=FeedbackAction.ACCEPT,
            task_id="task-42",
            user_id="user-7",
            time_of_day="evening",
        )
        signal = collector.collect(event, intent="reminder")

        assert signal.user_id == "user-7"
        assert signal.task_id == "task-42"
        assert signal.source_action == FeedbackAction.ACCEPT
        assert signal.source_event_id == event.id
        assert signal.context.time_of_day == "evening"
        assert signal.context.intent == "reminder"


# ---------------------------------------------------------------------------
# Batch collection tests
# ---------------------------------------------------------------------------


class TestBatchCollection:
    def test_batch_collects_all(self, collector: BehavioralSignalCollector) -> None:
        events = [
            _make_event(action=FeedbackAction.ACCEPT),
            _make_event(action=FeedbackAction.DISMISS),
            _make_event(action=FeedbackAction.COMPLETE),
        ]
        batch = collector.collect_batch(events)

        assert batch.signal_count == 3
        assert len(batch.signals) == 3

    def test_batch_net_reward(self, collector: BehavioralSignalCollector) -> None:
        events = [
            _make_event(action=FeedbackAction.COMPLETE),  # +2.0
            _make_event(action=FeedbackAction.ACCEPT),    # +1.0
            _make_event(action=FeedbackAction.DISMISS),   # -0.5
        ]
        batch = collector.collect_batch(events)

        # Net should be positive (completions + accepts outweigh dismiss)
        assert batch.net_reward > 0

    def test_batch_with_contributions(self, collector: BehavioralSignalCollector) -> None:
        events = [_make_event(action=FeedbackAction.ACCEPT)]
        contribs = {events[0].id: {"time_relevance": 0.6, "energy_match": 0.4}}

        batch = collector.collect_batch(events, contributions_map=contribs)

        assert batch.signal_count == 1
        assert batch.dimension_contributions  # Non-empty


# ---------------------------------------------------------------------------
# Interaction stats tests
# ---------------------------------------------------------------------------


class TestInteractionStats:
    def test_empty_stats(self, collector: BehavioralSignalCollector) -> None:
        stats = collector.get_interaction_stats("nonexistent-user")
        assert stats.total_interactions == 0
        assert stats.acceptance_rate == 0.0
        assert stats.engagement_score == 0.5  # Neutral default

    def test_stats_counts(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(_make_event(action=FeedbackAction.ACCEPT))
        collector.collect(_make_event(action=FeedbackAction.ACCEPT))
        collector.collect(_make_event(action=FeedbackAction.DISMISS))
        collector.collect(_make_event(action=FeedbackAction.COMPLETE))
        collector.collect(_make_event(action=FeedbackAction.SNOOZE))

        stats = collector.get_interaction_stats("user-1")

        assert stats.total_interactions == 5
        assert stats.accepts == 2
        assert stats.completions == 1
        assert stats.dismissals == 1
        assert stats.snoozes == 1

    def test_acceptance_rate(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(_make_event(action=FeedbackAction.ACCEPT))
        collector.collect(_make_event(action=FeedbackAction.COMPLETE))
        collector.collect(_make_event(action=FeedbackAction.DISMISS))
        collector.collect(_make_event(action=FeedbackAction.DISMISS))

        stats = collector.get_interaction_stats("user-1")
        assert stats.acceptance_rate == pytest.approx(0.5, abs=0.01)

    def test_stats_by_time_of_day(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(_make_event(action=FeedbackAction.COMPLETE, time_of_day="morning"))
        collector.collect(_make_event(action=FeedbackAction.DISMISS, time_of_day="evening"))

        stats = collector.get_interaction_stats("user-1")
        assert "morning" in stats.by_time_of_day
        assert "evening" in stats.by_time_of_day
        # Morning (completion) should have higher avg reward
        assert stats.by_time_of_day["morning"] > stats.by_time_of_day["evening"]

    def test_stats_by_intent(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(
            _make_event(action=FeedbackAction.COMPLETE),
            intent="task",
        )
        collector.collect(
            _make_event(action=FeedbackAction.DISMISS),
            intent="reminder",
        )

        stats = collector.get_interaction_stats("user-1")
        assert "task" in stats.by_intent
        assert "reminder" in stats.by_intent

    def test_velocity_improving(self, collector: BehavioralSignalCollector) -> None:
        """Velocity should be positive when recent signals are better."""
        now = datetime.now(timezone.utc)

        # Older: dismissals
        for i in range(5):
            collector.collect(_make_event(
                action=FeedbackAction.DISMISS,
                timestamp=now - timedelta(hours=10 - i),
            ))

        # Recent: completions
        for i in range(5):
            collector.collect(_make_event(
                action=FeedbackAction.COMPLETE,
                timestamp=now - timedelta(hours=4 - i),
            ))

        stats = collector.get_interaction_stats("user-1", window_hours=24)
        assert stats.velocity > 0

    def test_velocity_declining(self, collector: BehavioralSignalCollector) -> None:
        """Velocity should be negative when recent signals are worse."""
        now = datetime.now(timezone.utc)

        # Older: completions
        for i in range(5):
            collector.collect(_make_event(
                action=FeedbackAction.COMPLETE,
                timestamp=now - timedelta(hours=10 - i),
            ))

        # Recent: dismissals
        for i in range(5):
            collector.collect(_make_event(
                action=FeedbackAction.DISMISS,
                timestamp=now - timedelta(hours=4 - i),
            ))

        stats = collector.get_interaction_stats("user-1", window_hours=24)
        assert stats.velocity < 0


# ---------------------------------------------------------------------------
# Behavioral profile building
# ---------------------------------------------------------------------------


class TestBehavioralProfile:
    def test_empty_profile(self, collector: BehavioralSignalCollector) -> None:
        profile = collector.build_behavioral_profile("nonexistent-user")
        assert isinstance(profile, BehavioralProfile)
        assert profile.avg_daily_completions == 3.0  # Default
        assert profile.tasks_completed_today == 0

    def test_completion_by_time(self, collector: BehavioralSignalCollector) -> None:
        # Morning: 2 completions, 1 dismiss → 66% rate
        collector.collect(_make_event(action=FeedbackAction.COMPLETE, time_of_day="morning"))
        collector.collect(_make_event(action=FeedbackAction.COMPLETE, time_of_day="morning"))
        collector.collect(_make_event(action=FeedbackAction.DISMISS, time_of_day="morning"))

        # Evening: 0 completions, 2 dismisses → 0% rate
        collector.collect(_make_event(action=FeedbackAction.DISMISS, time_of_day="evening"))
        collector.collect(_make_event(action=FeedbackAction.DISMISS, time_of_day="evening"))

        profile = collector.build_behavioral_profile("user-1")

        assert profile.completion_by_time["morning"] == pytest.approx(0.667, abs=0.01)
        assert profile.completion_by_time["evening"] == pytest.approx(0.0, abs=0.01)

    def test_completion_by_load(self, collector: BehavioralSignalCollector) -> None:
        # Low load: complete
        collector.collect(
            _make_event(action=FeedbackAction.COMPLETE),
            cognitive_load=0.2,
        )
        # High load: dismiss
        collector.collect(
            _make_event(action=FeedbackAction.DISMISS),
            cognitive_load=0.9,
        )

        profile = collector.build_behavioral_profile("user-1")
        assert profile.completion_by_load.get("low", 0) > profile.completion_by_load.get("high", 0)

    def test_completion_by_tag(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(
            _make_event(action=FeedbackAction.COMPLETE),
            tags=["project-alpha"],
        )
        collector.collect(
            _make_event(action=FeedbackAction.DISMISS),
            tags=["project-beta"],
        )

        profile = collector.build_behavioral_profile("user-1")
        assert profile.completion_by_tag.get("project-alpha", 0) == 1.0
        assert profile.completion_by_tag.get("project-beta", 0) == 0.0

    def test_hourly_activity(self, collector: BehavioralSignalCollector) -> None:
        now = datetime.now(timezone.utc)
        # Create signals at different hours
        for h in [9, 9, 9, 14]:
            ts = now.replace(hour=h, minute=0, second=0, microsecond=0)
            collector.collect(_make_event(timestamp=ts))

        profile = collector.build_behavioral_profile("user-1")
        assert 9 in profile.hourly_activity
        assert profile.hourly_activity[9] == 1.0  # Most active hour

    def test_tasks_completed_today(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(_make_event(action=FeedbackAction.COMPLETE))
        collector.collect(_make_event(action=FeedbackAction.COMPLETE))
        collector.collect(_make_event(action=FeedbackAction.ACCEPT))

        profile = collector.build_behavioral_profile("user-1")
        assert profile.tasks_completed_today == 2

    def test_avg_daily_completions(self, collector: BehavioralSignalCollector) -> None:
        now = datetime.now(timezone.utc)
        # 3 completions over ~3 days
        for d in range(3):
            collector.collect(_make_event(
                action=FeedbackAction.COMPLETE,
                timestamp=now - timedelta(days=d),
            ))

        # Use 7-day window (default)
        profile = collector.build_behavioral_profile("user-1", window_hours=168)
        assert profile.avg_daily_completions == pytest.approx(3.0 / 7.0, abs=0.1)

    def test_hours_since_last_interaction(self, collector: BehavioralSignalCollector) -> None:
        collector.collect(_make_event(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=3),
        ))

        profile = collector.build_behavioral_profile("user-1")
        assert profile.hours_since_last_interaction == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# Anti-shame design tests
# ---------------------------------------------------------------------------


class TestAntiShameDesign:
    """Verify the collector follows anti-shame principles."""

    def test_penalties_smaller_than_rewards(self) -> None:
        """Penalties are always gentler than rewards."""
        config = RewardConfig()
        assert config.dismiss_penalty < config.accept_reward
        assert config.snooze_penalty < config.dismiss_penalty

    def test_snooze_not_fully_negative(self, collector: BehavioralSignalCollector) -> None:
        """Snooze has partial positive component (intent to engage)."""
        config = collector.config
        # Snooze reward > 0 shows intent is recognized
        assert config.snooze_reward > 0

    def test_repeated_dismissals_gentler(self) -> None:
        """Repeatedly dismissing the same task gets gentler, not harsher."""
        store = InMemorySignalStore()
        collector = BehavioralSignalCollector(store=store)

        magnitudes = []
        for _ in range(5):
            signal = collector.collect(_make_event(action=FeedbackAction.DISMISS))
            magnitudes.append(signal.magnitude)

        # Each subsequent magnitude should be <= previous
        for i in range(1, len(magnitudes)):
            assert magnitudes[i] <= magnitudes[i - 1]

    def test_no_interactions_is_neutral(self, collector: BehavioralSignalCollector) -> None:
        """No interactions produces neutral stats, not negative."""
        stats = collector.get_interaction_stats("silent-user")
        assert stats.engagement_score == 0.5  # Neutral, not negative
        assert stats.velocity == 0.0  # No trend

    def test_all_dismissals_still_finite(self) -> None:
        """Even all-dismiss history doesn't produce extreme values."""
        store = InMemorySignalStore()
        collector = BehavioralSignalCollector(store=store)

        for _ in range(20):
            collector.collect(_make_event(action=FeedbackAction.DISMISS))

        stats = collector.get_interaction_stats("user-1")
        assert 0.0 <= stats.engagement_score <= 1.0
        assert -1.0 <= stats.velocity <= 1.0


# ---------------------------------------------------------------------------
# Sampler state inspection
# ---------------------------------------------------------------------------


class TestSamplerStateInspection:
    def test_get_sampler_state(self, collector: BehavioralSignalCollector) -> None:
        state = collector.get_sampler_state()
        assert isinstance(state, dict)
        assert "time_relevance" in state
        assert "alpha" in state["time_relevance"]
        assert "beta" in state["time_relevance"]

    def test_get_expected_weights(self, collector: BehavioralSignalCollector) -> None:
        weights = collector.get_expected_weights()
        assert isinstance(weights, dict)
        # Should sum to ~1.0
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_weights_shift_with_feedback(self) -> None:
        """Weights should shift toward signals that predict completions."""
        sampler = ThompsonSampler()
        collector = BehavioralSignalCollector(sampler=sampler)

        # Heavy time_relevance contributions on completions
        for _ in range(20):
            collector.collect(
                _make_event(action=FeedbackAction.COMPLETE),
                signal_contributions={"time_relevance": 0.9, "energy_match": 0.1},
            )

        weights = collector.get_expected_weights()
        # Time relevance should have highest weight
        assert weights["time_relevance"] > weights["energy_match"]


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_energy_bucket(self) -> None:
        assert _energy_bucket(0.1) == "low_energy"
        assert _energy_bucket(0.5) == "med_energy"
        assert _energy_bucket(0.9) == "high_energy"

    def test_load_bucket(self) -> None:
        assert _load_bucket(0.1) == "low"
        assert _load_bucket(0.5) == "medium"
        assert _load_bucket(0.9) == "high"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_collect_with_empty_event(self, collector: BehavioralSignalCollector) -> None:
        """Minimal event still produces a valid signal."""
        event = FeedbackEvent()
        signal = collector.collect(event)
        assert signal.kind in (SignalKind.PENALTY, SignalKind.NEUTRAL, SignalKind.DEFERRAL, SignalKind.REWARD)
        assert isinstance(signal.magnitude, float)

    def test_batch_empty(self, collector: BehavioralSignalCollector) -> None:
        batch = collector.collect_batch([])
        assert batch.signal_count == 0
        assert batch.net_reward == 0.0

    def test_profile_window_zero(self, collector: BehavioralSignalCollector) -> None:
        """Zero-hour window should return empty profile."""
        collector.collect(_make_event())
        profile = collector.build_behavioral_profile("user-1", window_hours=0.0)
        # Window is 0, so all signals are outside it
        assert profile.tasks_completed_today == 0

    def test_signal_ids_unique(self, collector: BehavioralSignalCollector) -> None:
        signals = [
            collector.collect(_make_event()) for _ in range(10)
        ]
        ids = [s.id for s in signals]
        assert len(set(ids)) == len(ids)
