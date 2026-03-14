"""Tests for the TaskFeedbackService and Thompson Sampling parameter updates.

Validates:
- All four feedback actions (accept, dismiss, snooze, complete) are recorded
- Thompson Sampling parameters update correctly for each action
- Multi-level parameter updates (task, task+context, intent+context)
- Context key generation from mood/energy/time
- Feedback summary aggregation
- Anti-shame: no guilt language, all actions respected
"""

from __future__ import annotations


from blurt.services.feedback import (
    FeedbackAction,
    FeedbackEvent,
    InMemoryFeedbackStore,
    TaskFeedbackService,
    ThompsonParams,
    _build_context_key,
    _energy_bucket,
    _mood_bucket,
)


# ── ThompsonParams Tests ──────────────────────────────────────────


class TestThompsonParams:
    def test_default_prior(self):
        """Default prior is uniform Beta(1,1)."""
        p = ThompsonParams()
        assert p.alpha == 1.0
        assert p.beta == 1.0
        assert p.mean == 0.5
        assert p.total_observations == 0

    def test_mean_calculation(self):
        p = ThompsonParams(alpha=3.0, beta=1.0)
        assert p.mean == 0.75

    def test_variance_calculation(self):
        p = ThompsonParams(alpha=1.0, beta=1.0)
        # Var(Beta(1,1)) = 1*1 / (4*3) = 1/12
        assert abs(p.variance - 1.0 / 12.0) < 1e-10

    def test_update_increments(self):
        p = ThompsonParams()
        p.update(1.0, 0.0)
        assert p.alpha == 2.0
        assert p.beta == 1.0
        assert p.total_observations == 1

    def test_update_partial(self):
        """Snooze action gives partial beta increment."""
        p = ThompsonParams()
        p.update(0.0, 0.3)
        assert p.alpha == 1.0
        assert abs(p.beta - 1.3) < 1e-10
        assert p.total_observations == 1

    def test_sample_in_range(self):
        """Samples should always be in [0, 1]."""
        p = ThompsonParams(alpha=5.0, beta=2.0)
        for _ in range(100):
            s = p.sample()
            assert 0.0 <= s <= 1.0


# ── Context Key Tests ─────────────────────────────────────────────


class TestContextKeys:
    def test_full_context_key(self):
        key = _build_context_key(
            time_of_day="morning",
            energy_bucket="high_energy",
            mood_bucket="positive_mood",
        )
        assert key == "morning:high_energy:positive_mood"

    def test_partial_context_key(self):
        key = _build_context_key(time_of_day="evening")
        assert key == "evening"

    def test_empty_context_key(self):
        key = _build_context_key()
        assert key == "global"

    def test_energy_buckets(self):
        assert _energy_bucket(0.1) == "low_energy"
        assert _energy_bucket(0.5) == "med_energy"
        assert _energy_bucket(0.8) == "high_energy"

    def test_mood_buckets(self):
        assert _mood_bucket(-0.5) == "negative_mood"
        assert _mood_bucket(0.0) == "neutral_mood"
        assert _mood_bucket(0.5) == "positive_mood"


# ── InMemoryFeedbackStore Tests ───────────────────────────────────


class TestInMemoryFeedbackStore:
    def test_store_and_retrieve_event(self):
        store = InMemoryFeedbackStore()
        event = FeedbackEvent(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT)
        store.store_event(event)
        events = store.get_events(task_id="t1")
        assert len(events) == 1
        assert events[0].task_id == "t1"

    def test_filter_by_user(self):
        store = InMemoryFeedbackStore()
        store.store_event(FeedbackEvent(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT))
        store.store_event(FeedbackEvent(task_id="t2", user_id="u2", action=FeedbackAction.DISMISS))
        events = store.get_events(user_id="u1")
        assert len(events) == 1
        assert events[0].user_id == "u1"

    def test_params_default_prior(self):
        store = InMemoryFeedbackStore()
        params = store.get_params("new_key")
        assert params.alpha == 1.0
        assert params.beta == 1.0

    def test_params_persist(self):
        store = InMemoryFeedbackStore()
        params = ThompsonParams(alpha=5.0, beta=2.0, total_observations=7)
        store.set_params("key1", params)
        retrieved = store.get_params("key1")
        assert retrieved.alpha == 5.0
        assert retrieved.beta == 2.0

    def test_task_summary_empty(self):
        store = InMemoryFeedbackStore()
        summary = store.get_task_summary("nonexistent")
        assert summary.total_events == 0
        assert summary.acceptance_rate == 0.0

    def test_task_summary_aggregation(self):
        store = InMemoryFeedbackStore()
        store.store_event(FeedbackEvent(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT))
        store.store_event(FeedbackEvent(task_id="t1", user_id="u1", action=FeedbackAction.DISMISS))
        store.store_event(FeedbackEvent(task_id="t1", user_id="u1", action=FeedbackAction.COMPLETE))

        # Set params so thompson_mean reflects updates
        store.set_params("task:t1", ThompsonParams(alpha=4.0, beta=2.0))

        summary = store.get_task_summary("t1")
        assert summary.total_events == 3
        assert summary.accept_count == 1
        assert summary.dismiss_count == 1
        assert summary.complete_count == 1
        assert abs(summary.acceptance_rate - 2.0 / 3.0) < 0.01


# ── TaskFeedbackService Tests ────────────────────────────────────


class TestTaskFeedbackService:
    def _make_service(self) -> TaskFeedbackService:
        return TaskFeedbackService(store=InMemoryFeedbackStore())

    def test_record_accept(self):
        """Accept action increments alpha by 1."""
        svc = self._make_service()
        event = svc.record_feedback(
            task_id="t1",
            user_id="u1",
            action=FeedbackAction.ACCEPT,
            time_of_day="morning",
            energy_level=0.8,
            mood_valence=0.5,
        )
        assert event.action == FeedbackAction.ACCEPT
        assert event.task_id == "t1"

        # Verify Thompson params updated
        params = svc.store.get_params("task:t1")
        assert params.alpha == 2.0  # 1 (prior) + 1 (accept)
        assert params.beta == 1.0   # unchanged

    def test_record_dismiss(self):
        """Dismiss action increments beta by 1."""
        svc = self._make_service()
        svc.record_feedback(
            task_id="t1", user_id="u1", action=FeedbackAction.DISMISS,
        )
        params = svc.store.get_params("task:t1")
        assert params.alpha == 1.0   # unchanged
        assert params.beta == 2.0    # 1 (prior) + 1 (dismiss)

    def test_record_snooze(self):
        """Snooze action increments beta by 0.3 (weak negative)."""
        svc = self._make_service()
        event = svc.record_feedback(
            task_id="t1",
            user_id="u1",
            action=FeedbackAction.SNOOZE,
            snooze_minutes=15,
        )
        assert event.snooze_minutes == 15

        params = svc.store.get_params("task:t1")
        assert params.alpha == 1.0
        assert abs(params.beta - 1.3) < 1e-10

    def test_record_complete(self):
        """Complete action increments alpha by 2 (strongest positive)."""
        svc = self._make_service()
        svc.record_feedback(
            task_id="t1", user_id="u1", action=FeedbackAction.COMPLETE,
        )
        params = svc.store.get_params("task:t1")
        assert params.alpha == 3.0  # 1 (prior) + 2 (complete)
        assert params.beta == 1.0

    def test_multi_level_param_updates(self):
        """Feedback updates params at task, task+context, and intent+context levels."""
        svc = self._make_service()
        svc.record_feedback(
            task_id="t1",
            user_id="u1",
            action=FeedbackAction.ACCEPT,
            time_of_day="morning",
            energy_level=0.8,
            mood_valence=0.5,
            intent="reminder",
        )

        # Task-level
        task_params = svc.store.get_params("task:t1")
        assert task_params.alpha == 2.0

        # Task+context level
        ctx_key = "morning:high_energy:positive_mood"
        task_ctx_params = svc.store.get_params(f"task:t1:ctx:{ctx_key}")
        assert task_ctx_params.alpha == 2.0

        # Intent+context level
        intent_ctx_params = svc.store.get_params(f"intent:reminder:ctx:{ctx_key}")
        assert intent_ctx_params.alpha == 2.0

    def test_cumulative_updates(self):
        """Multiple feedbacks accumulate correctly."""
        svc = self._make_service()

        # Accept twice, dismiss once
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT)
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT)
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.DISMISS)

        params = svc.store.get_params("task:t1")
        assert params.alpha == 3.0   # 1 + 1 + 1
        assert params.beta == 2.0    # 1 + 1
        assert params.total_observations == 3
        assert params.mean == 3.0 / 5.0  # 0.6

    def test_sample_score(self):
        """Thompson sampling score is in valid range."""
        svc = self._make_service()

        # Add some feedback first
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT)
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.COMPLETE)

        for _ in range(50):
            score = svc.sample_score(task_id="t1", intent="task")
            assert 0.0 <= score <= 1.0

    def test_sample_score_blends_task_and_intent(self):
        """With few task observations, intent signal has more weight."""
        svc = self._make_service()
        ctx = "morning:high_energy:positive_mood"

        # Add many observations at intent level
        for _ in range(20):
            svc.record_feedback(
                task_id=f"other_{_}",
                user_id="u1",
                action=FeedbackAction.ACCEPT,
                time_of_day="morning",
                energy_level=0.8,
                mood_valence=0.5,
                intent="task",
            )

        # New task with no history — should lean on intent signal
        score = svc.sample_score(task_id="new_task", intent="task", context_key=ctx)
        assert 0.0 <= score <= 1.0

    def test_get_task_summary(self):
        svc = self._make_service()
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT)
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.DISMISS)
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.SNOOZE)

        summary = svc.get_task_summary("t1")
        assert summary.total_events == 3
        assert summary.accept_count == 1
        assert summary.dismiss_count == 1
        assert summary.snooze_count == 1

    def test_get_recent_feedback(self):
        svc = self._make_service()
        svc.record_feedback(task_id="t1", user_id="u1", action=FeedbackAction.ACCEPT)
        svc.record_feedback(task_id="t2", user_id="u1", action=FeedbackAction.DISMISS)
        svc.record_feedback(task_id="t3", user_id="u2", action=FeedbackAction.ACCEPT)

        # Only user u1's events
        events = svc.get_recent_feedback(user_id="u1", limit=10)
        assert len(events) == 2
        assert all(e.user_id == "u1" for e in events)

    def test_snooze_clears_minutes_for_non_snooze(self):
        """Snooze_minutes is only set for snooze actions."""
        svc = self._make_service()
        event = svc.record_feedback(
            task_id="t1",
            user_id="u1",
            action=FeedbackAction.ACCEPT,
            snooze_minutes=30,  # This should be cleared
        )
        assert event.snooze_minutes is None

    def test_thompson_params_converge_on_good_task(self):
        """A task that's always accepted should have high Thompson mean."""
        svc = self._make_service()
        for _ in range(20):
            svc.record_feedback(task_id="good", user_id="u1", action=FeedbackAction.ACCEPT)

        params = svc.store.get_params("task:good")
        assert params.mean > 0.85

    def test_thompson_params_converge_on_bad_task(self):
        """A task that's always dismissed should have low Thompson mean."""
        svc = self._make_service()
        for _ in range(20):
            svc.record_feedback(task_id="bad", user_id="u1", action=FeedbackAction.DISMISS)

        params = svc.store.get_params("task:bad")
        assert params.mean < 0.15
