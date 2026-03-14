"""E2E Scenario 8: Thompson Sampling — cold start, learning convergence,
preference shift adaptation, and anti-shame guarantees.

Exercises the full feedback loop through HTTP endpoints:
POST /api/v1/tasks/{id}/feedback → TaskFeedbackService → Thompson Sampling
parameter updates → GET feedback summary verification.

Validates:
1. Cold start exploration: uniform priors produce exploration, not exploitation.
2. Learning convergence: repeated positive feedback raises Thompson mean.
3. Preference shift adaptation: new feedback overrides old signals.
4. Anti-shame guarantees: dismiss/snooze never punish—params stay bounded.
5. Contextual learning: different time/energy contexts track independently.
6. Multi-granularity updates: task-level, context-level, and intent-level params.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from blurt.services.feedback import InMemoryFeedbackStore


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _record(
    client: httpx.AsyncClient,
    task_id: str,
    action: str,
    user_id: str = "e2e-test-user",
    mood_valence: float = 0.0,
    energy_level: float = 0.5,
    time_of_day: str = "morning",
    intent: str = "task",
    snooze_minutes: int | None = None,
) -> dict[str, Any]:
    """Record feedback via the HTTP endpoint and return the JSON response."""
    payload: dict[str, Any] = {
        "user_id": user_id,
        "action": action,
        "mood_valence": mood_valence,
        "energy_level": energy_level,
        "time_of_day": time_of_day,
        "intent": intent,
    }
    if snooze_minutes is not None:
        payload["snooze_minutes"] = snooze_minutes
    resp = await client.post(f"/api/v1/tasks/{task_id}/feedback", json=payload)
    assert resp.status_code == 201, f"Feedback recording failed: {resp.text}"
    return resp.json()


async def _summary(
    client: httpx.AsyncClient,
    task_id: str,
) -> dict[str, Any]:
    """Fetch feedback summary for a task via GET endpoint."""
    resp = await client.get(f"/api/v1/tasks/{task_id}/feedback")
    assert resp.status_code == 200, f"Summary fetch failed: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# Scenario 8a: Cold Start Exploration
# ---------------------------------------------------------------------------


class TestColdStartExploration:
    """Validates that a brand-new user sees exploration, not exploitation."""

    async def test_cold_start_uniform_priors(
        self,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Before any feedback, Thompson params are at the uniform prior (1, 1)."""
        params = feedback_store.get_params("task:brand-new-task")
        assert params.alpha == 1.0
        assert params.beta == 1.0
        assert params.mean == pytest.approx(0.5, abs=0.01)
        assert params.total_observations == 0

    async def test_cold_start_high_variance(
        self,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Uniform prior has high variance, encouraging exploration."""
        params = feedback_store.get_params("task:explore-me")
        # Beta(1,1) variance = 1*1 / (4*3) = 1/12 ≈ 0.0833
        assert params.variance == pytest.approx(1.0 / 12.0, abs=0.001)

    async def test_first_feedback_shifts_from_uniform(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """A single accept moves the params away from uniform."""
        await _record(client, "cold-task-1", "accept")

        params = feedback_store.get_params("task:cold-task-1")
        assert params.alpha > 1.0, "Accept should increase alpha"
        assert params.beta == 1.0, "Accept should not touch beta"
        assert params.mean > 0.5, "Mean should shift above 0.5"
        assert params.total_observations == 1

    async def test_cold_start_summary_reflects_no_history(
        self,
        client: httpx.AsyncClient,
    ):
        """GET summary for an unseen task shows zero events and 0.5 mean."""
        summary = await _summary(client, "never-seen-task")
        assert summary["total_events"] == 0
        assert summary["accept_count"] == 0
        assert summary["dismiss_count"] == 0
        assert summary["thompson_mean"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Scenario 8b: Learning Convergence
# ---------------------------------------------------------------------------


class TestLearningConvergence:
    """Validates that consistent positive feedback raises the Thompson mean."""

    async def test_repeated_accepts_raise_mean(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """10 consecutive accepts should push the mean well above 0.5."""
        task_id = "converge-accept"
        for _ in range(10):
            await _record(client, task_id, "accept")

        params = feedback_store.get_params(f"task:{task_id}")
        assert params.alpha == pytest.approx(11.0, abs=0.1)  # 1 + 10
        assert params.beta == pytest.approx(1.0, abs=0.1)
        assert params.mean > 0.85, f"Mean should converge high: {params.mean}"

    async def test_repeated_completes_converge_faster(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Complete gives alpha +2, so 5 completes = alpha 11 (same as 10 accepts)."""
        task_id = "converge-complete"
        for _ in range(5):
            await _record(client, task_id, "complete")

        params = feedback_store.get_params(f"task:{task_id}")
        # complete: alpha += 2, so after 5: alpha = 1 + 10 = 11
        assert params.alpha == pytest.approx(11.0, abs=0.1)
        assert params.mean > 0.85

    async def test_mixed_feedback_settles_to_ratio(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """6 accepts + 4 dismisses should give a mean near 7/12 ≈ 0.58."""
        task_id = "converge-mixed"
        for _ in range(6):
            await _record(client, task_id, "accept")
        for _ in range(4):
            await _record(client, task_id, "dismiss")

        params = feedback_store.get_params(f"task:{task_id}")
        # alpha = 1 + 6 = 7, beta = 1 + 4 = 5 → mean = 7/12 ≈ 0.583
        expected_mean = 7.0 / 12.0
        assert params.mean == pytest.approx(expected_mean, abs=0.05)

    async def test_summary_counts_match(
        self,
        client: httpx.AsyncClient,
    ):
        """GET summary accurately reflects event counts after a sequence."""
        task_id = "converge-counts"
        for _ in range(3):
            await _record(client, task_id, "accept")
        for _ in range(2):
            await _record(client, task_id, "dismiss")
        await _record(client, task_id, "complete")

        summary = await _summary(client, task_id)
        assert summary["total_events"] == 6
        assert summary["accept_count"] == 3
        assert summary["dismiss_count"] == 2
        assert summary["complete_count"] == 1
        assert summary["snooze_count"] == 0
        # acceptance_rate = (3 + 1) / 6 ≈ 0.6667
        assert summary["acceptance_rate"] == pytest.approx(4.0 / 6.0, abs=0.01)

    async def test_variance_decreases_with_observations(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Variance should decrease as more observations are recorded."""
        task_id = "converge-variance"
        params_before = feedback_store.get_params(f"task:{task_id}")
        initial_variance = params_before.variance

        for _ in range(20):
            await _record(client, task_id, "accept")

        params_after = feedback_store.get_params(f"task:{task_id}")
        assert params_after.variance < initial_variance, (
            "Variance should decrease with more observations"
        )


# ---------------------------------------------------------------------------
# Scenario 8c: Preference Shift Adaptation
# ---------------------------------------------------------------------------


class TestPreferenceShiftAdaptation:
    """Validates the model adapts when user preferences change."""

    async def test_shift_from_accept_to_dismiss(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """After a run of accepts, dismissals should lower the mean."""
        task_id = "shift-accept-dismiss"

        # Phase 1: 8 accepts → high mean
        for _ in range(8):
            await _record(client, task_id, "accept")
        params_phase1 = feedback_store.get_params(f"task:{task_id}")
        mean_after_accepts = params_phase1.mean
        assert mean_after_accepts > 0.8

        # Phase 2: 8 dismissals → mean should drop
        for _ in range(8):
            await _record(client, task_id, "dismiss")
        params_phase2 = feedback_store.get_params(f"task:{task_id}")
        mean_after_dismissals = params_phase2.mean
        assert mean_after_dismissals < mean_after_accepts, (
            f"Mean should drop: {mean_after_dismissals} < {mean_after_accepts}"
        )

    async def test_shift_from_dismiss_to_accept(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """After a run of dismissals, accepts should raise the mean."""
        task_id = "shift-dismiss-accept"

        # Phase 1: 6 dismissals → low mean
        for _ in range(6):
            await _record(client, task_id, "dismiss")
        mean_low = feedback_store.get_params(f"task:{task_id}").mean
        assert mean_low < 0.5

        # Phase 2: 6 accepts → mean should rise
        for _ in range(6):
            await _record(client, task_id, "accept")
        mean_high = feedback_store.get_params(f"task:{task_id}").mean
        assert mean_high > mean_low, "Mean should rise after accepts"

    async def test_context_shift_morning_vs_evening(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """User accepts tasks in morning but dismisses in evening —
        contextual params diverge."""
        task_id = "shift-time-ctx"

        # Morning: accept
        for _ in range(5):
            await _record(
                client, task_id, "accept",
                time_of_day="morning", energy_level=0.8,
            )

        # Evening: dismiss
        for _ in range(5):
            await _record(
                client, task_id, "dismiss",
                time_of_day="evening", energy_level=0.3,
            )

        # Context keys differ, so check intent-level params diverge
        morning_key = "intent:task:ctx:morning:high_energy:neutral_mood"
        evening_key = "intent:task:ctx:evening:low_energy:neutral_mood"

        morning_params = feedback_store.get_params(morning_key)
        evening_params = feedback_store.get_params(evening_key)

        assert morning_params.mean > evening_params.mean, (
            f"Morning mean ({morning_params.mean}) should be higher "
            f"than evening ({evening_params.mean})"
        )


# ---------------------------------------------------------------------------
# Scenario 8d: Anti-Shame Guarantees
# ---------------------------------------------------------------------------


class TestAntiShameGuarantees:
    """Validates that dismiss/snooze never punish and stay bounded."""

    async def test_dismiss_response_is_shame_free(
        self,
        client: httpx.AsyncClient,
    ):
        """The API response for dismiss contains no shame language."""
        result = await _record(client, "shame-task-1", "dismiss")
        assert result["thompson_update_applied"] is True
        msg = result["message"].lower()
        # Must not contain guilt/shame language
        for bad_word in ["overdue", "lazy", "haven't", "failed", "missed"]:
            assert bad_word not in msg, f"Shame word '{bad_word}' found in: {msg}"

    async def test_snooze_response_is_shame_free(
        self,
        client: httpx.AsyncClient,
    ):
        """The API response for snooze contains no shame language."""
        result = await _record(
            client, "shame-task-2", "snooze", snooze_minutes=30,
        )
        msg = result["message"].lower()
        for bad_word in ["overdue", "lazy", "haven't", "failed", "missed"]:
            assert bad_word not in msg, f"Shame word '{bad_word}' found in: {msg}"

    async def test_many_dismissals_never_zero_mean(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Even after many dismissals, the Thompson mean never reaches 0."""
        task_id = "antishame-many-dismiss"
        for _ in range(50):
            await _record(client, task_id, "dismiss")

        params = feedback_store.get_params(f"task:{task_id}")
        assert params.mean > 0.0, "Mean must never hit zero"
        # With alpha=1, beta=51: mean = 1/52 ≈ 0.019 — positive, not zero
        assert params.alpha >= 1.0, "Alpha must stay at or above the prior"

    async def test_snooze_gives_partial_positive(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Snooze adds a weak beta (0.3) but does NOT increase alpha,
        so it's a mild negative — never as harsh as dismiss."""
        task_id = "antishame-snooze"
        await _record(client, task_id, "snooze", snooze_minutes=15)

        params = feedback_store.get_params(f"task:{task_id}")
        # Snooze: alpha_delta=0.0, beta_delta=0.3
        assert params.beta == pytest.approx(1.3, abs=0.01)
        # Mean after snooze should still be close to 0.5 (not punished hard)
        assert params.mean > 0.4, f"Snooze mean too low: {params.mean}"

    async def test_dismiss_weaker_than_reject_pattern(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Dismiss adds beta +1.0. After one dismiss, mean = 1/(1+2) = 0.333.
        This is gentle — still within explorable range."""
        task_id = "antishame-dismiss-gentle"
        await _record(client, task_id, "dismiss")

        params = feedback_store.get_params(f"task:{task_id}")
        assert params.mean == pytest.approx(1.0 / 3.0, abs=0.02)
        # Still explorable — not crushed
        assert params.mean > 0.3

    async def test_all_actions_produce_valid_response(
        self,
        client: httpx.AsyncClient,
    ):
        """Every action type returns 201 with a valid, non-empty message."""
        for action in ["accept", "dismiss", "snooze", "complete"]:
            kwargs: dict[str, Any] = {}
            if action == "snooze":
                kwargs["snooze_minutes"] = 15
            result = await _record(client, f"valid-{action}", action, **kwargs)
            assert result["action"] == action
            assert result["thompson_update_applied"] is True
            assert len(result["message"]) > 0, f"Empty message for {action}"


# ---------------------------------------------------------------------------
# Scenario 8e: Contextual Multi-Granularity Updates
# ---------------------------------------------------------------------------


class TestContextualMultiGranularity:
    """Validates that feedback updates params at task, context, and intent levels."""

    async def test_three_param_keys_updated(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """A single accept should update params at 3 keys:
        task:{id}, task:{id}:ctx:{ctx}, intent:{intent}:ctx:{ctx}."""
        task_id = "multi-gran-1"
        await _record(
            client, task_id, "accept",
            time_of_day="morning", energy_level=0.8, mood_valence=0.5,
            intent="reminder",
        )

        ctx_key = "morning:high_energy:positive_mood"
        key_task = f"task:{task_id}"
        key_task_ctx = f"task:{task_id}:ctx:{ctx_key}"
        key_intent_ctx = f"intent:reminder:ctx:{ctx_key}"

        for key in [key_task, key_task_ctx, key_intent_ctx]:
            params = feedback_store.get_params(key)
            assert params.total_observations >= 1, f"Key {key} not updated"
            assert params.alpha > 1.0, f"Key {key} alpha not incremented"

    async def test_intent_level_generalizes_across_tasks(
        self,
        client: httpx.AsyncClient,
        feedback_store: InMemoryFeedbackStore,
    ):
        """Two different tasks with the same intent and context should both
        contribute to the intent-level params."""
        for task_id in ["gen-task-a", "gen-task-b", "gen-task-c"]:
            await _record(
                client, task_id, "accept",
                time_of_day="afternoon", energy_level=0.5, mood_valence=0.0,
                intent="idea",
            )

        ctx_key = "afternoon:med_energy:neutral_mood"
        intent_key = f"intent:idea:ctx:{ctx_key}"
        params = feedback_store.get_params(intent_key)
        # 3 accepts → alpha = 1 + 3 = 4
        assert params.alpha == pytest.approx(4.0, abs=0.1)
        assert params.total_observations == 3

    async def test_recent_feedback_returns_history(
        self,
        client: httpx.AsyncClient,
    ):
        """GET /api/v1/feedback/recent returns the recorded events."""
        user_id = "e2e-test-user"
        for i in range(3):
            await _record(client, f"recent-{i}", "accept", user_id=user_id)

        resp = await client.get(
            "/api/v1/feedback/recent",
            params={"user_id": user_id, "limit": 10},
        )
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) >= 3
        # Most recent first
        task_ids = [e["task_id"] for e in events[:3]]
        assert "recent-2" in task_ids
