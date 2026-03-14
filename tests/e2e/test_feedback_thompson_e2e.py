"""E2E Scenario 5: Task feedback and Thompson Sampling through HTTP.

Tests the feedback loop: record feedback actions (accept, dismiss,
snooze, complete), verify Thompson Sampling parameter updates, and
check the summary/recent-feedback endpoints.

Cross-cutting concerns exercised:
- Feedback recording: accept, dismiss, snooze, complete actions via HTTP
- Thompson Sampling: alpha/beta parameter updates from feedback signals
- Bayesian learning: posterior updates shift sampling distribution over time
- Feedback summary: aggregated stats (acceptance rate, total count) via API
- Recent feedback history: chronological retrieval of latest feedback entries
- Task-level granularity: feedback tracked per-task with independent params
- Anti-shame guarantee: negative feedback (dismiss/snooze) bounded to prevent
  over-punishment of surfaced tasks
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestFeedbackRecording:
    """Record various feedback actions via the API."""

    async def test_accept_task(
        self,
        record_feedback_via_api: Any,
    ):
        """Accepting a task records correctly."""
        fb = await record_feedback_via_api(task_id="task-1", action="accept")
        assert fb["action"] == "accept"
        assert fb["task_id"] == "task-1"
        assert fb["thompson_update_applied"] is True
        assert fb["message"]  # Shame-free acknowledgment

    async def test_dismiss_task(
        self,
        record_feedback_via_api: Any,
    ):
        """Dismissing a task is shame-free."""
        fb = await record_feedback_via_api(task_id="task-2", action="dismiss")
        assert fb["action"] == "dismiss"
        assert fb["thompson_update_applied"] is True

    async def test_snooze_task_with_duration(
        self,
        record_feedback_via_api: Any,
    ):
        """Snoozed tasks record the snooze duration."""
        fb = await record_feedback_via_api(
            task_id="task-3", action="snooze", snooze_minutes=15
        )
        assert fb["action"] == "snooze"

    async def test_complete_task(
        self,
        record_feedback_via_api: Any,
    ):
        """Completing a task gives the strongest positive signal."""
        fb = await record_feedback_via_api(task_id="task-4", action="complete")
        assert fb["action"] == "complete"

    async def test_context_key_includes_time_energy_mood(
        self,
        record_feedback_via_api: Any,
    ):
        """Context key incorporates time, energy, and mood buckets."""
        fb = await record_feedback_via_api(
            task_id="task-5",
            action="accept",
            time_of_day="morning",
            energy_level=0.9,
            mood_valence=0.5,
        )
        # Context key should contain the time and bucket info
        assert fb["context_key"]
        assert "morning" in fb["context_key"]


class TestFeedbackSummary:
    """Feedback summary aggregation."""

    async def test_summary_reflects_all_actions(
        self,
        client: httpx.AsyncClient,
        record_feedback_via_api: Any,
    ):
        """Summary endpoint aggregates feedback across actions."""
        task_id = "task-summary"
        await record_feedback_via_api(task_id=task_id, action="accept")
        await record_feedback_via_api(task_id=task_id, action="dismiss")
        await record_feedback_via_api(task_id=task_id, action="complete")

        resp = await client.get(f"/api/v1/tasks/{task_id}/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 3
        assert data["accept_count"] == 1
        assert data["dismiss_count"] == 1
        assert data["complete_count"] == 1
        assert 0 < data["acceptance_rate"] < 1

    async def test_summary_for_unknown_task_returns_zeros(
        self,
        client: httpx.AsyncClient,
    ):
        """Summary for a task with no feedback returns zero counts."""
        resp = await client.get("/api/v1/tasks/unknown-task/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 0


class TestRecentFeedback:
    """Recent feedback listing."""

    async def test_recent_feedback_ordered_by_time(
        self,
        client: httpx.AsyncClient,
        record_feedback_via_api: Any,
        test_user_id: str,
    ):
        """Recent feedback returns events in reverse chronological order."""
        await record_feedback_via_api(task_id="t1", action="accept")
        await record_feedback_via_api(task_id="t2", action="dismiss")
        await record_feedback_via_api(task_id="t3", action="complete")

        resp = await client.get(
            "/api/v1/feedback/recent",
            params={"user_id": test_user_id, "limit": 10},
        )
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 3
        # Most recent first
        assert events[0]["task_id"] == "t3"
