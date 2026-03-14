"""Tests for PgFeedbackStore — Postgres feedback & Thompson params store.

Validates that Thompson Sampling beta parameters persist across simulated
server restarts (new store instance, same pool) and that all SQL uses
parameterized queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from blurt.persistence.feedback_store import PgFeedbackStore, _row_to_event, _row_to_thompson
from blurt.services.feedback import (
    FeedbackAction,
    FeedbackEvent,
    ThompsonParams,
)


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    return AsyncMock()


@pytest.fixture
def store(mock_pool):
    return PgFeedbackStore(mock_pool)


# ── Thompson Params Persistence ──────────────────────────────────────────


class TestThompsonParamsPersistence:
    """Verify beta parameters survive simulated restarts."""

    @pytest.mark.asyncio
    async def test_set_params_uses_upsert(self, store, mock_pool):
        """set_params_async persists alpha/beta via parameterized upsert."""
        now = datetime.now(timezone.utc)
        params = ThompsonParams(alpha=3.5, beta=2.1, total_observations=5, last_updated=now)

        mock_pool.execute.return_value = "INSERT 0 1"
        await store.set_params_async("task:abc123", params)

        mock_pool.execute.assert_called_once()
        sql = mock_pool.execute.call_args[0][0]
        # Verify parameterized query — no string interpolation
        assert "$1" in sql and "$2" in sql and "$3" in sql
        assert "INSERT INTO thompson_params" in sql
        assert "ON CONFLICT" in sql
        # Verify the actual parameter values
        args = mock_pool.execute.call_args[0]
        assert args[1] == "task:abc123"
        assert args[2] == 3.5  # alpha
        assert args[3] == 2.1  # beta
        assert args[4] == 5    # total_observations

    @pytest.mark.asyncio
    async def test_get_params_returns_persisted(self, store, mock_pool):
        """get_params_async retrieves previously persisted beta parameters."""
        now = datetime.now(timezone.utc)
        mock_pool.fetchrow.return_value = {
            "key": "task:abc123",
            "alpha": 3.5,
            "beta": 2.1,
            "total_observations": 5,
            "last_updated": now,
        }

        result = await store.get_params_async("task:abc123")

        assert result.alpha == 3.5
        assert result.beta == 2.1
        assert result.total_observations == 5
        assert result.last_updated == now

    @pytest.mark.asyncio
    async def test_get_params_creates_default_if_missing(self, store, mock_pool):
        """get_params_async creates uniform prior (1,1) when key doesn't exist."""
        mock_pool.fetchrow.return_value = None
        mock_pool.execute.return_value = "INSERT 0 1"

        result = await store.get_params_async("task:new-task")

        # Should have uniform prior
        assert result.alpha == 1.0
        assert result.beta == 1.0
        assert result.total_observations == 0
        # Should have inserted the default
        mock_pool.execute.assert_called_once()
        sql = mock_pool.execute.call_args[0][0]
        assert "INSERT INTO thompson_params" in sql
        assert "ON CONFLICT" in sql

    @pytest.mark.asyncio
    async def test_params_survive_new_store_instance(self, mock_pool):
        """Simulate restart: create store, set params, create NEW store, get params back."""
        now = datetime.now(timezone.utc)

        # --- "Before restart" ---
        store1 = PgFeedbackStore(mock_pool)
        params_to_save = ThompsonParams(alpha=5.2, beta=3.8, total_observations=9, last_updated=now)
        mock_pool.execute.return_value = "INSERT 0 1"
        await store1.set_params_async("task:persist-test", params_to_save)

        # --- "After restart" — new store instance, same pool ---
        store2 = PgFeedbackStore(mock_pool)
        mock_pool.fetchrow.return_value = {
            "key": "task:persist-test",
            "alpha": 5.2,
            "beta": 3.8,
            "total_observations": 9,
            "last_updated": now,
        }
        result = await store2.get_params_async("task:persist-test")

        assert result.alpha == 5.2
        assert result.beta == 3.8
        assert result.total_observations == 9
        assert result.last_updated == now

    @pytest.mark.asyncio
    async def test_multiple_keys_independent(self, store, mock_pool):
        """Different Thompson param keys are stored independently."""
        now = datetime.now(timezone.utc)

        # Simulate different params for different keys
        async def mock_fetchrow(sql, key):
            if key == "task:t1":
                return {"key": "task:t1", "alpha": 10.0, "beta": 2.0,
                        "total_observations": 12, "last_updated": now}
            elif key == "intent:reminder:ctx:morning":
                return {"key": "intent:reminder:ctx:morning", "alpha": 1.5, "beta": 4.0,
                        "total_observations": 5, "last_updated": now}
            return None

        mock_pool.fetchrow.side_effect = mock_fetchrow

        p1 = await store.get_params_async("task:t1")
        p2 = await store.get_params_async("intent:reminder:ctx:morning")

        assert p1.alpha == 10.0
        assert p1.beta == 2.0
        assert p2.alpha == 1.5
        assert p2.beta == 4.0


# ── Feedback Event Persistence ────────────────────────────────────────────


class TestFeedbackEventPersistence:
    """Verify feedback events persist correctly."""

    @pytest.mark.asyncio
    async def test_store_event_parameterized(self, store, mock_pool):
        """store_event_async uses parameterized INSERT."""
        event = FeedbackEvent(
            id="evt-1",
            task_id="task-1",
            user_id="user-1",
            action=FeedbackAction.ACCEPT,
            context_key="morning:high_energy:positive_mood",
            mood_valence=0.5,
            energy_level=0.8,
            time_of_day="morning",
            metadata={"source": "test"},
        )
        mock_pool.execute.return_value = "INSERT 0 1"
        await store.store_event_async(event)

        sql = mock_pool.execute.call_args[0][0]
        assert "$1" in sql
        assert "INSERT INTO feedback_events" in sql
        assert "ON CONFLICT (id) DO NOTHING" in sql
        # No string interpolation
        assert "evt-1" not in sql
        assert "task-1" not in sql

    @pytest.mark.asyncio
    async def test_get_events_with_filters(self, store, mock_pool):
        """get_events_async builds parameterized WHERE clauses."""
        now = datetime.now(timezone.utc)
        mock_pool.fetch.return_value = [
            {
                "id": "evt-1",
                "task_id": "task-1",
                "user_id": "user-1",
                "action": "accept",
                "timestamp": now,
                "context_key": "global",
                "mood_valence": 0.0,
                "energy_level": 0.5,
                "time_of_day": "",
                "snooze_minutes": None,
                "metadata": "{}",
            }
        ]

        result = await store.get_events_async(task_id="task-1", user_id="user-1", limit=10)

        assert len(result) == 1
        assert result[0].task_id == "task-1"
        assert result[0].action == FeedbackAction.ACCEPT
        # Verify parameterized query
        sql = mock_pool.fetch.call_args[0][0]
        assert "task_id = $1" in sql
        assert "user_id = $2" in sql
        assert "LIMIT $3" in sql


# ── Task Summary ──────────────────────────────────────────────────────────


class TestTaskSummary:
    """Verify task summary aggregation."""

    @pytest.mark.asyncio
    async def test_summary_empty_task(self, store, mock_pool):
        """Task with no events returns default summary."""
        mock_pool.fetchrow.side_effect = [
            # First call: aggregation query returns zero
            {"total_events": 0, "accept_count": 0, "dismiss_count": 0,
             "snooze_count": 0, "complete_count": 0, "last_feedback_at": None},
        ]

        result = await store.get_task_summary_async("empty-task")
        assert result.total_events == 0
        assert result.acceptance_rate == 0.0

    @pytest.mark.asyncio
    async def test_summary_with_events(self, store, mock_pool):
        """Task with events returns correct aggregation including thompson_mean."""
        now = datetime.now(timezone.utc)
        mock_pool.fetchrow.side_effect = [
            # Aggregation query
            {"total_events": 10, "accept_count": 5, "dismiss_count": 3,
             "snooze_count": 1, "complete_count": 1, "last_feedback_at": now},
            # Thompson params query (called by get_params_async internally)
            {"key": "task:t1", "alpha": 7.0, "beta": 4.3,
             "total_observations": 10, "last_updated": now},
        ]

        result = await store.get_task_summary_async("t1")
        assert result.total_events == 10
        assert result.accept_count == 5
        assert result.complete_count == 1
        assert result.acceptance_rate == 0.6  # (5+1)/10
        assert result.thompson_mean == round(7.0 / (7.0 + 4.3), 4)


# ── Row Conversion ────────────────────────────────────────────────────────


class TestRowConversion:
    """Test row-to-model conversion functions."""

    def test_row_to_thompson(self):
        now = datetime.now(timezone.utc)
        row = {"alpha": 2.5, "beta": 1.5, "total_observations": 4, "last_updated": now}
        result = _row_to_thompson(row)
        assert result.alpha == 2.5
        assert result.beta == 1.5
        assert result.total_observations == 4

    def test_row_to_event_with_json_string_metadata(self):
        now = datetime.now(timezone.utc)
        row = {
            "id": "e1", "task_id": "t1", "user_id": "u1",
            "action": "dismiss", "timestamp": now,
            "context_key": "global", "mood_valence": -0.2,
            "energy_level": 0.3, "time_of_day": "evening",
            "snooze_minutes": None,
            "metadata": '{"reason": "busy"}',
        }
        result = _row_to_event(row)
        assert result.action == FeedbackAction.DISMISS
        assert result.metadata == {"reason": "busy"}

    def test_row_to_event_with_dict_metadata(self):
        now = datetime.now(timezone.utc)
        row = {
            "id": "e2", "task_id": "t2", "user_id": "u2",
            "action": "complete", "timestamp": now,
            "context_key": "morning", "mood_valence": 0.8,
            "energy_level": 0.9, "time_of_day": "morning",
            "snooze_minutes": None,
            "metadata": {"note": "done"},
        }
        result = _row_to_event(row)
        assert result.action == FeedbackAction.COMPLETE
        assert result.metadata == {"note": "done"}


# ── Sync API Raises ───────────────────────────────────────────────────────


class TestSyncApiRaises:
    """Verify sync protocol stubs raise NotImplementedError."""

    def test_store_event_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.store_event(FeedbackEvent())

    def test_get_events_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.get_events()

    def test_get_params_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.get_params("key")

    def test_set_params_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.set_params("key", ThompsonParams())

    def test_get_task_summary_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.get_task_summary("task-id")
