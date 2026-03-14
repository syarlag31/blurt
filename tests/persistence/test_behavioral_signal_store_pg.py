"""Tests for PgBehavioralSignalStore — Postgres behavioral signal store.

Tests the async API methods against the SQL schema and row conversion logic.
Uses mock asyncpg pool to avoid requiring a real database connection.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from blurt.persistence.behavioral_signal_store import (
    PgBehavioralSignalStore,
    _row_to_signal,
)
from blurt.services.behavioral_signals import (
    BehavioralSignal,
    SignalContext,
    SignalKind,
)
from blurt.services.feedback import FeedbackAction


@pytest.fixture
def mock_pool():
    return AsyncMock()


@pytest.fixture
def store(mock_pool):
    return PgBehavioralSignalStore(mock_pool)


@pytest.fixture
def sample_signal():
    return BehavioralSignal(
        id=str(uuid.uuid4()),
        user_id="test-user",
        task_id="task-1",
        kind=SignalKind.REWARD,
        magnitude=1.0,
        reward_value=1.0,
        signal_contributions={"recency": 0.8, "urgency": 0.2},
        context=SignalContext(
            time_of_day="morning",
            energy_level=0.8,
            mood_valence=0.5,
            intent="task",
            cognitive_load=0.3,
            tags=["work", "coding"],
            entity_ids=["e1"],
        ),
        source_action=FeedbackAction.ACCEPT,
        source_event_id="ev-1",
        timestamp=datetime.now(timezone.utc),
        metadata={"source": "test"},
    )


class TestPgBehavioralSignalStoreAsync:
    """Test async methods of PgBehavioralSignalStore."""

    @pytest.mark.asyncio
    async def test_store_signal(self, store, mock_pool, sample_signal):
        mock_pool.execute.return_value = "INSERT 0 1"
        await store.store_signal_async(sample_signal)
        mock_pool.execute.assert_called_once()
        sql = mock_pool.execute.call_args[0][0]
        assert "INSERT INTO behavioral_signals" in sql
        assert "$1" in sql
        # Verify all 12 parameters are passed
        args = mock_pool.execute.call_args[0]
        assert len(args) == 13  # SQL + 12 params

    @pytest.mark.asyncio
    async def test_get_signals_no_since(self, store, mock_pool):
        mock_pool.fetch.return_value = []
        result = await store.get_signals_async("user1")
        assert result == []
        sql = mock_pool.fetch.call_args[0][0]
        assert "timestamp >=" not in sql

    @pytest.mark.asyncio
    async def test_get_signals_with_since(self, store, mock_pool):
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_pool.fetch.return_value = []
        result = await store.get_signals_async("user1", since=since)
        assert result == []
        sql = mock_pool.fetch.call_args[0][0]
        assert "timestamp >=" in sql

    @pytest.mark.asyncio
    async def test_get_interaction_count_no_action(self, store, mock_pool):
        mock_pool.fetchrow.return_value = {"cnt": 5}
        result = await store.get_interaction_count_async("u1", "t1")
        assert result == 5

    @pytest.mark.asyncio
    async def test_get_interaction_count_with_action(self, store, mock_pool):
        mock_pool.fetchrow.return_value = {"cnt": 2}
        result = await store.get_interaction_count_async("u1", "t1", FeedbackAction.DISMISS)
        assert result == 2
        sql = mock_pool.fetchrow.call_args[0][0]
        assert "source_action = $3" in sql


class TestRowToSignal:
    """Test the row-to-signal conversion logic."""

    def test_basic_conversion(self):
        now = datetime.now(timezone.utc)
        row = {
            "id": "sig-1",
            "user_id": "u1",
            "task_id": "t1",
            "kind": "reward",
            "magnitude": 1.5,
            "reward_value": 1.5,
            "signal_contributions": {"recency": 0.9},
            "context": {
                "time_of_day": "afternoon",
                "energy_level": 0.7,
                "mood_valence": 0.3,
                "intent": "event",
                "cognitive_load": 0.5,
                "tags": ["meeting"],
                "entity_ids": [],
            },
            "source_action": "accept",
            "source_event_id": "ev-1",
            "timestamp": now,
            "metadata": {},
        }
        signal = _row_to_signal(row)
        assert signal.id == "sig-1"
        assert signal.kind == SignalKind.REWARD
        assert signal.source_action == FeedbackAction.ACCEPT
        assert signal.context.time_of_day == "afternoon"
        assert signal.context.tags == ["meeting"]

    def test_handles_string_jsonb(self):
        now = datetime.now(timezone.utc)
        row = {
            "id": "sig-2",
            "user_id": "u1",
            "task_id": "t1",
            "kind": "penalty",
            "magnitude": 0.5,
            "reward_value": -0.5,
            "signal_contributions": json.dumps({"urgency": 0.4}),
            "context": json.dumps({"time_of_day": "night", "energy_level": 0.2}),
            "source_action": "dismiss",
            "source_event_id": "ev-2",
            "timestamp": now,
            "metadata": json.dumps({"extra": "data"}),
        }
        signal = _row_to_signal(row)
        assert signal.kind == SignalKind.PENALTY
        assert signal.signal_contributions == {"urgency": 0.4}
        assert signal.metadata == {"extra": "data"}

    def test_naive_timestamp_gets_utc(self):
        naive_dt = datetime(2025, 6, 15, 10, 30, 0)
        row = {
            "id": "sig-3",
            "user_id": "u1",
            "task_id": "t1",
            "kind": "neutral",
            "magnitude": 0.0,
            "reward_value": 0.0,
            "signal_contributions": {},
            "context": {},
            "source_action": "dismiss",
            "source_event_id": "",
            "timestamp": naive_dt,
            "metadata": {},
        }
        signal = _row_to_signal(row)
        assert signal.timestamp.tzinfo == timezone.utc


class TestSyncApiRaises:
    """Verify sync wrappers raise NotImplementedError."""

    def test_store_signal_sync_raises(self, store, sample_signal):
        with pytest.raises(NotImplementedError):
            store.store_signal(sample_signal)

    def test_get_signals_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.get_signals("u1")

    def test_get_interaction_count_sync_raises(self, store):
        with pytest.raises(NotImplementedError):
            store.get_interaction_count("u1", "t1")
