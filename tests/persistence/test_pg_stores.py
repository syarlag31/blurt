"""Unit tests for Postgres store implementations (task, feedback, pattern).

Uses a mock asyncpg pool to verify SQL correctness and data mapping
without requiring a real database connection.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.models.entities import LearnedPattern, PatternType
from blurt.persistence.feedback_store import PgFeedbackStore, _row_to_event
from blurt.persistence.pattern_store import PgPatternStore, _row_to_pattern
from blurt.persistence.task_store import PgTaskStore, _row_to_task, _task_to_row
from blurt.services.feedback import (
    FeedbackAction,
    FeedbackEvent,
    ThompsonParams,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
    TaskStatus,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_pool() -> MagicMock:
    """Create a mock asyncpg pool with async context manager support."""
    pool = MagicMock()
    pool.execute = AsyncMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    return pool


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_task_row(**overrides) -> dict:
    """Create a dict that simulates an asyncpg Record for tasks."""
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "__default__",
        "content": "Test task",
        "status": "active",
        "intent": "task",
        "created_at": _utcnow(),
        "due_at": None,
        "last_mentioned_at": None,
        "estimated_energy": "medium",
        "estimated_duration_minutes": None,
        "entity_ids": [],
        "entity_names": [],
        "project": None,
        "capture_valence": 0.0,
        "capture_arousal": 0.5,
        "times_surfaced": 0,
        "times_deferred": 0,
        "metadata": "{}",
    }
    row.update(overrides)
    return row


def _make_feedback_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for feedback events."""
    row = {
        "id": str(uuid.uuid4()),
        "task_id": str(uuid.uuid4()),
        "user_id": "user-1",
        "action": "accept",
        "timestamp": _utcnow(),
        "context_key": "morning:high_energy:positive_mood",
        "mood_valence": 0.5,
        "energy_level": 0.8,
        "time_of_day": "morning",
        "snooze_minutes": None,
        "metadata": "{}",
    }
    row.update(overrides)
    return row


def _make_pattern_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for patterns."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "user-1",
        "pattern_type": "energy_rhythm",
        "description": "High energy in mornings",
        "parameters": '{"time_of_day": "morning"}',
        "confidence": 0.75,
        "observation_count": 5,
        "supporting_evidence": ["ev1", "ev2"],
        "embedding": None,
        "is_active": True,
        "first_detected": now,
        "last_confirmed": now,
        "created_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


# ── Row-to-model conversion tests ───────────────────────────────────


class TestRowConversions:
    """Test data mapping between database rows and Python models."""

    def test_row_to_task_basic(self):
        row = _make_task_row()
        task = _row_to_task(row)
        assert isinstance(task, SurfaceableTask)
        assert task.status == TaskStatus.ACTIVE
        assert task.estimated_energy == EnergyLevel.MEDIUM
        assert task.metadata == {}

    def test_row_to_task_with_json_metadata(self):
        row = _make_task_row(metadata='{"tags": ["work", "urgent"]}')
        task = _row_to_task(row)
        assert task.metadata == {"tags": ["work", "urgent"]}

    def test_row_to_task_with_dict_metadata(self):
        row = _make_task_row(metadata={"foo": "bar"})
        task = _row_to_task(row)
        assert task.metadata == {"foo": "bar"}

    def test_row_to_task_with_entities(self):
        row = _make_task_row(
            entity_ids=["e1", "e2"],
            entity_names=["Alice", "Bob"],
        )
        task = _row_to_task(row)
        assert task.entity_ids == ["e1", "e2"]
        assert task.entity_names == ["Alice", "Bob"]

    def test_task_to_row_roundtrip(self):
        task = SurfaceableTask(
            content="Buy groceries",
            status=TaskStatus.ACTIVE,
            intent="task",
            estimated_energy=EnergyLevel.LOW,
            entity_ids=["e1"],
            entity_names=["Store"],
            metadata={"tags": ["errands"]},
        )
        row = _task_to_row(task, "user-1")
        assert row["content"] == "Buy groceries"
        assert row["status"] == "active"
        assert row["estimated_energy"] == "low"
        assert row["user_id"] == "user-1"
        assert json.loads(row["metadata"]) == {"tags": ["errands"]}

    def test_row_to_event_basic(self):
        row = _make_feedback_row()
        event = _row_to_event(row)
        assert isinstance(event, FeedbackEvent)
        assert event.action == FeedbackAction.ACCEPT

    def test_row_to_event_with_snooze(self):
        row = _make_feedback_row(action="snooze", snooze_minutes=30)
        event = _row_to_event(row)
        assert event.action == FeedbackAction.SNOOZE
        assert event.snooze_minutes == 30

    def test_row_to_pattern_basic(self):
        row = _make_pattern_row()
        pattern = _row_to_pattern(row)
        assert isinstance(pattern, LearnedPattern)
        assert pattern.pattern_type == PatternType.ENERGY_RHYTHM
        assert pattern.confidence == 0.75
        assert pattern.parameters == {"time_of_day": "morning"}

    def test_row_to_pattern_with_embedding(self):
        embedding_str = "[0.1,0.2,0.3]"
        row = _make_pattern_row(embedding=embedding_str)
        pattern = _row_to_pattern(row)
        assert pattern.embedding == [0.1, 0.2, 0.3]


# ── PgTaskStore tests ────────────────────────────────────────────────


class TestPgTaskStore:
    """Test PgTaskStore async CRUD methods."""

    @pytest.fixture
    def pool(self):
        return _make_pool()

    @pytest.fixture
    def store(self, pool):
        return PgTaskStore(pool)

    @pytest.mark.asyncio
    async def test_add_task_async_calls_execute(self, store, pool):
        task = SurfaceableTask(content="Do laundry")
        await store.add_task_async(task, user_id="user-1")
        pool.execute.assert_awaited_once()
        call_args = pool.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO tasks" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_get_task_async_found(self, store, pool):
        row = _make_task_row(id="task-1", content="Found task")
        pool.fetchrow = AsyncMock(return_value=row)
        result = await store.get_task_async("task-1", user_id="__default__")
        assert result is not None
        assert result.content == "Found task"

    @pytest.mark.asyncio
    async def test_get_task_async_not_found(self, store, pool):
        pool.fetchrow = AsyncMock(return_value=None)
        result = await store.get_task_async("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_tasks_async(self, store, pool):
        rows = [_make_task_row(content="Task 1"), _make_task_row(content="Task 2")]
        pool.fetch = AsyncMock(return_value=rows)
        result = await store.get_all_tasks_async(user_id="user-1")
        assert len(result) == 2
        pool.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_task_async_deleted(self, store, pool):
        pool.execute = AsyncMock(return_value="DELETE 1")
        result = await store.remove_task_async("task-1", user_id="user-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_remove_task_async_not_found(self, store, pool):
        pool.execute = AsyncMock(return_value="DELETE 0")
        result = await store.remove_task_async("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_async_user(self, store, pool):
        await store.clear_async(user_id="user-1")
        pool.execute.assert_awaited_once()
        sql = pool.execute.call_args[0][0]
        assert "DELETE FROM tasks WHERE user_id = $1" in sql

    @pytest.mark.asyncio
    async def test_clear_async_all(self, store, pool):
        await store.clear_async()
        pool.execute.assert_awaited_once()
        sql = pool.execute.call_args[0][0]
        assert "DELETE FROM tasks" in sql

    def test_sync_methods_raise(self, store):
        task = SurfaceableTask(content="test")
        with pytest.raises(NotImplementedError):
            store.add_task(task)
        with pytest.raises(NotImplementedError):
            store.get_task("id")
        with pytest.raises(NotImplementedError):
            store.get_all_tasks()
        with pytest.raises(NotImplementedError):
            store.update_task(task)
        with pytest.raises(NotImplementedError):
            store.remove_task("id")


# ── PgFeedbackStore tests ────────────────────────────────────────────


class TestPgFeedbackStore:
    """Test PgFeedbackStore async CRUD methods."""

    @pytest.fixture
    def pool(self):
        return _make_pool()

    @pytest.fixture
    def store(self, pool):
        return PgFeedbackStore(pool)

    @pytest.mark.asyncio
    async def test_store_event_async(self, store, pool):
        event = FeedbackEvent(
            task_id="task-1",
            user_id="user-1",
            action=FeedbackAction.ACCEPT,
        )
        await store.store_event_async(event)
        pool.execute.assert_awaited_once()
        sql = pool.execute.call_args[0][0]
        assert "INSERT INTO feedback_events" in sql

    @pytest.mark.asyncio
    async def test_get_events_async_no_filters(self, store, pool):
        rows = [_make_feedback_row(), _make_feedback_row()]
        pool.fetch = AsyncMock(return_value=rows)
        result = await store.get_events_async(limit=50)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_events_async_with_task_filter(self, store, pool):
        pool.fetch = AsyncMock(return_value=[])
        await store.get_events_async(task_id="task-1")
        sql = pool.fetch.call_args[0][0]
        assert "task_id = $1" in sql

    @pytest.mark.asyncio
    async def test_get_events_async_with_both_filters(self, store, pool):
        pool.fetch = AsyncMock(return_value=[])
        await store.get_events_async(task_id="task-1", user_id="user-1")
        sql = pool.fetch.call_args[0][0]
        assert "task_id = $1" in sql
        assert "user_id = $2" in sql

    @pytest.mark.asyncio
    async def test_get_params_async_creates_default(self, store, pool):
        pool.fetchrow = AsyncMock(return_value=None)
        pool.execute = AsyncMock()
        result = await store.get_params_async("test-key")
        assert result.alpha == 1.0
        assert result.beta == 1.0
        assert result.total_observations == 0
        # Should have inserted default params
        pool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_params_async_existing(self, store, pool):
        now = _utcnow()
        pool.fetchrow = AsyncMock(
            return_value={
                "key": "test-key",
                "alpha": 3.0,
                "beta": 2.0,
                "total_observations": 5,
                "last_updated": now,
            }
        )
        result = await store.get_params_async("test-key")
        assert result.alpha == 3.0
        assert result.beta == 2.0
        assert result.total_observations == 5

    @pytest.mark.asyncio
    async def test_set_params_async(self, store, pool):
        params = ThompsonParams(alpha=5.0, beta=3.0, total_observations=8)
        await store.set_params_async("task:123", params)
        pool.execute.assert_awaited_once()
        sql = pool.execute.call_args[0][0]
        assert "INSERT INTO thompson_params" in sql
        assert "ON CONFLICT (key) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_get_task_summary_async_empty(self, store, pool):
        pool.fetchrow = AsyncMock(
            side_effect=[
                # First call: aggregation query returns zero events
                {
                    "total_events": 0,
                    "accept_count": 0,
                    "dismiss_count": 0,
                    "snooze_count": 0,
                    "complete_count": 0,
                    "last_feedback_at": None,
                },
            ]
        )
        result = await store.get_task_summary_async("task-1")
        assert result.total_events == 0
        assert result.task_id == "task-1"

    @pytest.mark.asyncio
    async def test_get_task_summary_async_with_data(self, store, pool):
        now = _utcnow()
        pool.fetchrow = AsyncMock(
            side_effect=[
                # First call: aggregation query
                {
                    "total_events": 10,
                    "accept_count": 5,
                    "dismiss_count": 2,
                    "snooze_count": 2,
                    "complete_count": 1,
                    "last_feedback_at": now,
                },
                # Second call: get_params_async -> thompson_params lookup
                {
                    "key": "task:task-1",
                    "alpha": 7.0,
                    "beta": 3.3,
                    "total_observations": 10,
                    "last_updated": now,
                },
            ]
        )
        result = await store.get_task_summary_async("task-1")
        assert result.total_events == 10
        assert result.accept_count == 5
        assert result.acceptance_rate == 0.6
        assert result.thompson_mean == round(7.0 / (7.0 + 3.3), 4)

    def test_sync_methods_raise(self, store):
        with pytest.raises(NotImplementedError):
            store.store_event(FeedbackEvent())
        with pytest.raises(NotImplementedError):
            store.get_events()
        with pytest.raises(NotImplementedError):
            store.get_params("key")
        with pytest.raises(NotImplementedError):
            store.set_params("key", ThompsonParams())
        with pytest.raises(NotImplementedError):
            store.get_task_summary("task-1")


# ── PgPatternStore tests ─────────────────────────────────────────────


class TestPgPatternStore:
    """Test PgPatternStore async CRUD methods."""

    @pytest.fixture
    def pool(self):
        return _make_pool()

    @pytest.fixture
    def store(self, pool):
        return PgPatternStore(pool)

    @pytest.mark.asyncio
    async def test_save_new_pattern(self, store, pool):
        pattern = LearnedPattern(
            user_id="user-1",
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="High energy mornings",
        )
        result = await store.save(pattern)
        assert result.id  # ID should be assigned
        pool.execute.assert_awaited_once()
        sql = pool.execute.call_args[0][0]
        assert "INSERT INTO learned_patterns" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_save_generates_id_if_empty(self, store, pool):
        pattern = LearnedPattern(
            id="",
            user_id="user-1",
            pattern_type=PatternType.MOOD_CYCLE,
            description="Test",
        )
        result = await store.save(pattern)
        assert result.id != ""

    @pytest.mark.asyncio
    async def test_get_found(self, store, pool):
        row = _make_pattern_row(id="pat-1")
        pool.fetchrow = AsyncMock(return_value=row)
        result = await store.get("pat-1")
        assert result is not None
        assert result.id == "pat-1"

    @pytest.mark.asyncio
    async def test_get_not_found(self, store, pool):
        pool.fetchrow = AsyncMock(return_value=None)
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_basic(self, store, pool):
        rows = [_make_pattern_row(), _make_pattern_row()]
        pool.fetch = AsyncMock(return_value=rows)
        result = await store.query("user-1")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_query_with_type_filter(self, store, pool):
        pool.fetch = AsyncMock(return_value=[])
        await store.query("user-1", pattern_type=PatternType.MOOD_CYCLE)
        sql = pool.fetch.call_args[0][0]
        assert "pattern_type = $4" in sql

    @pytest.mark.asyncio
    async def test_query_with_day_filter(self, store, pool):
        """Day filter is applied in Python after SQL fetch."""
        rows = [
            _make_pattern_row(parameters='{"day_of_week": "monday"}'),
            _make_pattern_row(parameters='{"day_of_week": "friday"}'),
        ]
        pool.fetch = AsyncMock(return_value=rows)
        result = await store.query("user-1", day_of_week="monday")
        assert len(result) == 1
        assert result[0].parameters["day_of_week"] == "monday"

    @pytest.mark.asyncio
    async def test_query_with_time_filter(self, store, pool):
        """Time filter is applied in Python after SQL fetch."""
        rows = [
            _make_pattern_row(parameters='{"time_of_day": "morning"}'),
            _make_pattern_row(parameters='{"time_of_day": "evening"}'),
        ]
        pool.fetch = AsyncMock(return_value=rows)
        result = await store.query("user-1", time_of_day="morning")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_query_offset_limit(self, store, pool):
        rows = [_make_pattern_row() for _ in range(5)]
        pool.fetch = AsyncMock(return_value=rows)
        result = await store.query("user-1", offset=2, limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_count(self, store, pool):
        pool.fetchrow = AsyncMock(return_value={"cnt": 7})
        result = await store.count("user-1")
        assert result == 7

    @pytest.mark.asyncio
    async def test_delete_found(self, store, pool):
        pool.execute = AsyncMock(return_value="DELETE 1")
        result = await store.delete("pat-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self, store, pool):
        pool.execute = AsyncMock(return_value="DELETE 0")
        result = await store.delete("nonexistent")
        assert result is False


# ── SQL parameterization tests ───────────────────────────────────────


class TestParameterizedQueries:
    """Verify all SQL uses parameterized queries, not string interpolation."""

    @pytest.mark.asyncio
    async def test_task_store_no_interpolation(self):
        pool = _make_pool()
        store = PgTaskStore(pool)
        task = SurfaceableTask(content="'; DROP TABLE tasks; --")
        await store.add_task_async(task, user_id="user-1")
        sql = pool.execute.call_args[0][0]
        # The malicious content should NOT appear in the SQL string
        assert "DROP TABLE" not in sql
        # Should use $N placeholders
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_feedback_store_no_interpolation(self):
        pool = _make_pool()
        store = PgFeedbackStore(pool)
        pool.fetch = AsyncMock(return_value=[])
        await store.get_events_async(task_id="'; DROP TABLE feedback_events; --")
        sql = pool.fetch.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_pattern_store_no_interpolation(self):
        pool = _make_pool()
        store = PgPatternStore(pool)
        pool.fetch = AsyncMock(return_value=[])
        await store.query("'; DROP TABLE learned_patterns; --")
        sql = pool.fetch.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql


# ── Schema DDL tests ─────────────────────────────────────────────────


class TestSchemaDDL:
    """Verify schema migration includes all three tables."""

    @pytest.mark.asyncio
    async def test_schema_creates_tables(self):
        from blurt.persistence.database import run_schema_migrations

        # Mock pool with context manager for acquire
        pool = MagicMock()
        conn = AsyncMock()
        conn.execute = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        await run_schema_migrations(pool)

        # Collect all executed SQL
        executed_sqls = [call[0][0] for call in conn.execute.call_args_list]
        all_sql = "\n".join(executed_sqls)

        # Verify our three tables are created
        assert "CREATE TABLE IF NOT EXISTS tasks" in all_sql
        assert "CREATE TABLE IF NOT EXISTS feedback_events" in all_sql
        assert "CREATE TABLE IF NOT EXISTS thompson_params" in all_sql
        assert "CREATE TABLE IF NOT EXISTS learned_patterns" in all_sql

        # Verify indexes
        assert "idx_tasks_user_id" in all_sql
        assert "idx_tasks_user_status" in all_sql
        assert "idx_feedback_events_task" in all_sql
        assert "idx_feedback_events_user" in all_sql
        assert "idx_feedback_events_timestamp" in all_sql
