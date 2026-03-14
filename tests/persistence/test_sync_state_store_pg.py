"""Tests for PgSyncStateStore — Postgres sync state store.

Tests the async API methods against the SQL schema and row conversion logic.
Uses mock asyncpg pool to avoid requiring a real database connection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.models.sync import (
    ConflictRecord,
    SyncDirection,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
)
from blurt.persistence.sync_state_store import (
    PgSyncStateStore,
    _ensure_tz,
    _parse_jsonb,
    _row_to_conflict,
    _row_to_operation,
    _row_to_sync_record,
)


@pytest.fixture
def mock_pool():
    return AsyncMock()


@pytest.fixture
def store(mock_pool):
    return PgSyncStateStore(mock_pool)


def _make_record_row(**overrides):
    """Create a mock row for sync_records."""
    now = datetime.now(timezone.utc)
    defaults = {
        "id": str(uuid.uuid4()),
        "blurt_id": "blurt-1",
        "provider": "google_calendar",
        "external_id": "ext-1",
        "direction": "bidirectional",
        "status": "pending",
        "blurt_version": 1,
        "external_version": None,
        "last_synced_at": None,
        "last_blurt_modified_at": None,
        "last_external_modified_at": None,
        "error_message": None,
        "retry_count": 0,
        "max_retries": 3,
        "created_at": now,
        "metadata": {},
    }
    defaults.update(overrides)
    return defaults


def _make_operation_row(**overrides):
    now = datetime.now(timezone.utc)
    defaults = {
        "id": str(uuid.uuid4()),
        "sync_record_id": "rec-1",
        "provider": "google_calendar",
        "direction": "outbound",
        "operation_type": "create",
        "payload": {},
        "status": "pending",
        "created_at": now,
        "completed_at": None,
        "error_message": None,
        "result": None,
    }
    defaults.update(overrides)
    return defaults


def _make_conflict_row(**overrides):
    now = datetime.now(timezone.utc)
    defaults = {
        "id": str(uuid.uuid4()),
        "sync_record_id": "rec-1",
        "provider": "google_calendar",
        "blurt_data": {"title": "from blurt"},
        "external_data": {"title": "from google"},
        "resolution_strategy": "latest_wins",
        "resolved": False,
        "resolved_at": None,
        "resolution_result": None,
        "created_at": now,
    }
    defaults.update(overrides)
    return defaults


class TestPgSyncStateStoreSyncRecords:
    """Test sync record CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_sync_record_found(self, store, mock_pool):
        row = _make_record_row()
        mock_pool.fetchrow.return_value = row
        record = await store.get_sync_record(row["id"])
        assert record is not None
        assert record.blurt_id == "blurt-1"
        assert record.provider == SyncProvider.GOOGLE_CALENDAR

    @pytest.mark.asyncio
    async def test_get_sync_record_not_found(self, store, mock_pool):
        mock_pool.fetchrow.return_value = None
        result = await store.get_sync_record("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_sync_record_by_blurt_id(self, store, mock_pool):
        row = _make_record_row()
        mock_pool.fetchrow.return_value = row
        record = await store.get_sync_record_by_blurt_id("blurt-1", SyncProvider.GOOGLE_CALENDAR)
        assert record is not None
        sql = mock_pool.fetchrow.call_args[0][0]
        assert "blurt_id = $1" in sql
        assert "provider = $2" in sql

    @pytest.mark.asyncio
    async def test_upsert_sync_record(self, store, mock_pool):
        record = SyncRecord(blurt_id="b1", provider=SyncProvider.GOOGLE_CALENDAR)
        mock_pool.execute.return_value = "INSERT 0 1"
        result = await store.upsert_sync_record(record)
        assert result == record
        sql = mock_pool.execute.call_args[0][0]
        assert "INSERT INTO sync_records" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_mark_synced(self, store, mock_pool):
        row = _make_record_row(status="completed", retry_count=0)
        mock_pool.fetchrow.return_value = row
        result = await store.mark_synced(row["id"], external_id="new-ext")
        assert result is not None
        sql = mock_pool.fetchrow.call_args[0][0]
        assert "UPDATE sync_records" in sql

    @pytest.mark.asyncio
    async def test_mark_synced_not_found(self, store, mock_pool):
        mock_pool.fetchrow.return_value = None
        result = await store.mark_synced("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_mark_failed(self, store, mock_pool):
        row = _make_record_row(status="failed", retry_count=1, error_message="timeout")
        mock_pool.fetchrow.return_value = row
        result = await store.mark_failed(row["id"], "timeout")
        assert result is not None

    @pytest.mark.asyncio
    async def test_mark_conflict(self, store, mock_pool):
        row = _make_record_row(status="conflict")
        mock_pool.fetchrow.return_value = row
        result = await store.mark_conflict(row["id"])
        assert result is not None


class TestPgSyncStateStoreOperations:
    """Test sync operation CRUD."""

    @pytest.mark.asyncio
    async def test_add_operation(self, store, mock_pool):
        op = SyncOperation(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            direction=SyncDirection.OUTBOUND,
            operation_type="create",
        )
        mock_pool.execute.return_value = "INSERT 0 1"
        result = await store.add_operation(op)
        assert result == op
        sql = mock_pool.execute.call_args[0][0]
        assert "INSERT INTO sync_operations" in sql

    @pytest.mark.asyncio
    async def test_get_operation(self, store, mock_pool):
        row = _make_operation_row()
        mock_pool.fetchrow.return_value = row
        result = await store.get_operation(row["id"])
        assert result is not None
        assert result.operation_type == "create"

    @pytest.mark.asyncio
    async def test_get_operations_for_record(self, store, mock_pool):
        mock_pool.fetch.return_value = [_make_operation_row(), _make_operation_row()]
        result = await store.get_operations_for_record("rec-1")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_complete_operation(self, store, mock_pool):
        row = _make_operation_row(status="completed")
        mock_pool.fetchrow.return_value = row
        result = await store.complete_operation(row["id"], result={"ok": True})
        assert result is not None

    @pytest.mark.asyncio
    async def test_fail_operation(self, store, mock_pool):
        row = _make_operation_row(status="failed", error_message="oops")
        mock_pool.fetchrow.return_value = row
        result = await store.fail_operation(row["id"], "oops")
        assert result is not None


class TestPgSyncStateStoreConflicts:
    """Test conflict CRUD."""

    @pytest.mark.asyncio
    async def test_add_conflict(self, store, mock_pool):
        conflict = ConflictRecord(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={"a": 1},
            external_data={"b": 2},
        )
        mock_pool.execute.return_value = "INSERT 0 1"
        result = await store.add_conflict(conflict)
        assert result == conflict

    @pytest.mark.asyncio
    async def test_get_unresolved_conflicts(self, store, mock_pool):
        mock_pool.fetch.return_value = [_make_conflict_row()]
        result = await store.get_unresolved_conflicts()
        assert len(result) == 1
        assert result[0].resolved is False

    @pytest.mark.asyncio
    async def test_resolve_conflict(self, store, mock_pool):
        row = _make_conflict_row(resolved=True)
        mock_pool.fetchrow.return_value = row
        result = await store.resolve_conflict(row["id"], {"winner": "blurt"})
        assert result is not None


class TestPgSyncStateStoreStats:
    """Test stats aggregation."""

    @pytest.mark.asyncio
    async def test_stats(self, mock_pool):
        # pool.acquire() in asyncpg returns a sync context manager (PoolAcquireContext)
        # that supports async with. We need a non-async mock for acquire().
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [10, 5, 3, 1]
        mock_conn.fetch.return_value = [
            {"status": "pending", "cnt": 5},
            {"status": "completed", "cnt": 5},
        ]

        # Use a MagicMock for pool so acquire() is not a coroutine
        pool = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire.return_value = cm

        store = PgSyncStateStore(pool)
        result = await store.stats()
        assert result["total_records"] == 10
        assert result["total_operations"] == 5
        assert result["total_conflicts"] == 3
        assert result["unresolved_conflicts"] == 1
        assert result["status_counts"]["pending"] == 5


class TestHelpers:
    """Test helper functions."""

    def test_ensure_tz_none(self):
        assert _ensure_tz(None) is None

    def test_ensure_tz_aware(self):
        dt = datetime.now(timezone.utc)
        assert _ensure_tz(dt) is dt

    def test_ensure_tz_naive(self):
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = _ensure_tz(dt)
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_parse_jsonb_none(self):
        assert _parse_jsonb(None) == {}

    def test_parse_jsonb_str(self):
        assert _parse_jsonb('{"a": 1}') == {"a": 1}

    def test_parse_jsonb_dict(self):
        assert _parse_jsonb({"a": 1}) == {"a": 1}


class TestRowConverters:
    """Test row-to-model conversion functions."""

    def test_row_to_sync_record(self):
        row = _make_record_row()
        record = _row_to_sync_record(row)
        assert record.blurt_id == "blurt-1"
        assert record.provider == SyncProvider.GOOGLE_CALENDAR
        assert record.status == SyncStatus.PENDING

    def test_row_to_operation(self):
        row = _make_operation_row()
        op = _row_to_operation(row)
        assert op.operation_type == "create"
        assert op.provider == SyncProvider.GOOGLE_CALENDAR

    def test_row_to_conflict(self):
        row = _make_conflict_row()
        conflict = _row_to_conflict(row)
        assert conflict.resolved is False
        assert conflict.blurt_data == {"title": "from blurt"}
