"""Tests for PgPreferenceBackend — Postgres preference store.

Tests the async API methods against the SQL schema and row conversion logic.
Uses mock asyncpg pool to avoid requiring a real database connection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from blurt.persistence.preference_store import PgPreferenceBackend, _row_to_snapshot
from blurt.services.preference_store import UserPreferenceSnapshot
from blurt.services.thompson_sampling import BetaParams


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = AsyncMock()
    return pool


@pytest.fixture
def backend(mock_pool):
    return PgPreferenceBackend(mock_pool)


@pytest.fixture
def sample_snapshot():
    now = datetime.now(timezone.utc)
    return UserPreferenceSnapshot(
        user_id="test-user-1",
        params={
            "task": BetaParams(alpha=2.0, beta=1.5, last_updated=now, total_observations=3),
            "event": BetaParams(alpha=1.0, beta=1.0, last_updated=now, total_observations=0),
        },
        created_at=now,
        last_interaction=now,
        total_feedback_count=3,
        version=1,
    )


class TestPgPreferenceBackendAsync:
    """Test async methods of PgPreferenceBackend."""

    @pytest.mark.asyncio
    async def test_load_snapshot_not_found(self, backend, mock_pool):
        mock_pool.fetchrow.return_value = None
        result = await backend.load_snapshot_async("nonexistent")
        assert result is None
        mock_pool.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_snapshot_found(self, backend, mock_pool):
        now = datetime.now(timezone.utc)
        params_json = json.dumps({
            "task": {"alpha": 2.0, "beta": 1.5, "last_updated": now.isoformat(), "total_observations": 3},
        })
        mock_pool.fetchrow.return_value = {
            "user_id": "u1",
            "params": params_json,
            "created_at": now,
            "last_interaction": now,
            "total_feedback_count": 3,
            "version": 1,
        }
        result = await backend.load_snapshot_async("u1")
        assert result is not None
        assert result.user_id == "u1"
        assert "task" in result.params
        assert result.params["task"].alpha == 2.0
        assert result.total_feedback_count == 3

    @pytest.mark.asyncio
    async def test_save_snapshot(self, backend, mock_pool, sample_snapshot):
        mock_pool.execute.return_value = "INSERT 0 1"
        await backend.save_snapshot_async(sample_snapshot)
        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        # Verify parameterized query (no string interpolation)
        sql = call_args[0][0]
        assert "$1" in sql
        assert "$2" in sql
        assert "INSERT INTO user_preferences" in sql

    @pytest.mark.asyncio
    async def test_delete_snapshot_found(self, backend, mock_pool):
        mock_pool.execute.return_value = "DELETE 1"
        result = await backend.delete_snapshot_async("u1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_snapshot_not_found(self, backend, mock_pool):
        mock_pool.execute.return_value = "DELETE 0"
        result = await backend.delete_snapshot_async("u1")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_users(self, backend, mock_pool):
        mock_pool.fetch.return_value = [
            {"user_id": "u1"},
            {"user_id": "u2"},
        ]
        result = await backend.list_users_async()
        assert result == ["u1", "u2"]


class TestRowToSnapshot:
    """Test the row-to-snapshot conversion logic."""

    def test_basic_conversion(self):
        now = datetime.now(timezone.utc)
        row = {
            "user_id": "u1",
            "params": {"task": {"alpha": 3.0, "beta": 2.0, "last_updated": now.isoformat(), "total_observations": 5}},
            "created_at": now,
            "last_interaction": now,
            "total_feedback_count": 5,
            "version": 1,
        }
        snap = _row_to_snapshot(row)
        assert snap.user_id == "u1"
        assert snap.params["task"].alpha == 3.0
        assert snap.total_feedback_count == 5

    def test_handles_string_params(self):
        now = datetime.now(timezone.utc)
        params_str = json.dumps({"event": {"alpha": 1.0, "beta": 1.0, "last_updated": now.isoformat(), "total_observations": 0}})
        row = {
            "user_id": "u2",
            "params": params_str,
            "created_at": now,
            "last_interaction": now,
            "total_feedback_count": 0,
            "version": 1,
        }
        snap = _row_to_snapshot(row)
        assert "event" in snap.params

    def test_naive_datetime_gets_utc(self):
        naive_dt = datetime(2025, 1, 1, 12, 0, 0)  # No tzinfo
        row = {
            "user_id": "u3",
            "params": {},
            "created_at": naive_dt,
            "last_interaction": naive_dt,
            "total_feedback_count": 0,
            "version": 1,
        }
        snap = _row_to_snapshot(row)
        assert snap.created_at.tzinfo == timezone.utc


class TestBetaParamsSurviveRestart:
    """Verify Thompson Sampling beta parameters survive simulated server restart."""

    @pytest.mark.asyncio
    async def test_beta_params_persist_across_store_instances(self, mock_pool):
        """Simulate restart: save snapshot via store1, load via store2 (new instance)."""
        now = datetime.now(timezone.utc)
        snapshot = UserPreferenceSnapshot(
            user_id="restart-user",
            params={
                "task": BetaParams(alpha=5.0, beta=2.0, last_updated=now, total_observations=7),
                "reminder": BetaParams(alpha=1.5, beta=3.0, last_updated=now, total_observations=4),
            },
            created_at=now,
            last_interaction=now,
            total_feedback_count=11,
            version=2,
        )

        # --- "Before restart" ---
        store1 = PgPreferenceBackend(mock_pool)
        mock_pool.execute.return_value = "INSERT 0 1"
        await store1.save_snapshot_async(snapshot)

        # --- "After restart" — new store instance, same pool ---
        store2 = PgPreferenceBackend(mock_pool)
        mock_pool.fetchrow.return_value = {
            "user_id": "restart-user",
            "params": json.dumps({
                "task": {"alpha": 5.0, "beta": 2.0, "last_updated": now.isoformat(), "total_observations": 7},
                "reminder": {"alpha": 1.5, "beta": 3.0, "last_updated": now.isoformat(), "total_observations": 4},
            }),
            "created_at": now,
            "last_interaction": now,
            "total_feedback_count": 11,
            "version": 2,
        }
        result = await store2.load_snapshot_async("restart-user")

        assert result is not None
        assert result.user_id == "restart-user"
        assert result.params["task"].alpha == 5.0
        assert result.params["task"].beta == 2.0
        assert result.params["task"].total_observations == 7
        assert result.params["reminder"].alpha == 1.5
        assert result.params["reminder"].beta == 3.0
        assert result.total_feedback_count == 11
        assert result.version == 2


class TestSyncApiRaises:
    """Verify sync wrappers raise NotImplementedError."""

    def test_load_snapshot_sync_raises(self, backend):
        with pytest.raises(NotImplementedError):
            backend.load_snapshot("u1")

    def test_save_snapshot_sync_raises(self, backend, sample_snapshot):
        with pytest.raises(NotImplementedError):
            backend.save_snapshot(sample_snapshot)

    def test_delete_snapshot_sync_raises(self, backend):
        with pytest.raises(NotImplementedError):
            backend.delete_snapshot("u1")

    def test_list_users_sync_raises(self, backend):
        with pytest.raises(NotImplementedError):
            backend.list_users()
