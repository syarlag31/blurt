"""Neon Postgres backend for SyncStateStore.

Replaces the in-memory SyncStateStore with Postgres persistence using asyncpg
and parameterized queries. Tracks sync records, operations, and conflicts.

Tables:
- sync_records: Maps Blurt entities to external service entities
- sync_operations: Individual sync operations (create/update/delete)
- sync_conflicts: Detected conflicts between Blurt and external services
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.models.sync import (
    ConflictRecord,
    ConflictResolutionStrategy,
    SyncDirection,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
)

logger = logging.getLogger(__name__)


# ── Schema DDL ───────────────────────────────────────────────────────────────

CREATE_SYNC_RECORDS_SQL = """
CREATE TABLE IF NOT EXISTS sync_records (
    id                      TEXT PRIMARY KEY,
    blurt_id                TEXT NOT NULL,
    provider                TEXT NOT NULL,
    external_id             TEXT,
    direction               TEXT NOT NULL DEFAULT 'bidirectional',
    status                  TEXT NOT NULL DEFAULT 'pending',
    blurt_version           INTEGER NOT NULL DEFAULT 1,
    external_version        TEXT,
    last_synced_at          TIMESTAMPTZ,
    last_blurt_modified_at  TIMESTAMPTZ,
    last_external_modified_at TIMESTAMPTZ,
    error_message           TEXT,
    retry_count             INTEGER NOT NULL DEFAULT 0,
    max_retries             INTEGER NOT NULL DEFAULT 3,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata                JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""

CREATE_SYNC_RECORDS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sync_records_blurt_provider
ON sync_records (blurt_id, provider)
"""

CREATE_SYNC_RECORDS_EXTERNAL_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sync_records_external
ON sync_records (external_id, provider)
"""

CREATE_SYNC_RECORDS_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sync_records_status
ON sync_records (status)
"""

CREATE_SYNC_OPERATIONS_SQL = """
CREATE TABLE IF NOT EXISTS sync_operations (
    id              TEXT PRIMARY KEY,
    sync_record_id  TEXT NOT NULL REFERENCES sync_records(id) ON DELETE CASCADE,
    provider        TEXT NOT NULL,
    direction       TEXT NOT NULL DEFAULT 'outbound',
    operation_type  TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    result          JSONB
)
"""

CREATE_SYNC_OPERATIONS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sync_operations_record
ON sync_operations (sync_record_id, created_at)
"""

CREATE_SYNC_CONFLICTS_SQL = """
CREATE TABLE IF NOT EXISTS sync_conflicts (
    id                      TEXT PRIMARY KEY,
    sync_record_id          TEXT NOT NULL REFERENCES sync_records(id) ON DELETE CASCADE,
    provider                TEXT NOT NULL,
    blurt_data              JSONB NOT NULL DEFAULT '{}'::jsonb,
    external_data           JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolution_strategy     TEXT NOT NULL DEFAULT 'latest_wins',
    resolved                BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at             TIMESTAMPTZ,
    resolution_result       JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_SYNC_CONFLICTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_unresolved
ON sync_conflicts (resolved) WHERE NOT resolved
"""


class PgSyncStateStore:
    """Postgres implementation of SyncStateStore.

    Provides async methods matching the in-memory SyncStateStore interface.
    All SQL uses parameterized queries — no string interpolation.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Sync Records ─────────────────────────────────────────────────────

    async def get_sync_record(self, record_id: str) -> SyncRecord | None:
        """Get a sync record by its ID."""
        row = await self._pool.fetchrow(
            "SELECT * FROM sync_records WHERE id = $1", record_id
        )
        if row is None:
            return None
        return _row_to_sync_record(row)

    async def get_sync_record_by_blurt_id(
        self, blurt_id: str, provider: SyncProvider
    ) -> SyncRecord | None:
        """Find the sync record for a Blurt entity and provider."""
        row = await self._pool.fetchrow(
            "SELECT * FROM sync_records WHERE blurt_id = $1 AND provider = $2",
            blurt_id,
            provider.value,
        )
        if row is None:
            return None
        return _row_to_sync_record(row)

    async def get_sync_record_by_external_id(
        self, external_id: str, provider: SyncProvider
    ) -> SyncRecord | None:
        """Find the sync record for an external entity."""
        row = await self._pool.fetchrow(
            "SELECT * FROM sync_records WHERE external_id = $1 AND provider = $2",
            external_id,
            provider.value,
        )
        if row is None:
            return None
        return _row_to_sync_record(row)

    async def upsert_sync_record(self, record: SyncRecord) -> SyncRecord:
        """Create or update a sync record."""
        metadata_json = json.dumps(record.metadata)
        await self._pool.execute(
            """
            INSERT INTO sync_records
                (id, blurt_id, provider, external_id, direction, status,
                 blurt_version, external_version, last_synced_at,
                 last_blurt_modified_at, last_external_modified_at,
                 error_message, retry_count, max_retries, created_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                blurt_id = $2,
                provider = $3,
                external_id = $4,
                direction = $5,
                status = $6,
                blurt_version = $7,
                external_version = $8,
                last_synced_at = $9,
                last_blurt_modified_at = $10,
                last_external_modified_at = $11,
                error_message = $12,
                retry_count = $13,
                max_retries = $14,
                metadata = $16::jsonb
            """,
            record.id,
            record.blurt_id,
            record.provider.value,
            record.external_id,
            record.direction.value,
            record.status.value,
            record.blurt_version,
            record.external_version,
            record.last_synced_at,
            record.last_blurt_modified_at,
            record.last_external_modified_at,
            record.error_message,
            record.retry_count,
            record.max_retries,
            record.created_at,
            metadata_json,
        )
        return record

    async def get_pending_records(
        self, provider: SyncProvider | None = None
    ) -> list[SyncRecord]:
        """Get all records that need syncing (pending/failed with retries, or needing sync)."""
        if provider is not None:
            rows = await self._pool.fetch(
                """
                SELECT * FROM sync_records
                WHERE provider = $1 AND (
                    (status IN ('pending', 'failed') AND retry_count < max_retries)
                    OR last_synced_at IS NULL
                    OR (last_blurt_modified_at IS NOT NULL AND (last_synced_at IS NULL OR last_blurt_modified_at > last_synced_at))
                    OR (last_external_modified_at IS NOT NULL AND (last_synced_at IS NULL OR last_external_modified_at > last_synced_at))
                )
                """,
                provider.value,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM sync_records
                WHERE (
                    (status IN ('pending', 'failed') AND retry_count < max_retries)
                    OR last_synced_at IS NULL
                    OR (last_blurt_modified_at IS NOT NULL AND (last_synced_at IS NULL OR last_blurt_modified_at > last_synced_at))
                    OR (last_external_modified_at IS NOT NULL AND (last_synced_at IS NULL OR last_external_modified_at > last_synced_at))
                )
                """
            )
        return [_row_to_sync_record(row) for row in rows]

    async def get_conflicted_records(self) -> list[SyncRecord]:
        """Get all records with unresolved conflicts."""
        rows = await self._pool.fetch(
            "SELECT * FROM sync_records WHERE status = $1",
            SyncStatus.CONFLICT.value,
        )
        return [_row_to_sync_record(row) for row in rows]

    async def mark_synced(
        self,
        record_id: str,
        external_id: str | None = None,
        external_version: str | None = None,
    ) -> SyncRecord | None:
        """Mark a sync record as successfully synced."""
        now = datetime.now(timezone.utc)
        # Build dynamic update
        row = await self._pool.fetchrow(
            """
            UPDATE sync_records SET
                status = $2,
                last_synced_at = $3,
                error_message = NULL,
                retry_count = 0,
                external_id = COALESCE($4, external_id),
                external_version = COALESCE($5, external_version)
            WHERE id = $1
            RETURNING *
            """,
            record_id,
            SyncStatus.COMPLETED.value,
            now,
            external_id,
            external_version,
        )
        if row is None:
            return None
        return _row_to_sync_record(row)

    async def mark_failed(self, record_id: str, error: str) -> SyncRecord | None:
        """Mark a sync record as failed."""
        row = await self._pool.fetchrow(
            """
            UPDATE sync_records SET
                status = $2,
                error_message = $3,
                retry_count = retry_count + 1
            WHERE id = $1
            RETURNING *
            """,
            record_id,
            SyncStatus.FAILED.value,
            error,
        )
        if row is None:
            return None
        return _row_to_sync_record(row)

    async def mark_conflict(self, record_id: str) -> SyncRecord | None:
        """Mark a sync record as having a conflict."""
        row = await self._pool.fetchrow(
            """
            UPDATE sync_records SET status = $2
            WHERE id = $1
            RETURNING *
            """,
            record_id,
            SyncStatus.CONFLICT.value,
        )
        if row is None:
            return None
        return _row_to_sync_record(row)

    # ── Operations ───────────────────────────────────────────────────────

    async def add_operation(self, operation: SyncOperation) -> SyncOperation:
        """Record a sync operation."""
        payload_json = json.dumps(operation.payload)
        result_json = json.dumps(operation.result) if operation.result else None
        await self._pool.execute(
            """
            INSERT INTO sync_operations
                (id, sync_record_id, provider, direction, operation_type,
                 payload, status, created_at, completed_at, error_message, result)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            operation.id,
            operation.sync_record_id,
            operation.provider.value,
            operation.direction.value,
            operation.operation_type,
            payload_json,
            operation.status.value,
            operation.created_at,
            operation.completed_at,
            operation.error_message,
            result_json,
        )
        return operation

    async def get_operation(self, operation_id: str) -> SyncOperation | None:
        """Get a sync operation by ID."""
        row = await self._pool.fetchrow(
            "SELECT * FROM sync_operations WHERE id = $1", operation_id
        )
        if row is None:
            return None
        return _row_to_operation(row)

    async def get_operations_for_record(
        self, sync_record_id: str
    ) -> list[SyncOperation]:
        """Get all operations for a sync record, ordered by creation time."""
        rows = await self._pool.fetch(
            """
            SELECT * FROM sync_operations
            WHERE sync_record_id = $1
            ORDER BY created_at
            """,
            sync_record_id,
        )
        return [_row_to_operation(row) for row in rows]

    async def complete_operation(
        self, operation_id: str, result: dict[str, Any] | None = None
    ) -> SyncOperation | None:
        """Mark an operation as completed."""
        now = datetime.now(timezone.utc)
        result_json = json.dumps(result) if result else None
        row = await self._pool.fetchrow(
            """
            UPDATE sync_operations SET
                status = $2,
                completed_at = $3,
                result = $4::jsonb
            WHERE id = $1
            RETURNING *
            """,
            operation_id,
            SyncStatus.COMPLETED.value,
            now,
            result_json,
        )
        if row is None:
            return None
        return _row_to_operation(row)

    async def fail_operation(
        self, operation_id: str, error: str
    ) -> SyncOperation | None:
        """Mark an operation as failed."""
        row = await self._pool.fetchrow(
            """
            UPDATE sync_operations SET
                status = $2,
                error_message = $3
            WHERE id = $1
            RETURNING *
            """,
            operation_id,
            SyncStatus.FAILED.value,
            error,
        )
        if row is None:
            return None
        return _row_to_operation(row)

    # ── Conflicts ────────────────────────────────────────────────────────

    async def add_conflict(self, conflict: ConflictRecord) -> ConflictRecord:
        """Record a conflict."""
        blurt_data_json = json.dumps(conflict.blurt_data)
        external_data_json = json.dumps(conflict.external_data)
        resolution_json = json.dumps(conflict.resolution_result) if conflict.resolution_result else None
        await self._pool.execute(
            """
            INSERT INTO sync_conflicts
                (id, sync_record_id, provider, blurt_data, external_data,
                 resolution_strategy, resolved, resolved_at, resolution_result, created_at)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8, $9::jsonb, $10)
            ON CONFLICT (id) DO NOTHING
            """,
            conflict.id,
            conflict.sync_record_id,
            conflict.provider.value,
            blurt_data_json,
            external_data_json,
            conflict.resolution_strategy.value,
            conflict.resolved,
            conflict.resolved_at,
            resolution_json,
            conflict.created_at,
        )
        return conflict

    async def get_conflict(self, conflict_id: str) -> ConflictRecord | None:
        """Get a conflict by ID."""
        row = await self._pool.fetchrow(
            "SELECT * FROM sync_conflicts WHERE id = $1", conflict_id
        )
        if row is None:
            return None
        return _row_to_conflict(row)

    async def get_unresolved_conflicts(self) -> list[ConflictRecord]:
        """Get all unresolved conflicts."""
        rows = await self._pool.fetch(
            "SELECT * FROM sync_conflicts WHERE resolved = FALSE"
        )
        return [_row_to_conflict(row) for row in rows]

    async def resolve_conflict(
        self, conflict_id: str, result: dict[str, Any]
    ) -> ConflictRecord | None:
        """Mark a conflict as resolved."""
        now = datetime.now(timezone.utc)
        result_json = json.dumps(result)
        row = await self._pool.fetchrow(
            """
            UPDATE sync_conflicts SET
                resolved = TRUE,
                resolved_at = $2,
                resolution_result = $3::jsonb
            WHERE id = $1
            RETURNING *
            """,
            conflict_id,
            now,
            result_json,
        )
        if row is None:
            return None
        return _row_to_conflict(row)

    # ── Stats ────────────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Return sync state statistics."""
        async with self._pool.acquire() as conn:
            total_records = await conn.fetchval(
                "SELECT COUNT(*) FROM sync_records"
            )
            total_operations = await conn.fetchval(
                "SELECT COUNT(*) FROM sync_operations"
            )
            total_conflicts = await conn.fetchval(
                "SELECT COUNT(*) FROM sync_conflicts"
            )
            unresolved = await conn.fetchval(
                "SELECT COUNT(*) FROM sync_conflicts WHERE resolved = FALSE"
            )
            status_rows = await conn.fetch(
                "SELECT status, COUNT(*) AS cnt FROM sync_records GROUP BY status"
            )

        status_counts = {row["status"]: row["cnt"] for row in status_rows}

        return {
            "total_records": total_records,
            "total_operations": total_operations,
            "total_conflicts": total_conflicts,
            "unresolved_conflicts": unresolved,
            "status_counts": status_counts,
        }


# ── Row converters ───────────────────────────────────────────────────────────


def _ensure_tz(dt: datetime | None) -> datetime | None:
    """Ensure datetime has UTC timezone info."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_jsonb(value: Any) -> dict[str, Any]:
    """Parse a JSONB value that may be str or dict."""
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _row_to_sync_record(row: Mapping[str, Any]) -> SyncRecord:
    """Convert an asyncpg Record to a SyncRecord."""
    return SyncRecord(
        id=row["id"],
        blurt_id=row["blurt_id"],
        provider=SyncProvider(row["provider"]),
        external_id=row["external_id"],
        direction=SyncDirection(row["direction"]),
        status=SyncStatus(row["status"]),
        blurt_version=row["blurt_version"],
        external_version=row["external_version"],
        last_synced_at=_ensure_tz(row["last_synced_at"]),
        last_blurt_modified_at=_ensure_tz(row["last_blurt_modified_at"]),
        last_external_modified_at=_ensure_tz(row["last_external_modified_at"]),
        error_message=row["error_message"],
        retry_count=row["retry_count"],
        max_retries=row["max_retries"],
        created_at=_ensure_tz(row["created_at"]) or datetime.now(timezone.utc),
        metadata=_parse_jsonb(row["metadata"]),
    )


def _row_to_operation(row: Mapping[str, Any]) -> SyncOperation:
    """Convert an asyncpg Record to a SyncOperation."""
    return SyncOperation(
        id=row["id"],
        sync_record_id=row["sync_record_id"],
        provider=SyncProvider(row["provider"]),
        direction=SyncDirection(row["direction"]),
        operation_type=row["operation_type"],
        payload=_parse_jsonb(row["payload"]),
        status=SyncStatus(row["status"]),
        created_at=_ensure_tz(row["created_at"]) or datetime.now(timezone.utc),
        completed_at=_ensure_tz(row["completed_at"]),
        error_message=row["error_message"],
        result=_parse_jsonb(row["result"]) if row["result"] is not None else None,
    )


def _row_to_conflict(row: Mapping[str, Any]) -> ConflictRecord:
    """Convert an asyncpg Record to a ConflictRecord."""
    return ConflictRecord(
        id=row["id"],
        sync_record_id=row["sync_record_id"],
        provider=SyncProvider(row["provider"]),
        blurt_data=_parse_jsonb(row["blurt_data"]),
        external_data=_parse_jsonb(row["external_data"]),
        resolution_strategy=ConflictResolutionStrategy(row["resolution_strategy"]),
        resolved=row["resolved"],
        resolved_at=_ensure_tz(row["resolved_at"]),
        resolution_result=_parse_jsonb(row["resolution_result"]) if row["resolution_result"] is not None else None,
        created_at=_ensure_tz(row["created_at"]) or datetime.now(timezone.utc),
    )
