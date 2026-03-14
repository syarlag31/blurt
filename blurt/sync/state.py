"""Sync state tracking — persistence layer for sync records and operations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from blurt.models.sync import (
    ConflictRecord,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
)

logger = logging.getLogger(__name__)


class SyncStateStore:
    """In-memory sync state store.

    Tracks sync records (mapping between Blurt entities and external entities),
    pending operations, and conflict records.

    In production, this would be backed by a database. The in-memory
    implementation supports the same interface for testing and local-only mode.
    """

    def __init__(self) -> None:
        self._sync_records: dict[str, SyncRecord] = {}
        self._operations: dict[str, SyncOperation] = {}
        self._conflicts: dict[str, ConflictRecord] = {}
        # Index: blurt_id + provider -> sync_record_id
        self._blurt_provider_index: dict[str, str] = {}

    def _index_key(self, blurt_id: str, provider: SyncProvider) -> str:
        return f"{blurt_id}:{provider.value}"

    # --- Sync Records ---

    def get_sync_record(self, record_id: str) -> SyncRecord | None:
        """Get a sync record by its ID."""
        return self._sync_records.get(record_id)

    def get_sync_record_by_blurt_id(
        self, blurt_id: str, provider: SyncProvider
    ) -> SyncRecord | None:
        """Find the sync record for a Blurt entity and provider."""
        key = self._index_key(blurt_id, provider)
        record_id = self._blurt_provider_index.get(key)
        if record_id:
            return self._sync_records.get(record_id)
        return None

    def get_sync_record_by_external_id(
        self, external_id: str, provider: SyncProvider
    ) -> SyncRecord | None:
        """Find the sync record for an external entity."""
        for record in self._sync_records.values():
            if record.external_id == external_id and record.provider == provider:
                return record
        return None

    def upsert_sync_record(self, record: SyncRecord) -> SyncRecord:
        """Create or update a sync record."""
        self._sync_records[record.id] = record
        key = self._index_key(record.blurt_id, record.provider)
        self._blurt_provider_index[key] = record.id
        return record

    def get_pending_records(self, provider: SyncProvider | None = None) -> list[SyncRecord]:
        """Get all records that need syncing."""
        results = []
        for record in self._sync_records.values():
            if provider and record.provider != provider:
                continue
            if record.status in (SyncStatus.PENDING, SyncStatus.FAILED) and record.can_retry:
                results.append(record)
            elif record.needs_outbound_sync or record.needs_inbound_sync:
                results.append(record)
        return results

    def get_conflicted_records(self) -> list[SyncRecord]:
        """Get all records with unresolved conflicts."""
        return [
            r for r in self._sync_records.values()
            if r.status == SyncStatus.CONFLICT
        ]

    def mark_synced(
        self,
        record_id: str,
        external_id: str | None = None,
        external_version: str | None = None,
    ) -> SyncRecord | None:
        """Mark a sync record as successfully synced."""
        record = self._sync_records.get(record_id)
        if record is None:
            return None

        now = datetime.now(timezone.utc)
        record.status = SyncStatus.COMPLETED
        record.last_synced_at = now
        record.error_message = None
        record.retry_count = 0
        if external_id:
            record.external_id = external_id
        if external_version:
            record.external_version = external_version
        return record

    def mark_failed(self, record_id: str, error: str) -> SyncRecord | None:
        """Mark a sync record as failed."""
        record = self._sync_records.get(record_id)
        if record is None:
            return None

        record.status = SyncStatus.FAILED
        record.error_message = error
        record.retry_count += 1
        return record

    def mark_conflict(self, record_id: str) -> SyncRecord | None:
        """Mark a sync record as having a conflict."""
        record = self._sync_records.get(record_id)
        if record is None:
            return None

        record.status = SyncStatus.CONFLICT
        return record

    # --- Operations ---

    def add_operation(self, operation: SyncOperation) -> SyncOperation:
        """Record a sync operation."""
        self._operations[operation.id] = operation
        return operation

    def get_operation(self, operation_id: str) -> SyncOperation | None:
        return self._operations.get(operation_id)

    def get_operations_for_record(self, sync_record_id: str) -> list[SyncOperation]:
        """Get all operations for a sync record, ordered by creation time."""
        ops = [
            op for op in self._operations.values()
            if op.sync_record_id == sync_record_id
        ]
        return sorted(ops, key=lambda o: o.created_at)

    def complete_operation(
        self, operation_id: str, result: dict[str, Any] | None = None
    ) -> SyncOperation | None:
        """Mark an operation as completed."""
        op = self._operations.get(operation_id)
        if op is None:
            return None
        op.status = SyncStatus.COMPLETED
        op.completed_at = datetime.now(timezone.utc)
        op.result = result
        return op

    def fail_operation(self, operation_id: str, error: str) -> SyncOperation | None:
        """Mark an operation as failed."""
        op = self._operations.get(operation_id)
        if op is None:
            return None
        op.status = SyncStatus.FAILED
        op.error_message = error
        return op

    # --- Conflicts ---

    def add_conflict(self, conflict: ConflictRecord) -> ConflictRecord:
        """Record a conflict."""
        self._conflicts[conflict.id] = conflict
        return conflict

    def get_conflict(self, conflict_id: str) -> ConflictRecord | None:
        return self._conflicts.get(conflict_id)

    def get_unresolved_conflicts(self) -> list[ConflictRecord]:
        """Get all unresolved conflicts."""
        return [c for c in self._conflicts.values() if not c.resolved]

    def resolve_conflict(
        self, conflict_id: str, result: dict[str, Any]
    ) -> ConflictRecord | None:
        """Mark a conflict as resolved."""
        conflict = self._conflicts.get(conflict_id)
        if conflict is None:
            return None
        conflict.resolved = True
        conflict.resolved_at = datetime.now(timezone.utc)
        conflict.resolution_result = result
        return conflict

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        """Return sync state statistics."""
        status_counts: dict[str, int] = {}
        for record in self._sync_records.values():
            status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1

        return {
            "total_records": len(self._sync_records),
            "total_operations": len(self._operations),
            "total_conflicts": len(self._conflicts),
            "unresolved_conflicts": len(self.get_unresolved_conflicts()),
            "status_counts": status_counts,
        }
