"""Conflict resolution engine for bidirectional sync."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from blurt.models.sync import (
    ConflictRecord,
    ConflictResolutionStrategy,
    SyncRecord,
)

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Resolves conflicts when both Blurt and an external service
    have modified the same entity since the last sync.

    Supports multiple strategies:
    - LATEST_WINS: whichever side was modified most recently wins
    - BLURT_WINS: Blurt's version always takes precedence
    - EXTERNAL_WINS: external service's version always takes precedence
    - MERGE: attempt field-level merge of non-conflicting changes
    - MANUAL: flag for user to resolve later
    """

    def __init__(
        self,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS,
    ) -> None:
        self.default_strategy = default_strategy
        self._strategy_handlers = {
            ConflictResolutionStrategy.LATEST_WINS: self._resolve_latest_wins,
            ConflictResolutionStrategy.BLURT_WINS: self._resolve_blurt_wins,
            ConflictResolutionStrategy.EXTERNAL_WINS: self._resolve_external_wins,
            ConflictResolutionStrategy.MERGE: self._resolve_merge,
            ConflictResolutionStrategy.MANUAL: self._resolve_manual,
        }

    def detect_conflict(self, sync_record: SyncRecord) -> bool:
        """Detect if a sync record has a conflict (both sides modified)."""
        return sync_record.has_conflict

    async def resolve(
        self,
        conflict: ConflictRecord,
        strategy: ConflictResolutionStrategy | None = None,
    ) -> ConflictRecord:
        """Resolve a conflict using the specified strategy.

        Returns the updated ConflictRecord with resolution result.
        """
        strategy = strategy or conflict.resolution_strategy or self.default_strategy
        handler = self._strategy_handlers.get(strategy)

        if handler is None:
            logger.error("Unknown conflict resolution strategy: %s", strategy)
            conflict.resolution_strategy = ConflictResolutionStrategy.MANUAL
            return conflict

        logger.info(
            "Resolving conflict %s with strategy %s",
            conflict.id,
            strategy.value,
        )

        result = await handler(conflict)
        conflict.resolution_result = result
        conflict.resolved = strategy != ConflictResolutionStrategy.MANUAL
        if conflict.resolved:
            conflict.resolved_at = datetime.now(timezone.utc)

        return conflict

    def create_conflict_record(
        self,
        sync_record: SyncRecord,
        blurt_data: dict[str, Any],
        external_data: dict[str, Any],
        strategy: ConflictResolutionStrategy | None = None,
    ) -> ConflictRecord:
        """Create a conflict record from a sync record and the two versions."""
        return ConflictRecord(
            sync_record_id=sync_record.id,
            provider=sync_record.provider,
            blurt_data=blurt_data,
            external_data=external_data,
            resolution_strategy=strategy or self.default_strategy,
        )

    async def _resolve_latest_wins(
        self, conflict: ConflictRecord
    ) -> dict[str, Any]:
        """Most recently modified version wins."""
        blurt_time = conflict.blurt_data.get("modified_at")
        external_time = conflict.external_data.get("modified_at")

        if blurt_time and external_time:
            blurt_dt = _parse_datetime(blurt_time)
            external_dt = _parse_datetime(external_time)
            if blurt_dt >= external_dt:
                return {"winner": "blurt", "data": conflict.blurt_data}
            return {"winner": "external", "data": conflict.external_data}

        # If we can't compare timestamps, default to Blurt
        return {"winner": "blurt", "data": conflict.blurt_data}

    async def _resolve_blurt_wins(
        self, conflict: ConflictRecord
    ) -> dict[str, Any]:
        """Blurt's version always wins."""
        return {"winner": "blurt", "data": conflict.blurt_data}

    async def _resolve_external_wins(
        self, conflict: ConflictRecord
    ) -> dict[str, Any]:
        """External service's version always wins."""
        return {"winner": "external", "data": conflict.external_data}

    async def _resolve_merge(
        self, conflict: ConflictRecord
    ) -> dict[str, Any]:
        """Attempt to merge non-conflicting field changes.

        For fields that only changed on one side, take that change.
        For fields that changed on both sides, mark as needing manual resolution.
        """
        blurt = conflict.blurt_data
        external = conflict.external_data
        base = blurt.get("_base", {})

        merged: dict[str, Any] = {}
        field_conflicts: list[str] = []
        all_keys = set(blurt.keys()) | set(external.keys())

        for key in all_keys:
            if key.startswith("_"):
                continue

            blurt_val = blurt.get(key)
            external_val = external.get(key)
            base_val = base.get(key) if base else None

            # Both same — no conflict
            if blurt_val == external_val:
                merged[key] = blurt_val
            # Only blurt changed from base
            elif base_val is not None and blurt_val != base_val and external_val == base_val:
                merged[key] = blurt_val
            # Only external changed from base
            elif base_val is not None and external_val != base_val and blurt_val == base_val:
                merged[key] = external_val
            # Both changed — true conflict
            else:
                field_conflicts.append(key)
                # Default to blurt's value for conflicting fields
                merged[key] = blurt_val

        return {
            "winner": "merged",
            "data": merged,
            "field_conflicts": field_conflicts,
            "clean_merge": len(field_conflicts) == 0,
        }

    async def _resolve_manual(
        self, conflict: ConflictRecord
    ) -> dict[str, Any]:
        """Flag for manual resolution — don't auto-resolve."""
        return {
            "winner": "pending",
            "requires_manual_resolution": True,
            "blurt_data": conflict.blurt_data,
            "external_data": conflict.external_data,
        }


def _parse_datetime(value: str | datetime) -> datetime:
    """Parse a datetime from string or return as-is."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)
