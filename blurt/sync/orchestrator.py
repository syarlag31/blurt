"""Bidirectional sync orchestrator.

Coordinates sync between Blurt and external services (Google Calendar, Notion).
Handles:
- Triggering sync when relevant Blurt intents are classified
- Outbound sync (Blurt -> external)
- Inbound sync (external -> Blurt)
- Conflict detection and resolution
- Retry with backoff
- Sync state tracking
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from blurt.models.intents import SYNCABLE_INTENTS, BlurtIntent
from blurt.models.sync import (
    SyncDirection,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
    SyncTrigger,
)
from blurt.sync.conflict import ConflictResolver
from blurt.sync.providers import SyncProviderAdapter
from blurt.sync.state import SyncStateStore

logger = logging.getLogger(__name__)

# Maps intents to their target sync providers
DEFAULT_INTENT_PROVIDER_MAP: dict[BlurtIntent, list[SyncProvider]] = {
    BlurtIntent.EVENT: [SyncProvider.GOOGLE_CALENDAR],
    BlurtIntent.REMINDER: [SyncProvider.GOOGLE_CALENDAR],
    BlurtIntent.TASK: [SyncProvider.NOTION],
    BlurtIntent.UPDATE: [SyncProvider.NOTION, SyncProvider.GOOGLE_CALENDAR],
}

# Callback type for notifying the rest of the system about inbound changes
InboundChangeCallback = Callable[[str, SyncProvider, dict[str, Any]], Any]


class SyncOrchestrator:
    """Orchestrates bidirectional sync between Blurt and external services.

    Usage:
        orchestrator = SyncOrchestrator()
        orchestrator.register_provider(google_cal_adapter)
        orchestrator.register_provider(notion_adapter)

        # Triggered automatically when a Blurt is classified:
        await orchestrator.on_blurt_classified(blurt_id, intent, confidence, payload)

        # Periodic inbound sync:
        await orchestrator.pull_inbound_changes()

        # Retry failed syncs:
        await orchestrator.retry_failed()
    """

    def __init__(
        self,
        state_store: SyncStateStore | None = None,
        conflict_resolver: ConflictResolver | None = None,
        intent_provider_map: dict[BlurtIntent, list[SyncProvider]] | None = None,
    ) -> None:
        self._state = state_store or SyncStateStore()
        self._conflict_resolver = conflict_resolver or ConflictResolver()
        self._providers: dict[SyncProvider, SyncProviderAdapter] = {}
        self._intent_provider_map = intent_provider_map or DEFAULT_INTENT_PROVIDER_MAP
        self._inbound_callbacks: list[InboundChangeCallback] = []

    @property
    def state_store(self) -> SyncStateStore:
        """Expose state store for inspection."""
        return self._state

    def register_provider(self, adapter: SyncProviderAdapter) -> None:
        """Register an external service adapter."""
        self._providers[adapter.provider] = adapter
        logger.info("Registered sync provider: %s", adapter.provider.value)

    def register_inbound_callback(self, callback: InboundChangeCallback) -> None:
        """Register a callback for when inbound changes are received."""
        self._inbound_callbacks.append(callback)

    def get_provider(self, provider: SyncProvider) -> SyncProviderAdapter | None:
        """Get a registered provider adapter."""
        return self._providers.get(provider)

    # --- Intent-triggered sync ---

    def should_sync(self, intent: str, confidence: float) -> bool:
        """Determine if an intent classification should trigger a sync.

        Only syncable intents with sufficient confidence trigger sync.
        """
        try:
            blurt_intent = BlurtIntent(intent)
        except ValueError:
            return False

        if blurt_intent not in SYNCABLE_INTENTS:
            return False

        # Require reasonable confidence to avoid spurious syncs
        if confidence < 0.5:
            return False

        # Check that we have at least one provider for this intent
        providers = self._intent_provider_map.get(blurt_intent, [])
        return any(p in self._providers for p in providers)

    def get_target_providers(self, intent: str) -> list[SyncProvider]:
        """Get which providers a given intent should sync to."""
        try:
            blurt_intent = BlurtIntent(intent)
        except ValueError:
            return []
        providers = self._intent_provider_map.get(blurt_intent, [])
        # Only return providers that are actually registered
        return [p for p in providers if p in self._providers]

    async def on_blurt_classified(
        self,
        blurt_id: str,
        intent: str,
        confidence: float,
        payload: dict[str, Any],
        entities: list[dict[str, Any]] | None = None,
    ) -> list[SyncTrigger]:
        """Called when a Blurt is classified. Triggers sync if appropriate.

        This is the main entry point from the classification pipeline.
        Returns a list of SyncTrigger objects describing what was triggered.
        """
        if not self.should_sync(intent, confidence):
            return []

        providers = self.get_target_providers(intent)
        triggers: list[SyncTrigger] = []

        for provider in providers:
            trigger = SyncTrigger(
                blurt_id=blurt_id,
                intent=intent,
                confidence=confidence,
                entities=entities or [],
                provider=provider,
                payload=payload,
            )
            triggers.append(trigger)

            # Create or update sync record
            await self._execute_outbound_sync(trigger)

        return triggers

    async def _execute_outbound_sync(self, trigger: SyncTrigger) -> SyncOperation | None:
        """Execute an outbound sync for a single trigger."""
        adapter = self._providers.get(trigger.provider)
        if adapter is None:
            logger.warning("No adapter registered for %s", trigger.provider.value)
            return None

        # Get or create sync record
        sync_record = self._state.get_sync_record_by_blurt_id(
            trigger.blurt_id, trigger.provider
        )

        if sync_record is None:
            # First time syncing this entity
            sync_record = SyncRecord(
                blurt_id=trigger.blurt_id,
                provider=trigger.provider,
                status=SyncStatus.PENDING,
                last_blurt_modified_at=datetime.now(timezone.utc),
            )
            self._state.upsert_sync_record(sync_record)
            operation_type = "create"
        else:
            # Update existing sync
            sync_record.blurt_version += 1
            sync_record.last_blurt_modified_at = datetime.now(timezone.utc)

            # Check for conflict before pushing
            if sync_record.has_conflict:
                return await self._handle_conflict_during_outbound(
                    sync_record, trigger.payload
                )

            sync_record.status = SyncStatus.IN_PROGRESS
            self._state.upsert_sync_record(sync_record)
            operation_type = "update"

        # Create the operation
        operation = SyncOperation(
            sync_record_id=sync_record.id,
            provider=trigger.provider,
            direction=SyncDirection.OUTBOUND,
            operation_type=operation_type,
            payload=trigger.payload,
            status=SyncStatus.IN_PROGRESS,
        )
        self._state.add_operation(operation)

        # Execute the push
        try:
            result = await adapter.push(operation)
            self._state.complete_operation(operation.id, result)
            self._state.mark_synced(
                sync_record.id,
                external_id=result.get("external_id"),
                external_version=result.get("external_version"),
            )
            logger.info(
                "Outbound sync completed: blurt=%s provider=%s external_id=%s",
                trigger.blurt_id,
                trigger.provider.value,
                result.get("external_id"),
            )
            return operation

        except Exception as exc:
            error_msg = str(exc)
            self._state.fail_operation(operation.id, error_msg)
            self._state.mark_failed(sync_record.id, error_msg)
            logger.error(
                "Outbound sync failed: blurt=%s provider=%s error=%s",
                trigger.blurt_id,
                trigger.provider.value,
                error_msg,
            )
            return operation

    async def _handle_conflict_during_outbound(
        self,
        sync_record: SyncRecord,
        blurt_payload: dict[str, Any],
    ) -> SyncOperation | None:
        """Handle a conflict detected during outbound sync."""
        adapter = self._providers.get(sync_record.provider)
        if adapter is None or sync_record.external_id is None:
            self._state.mark_conflict(sync_record.id)
            return None

        # Pull current external state
        try:
            external_data = await adapter.pull(sync_record.external_id)
        except Exception:
            logger.error("Failed to pull external data for conflict resolution")
            self._state.mark_conflict(sync_record.id)
            return None

        # Create conflict record
        conflict = self._conflict_resolver.create_conflict_record(
            sync_record,
            blurt_data={**blurt_payload, "modified_at": sync_record.last_blurt_modified_at},
            external_data={**external_data, "modified_at": sync_record.last_external_modified_at},
        )
        self._state.add_conflict(conflict)

        # Try to resolve
        resolved = await self._conflict_resolver.resolve(conflict)
        self._state.resolve_conflict(conflict.id, resolved.resolution_result or {})

        if not resolved.resolved:
            self._state.mark_conflict(sync_record.id)
            return None

        # Apply resolution: push the winning data
        winner_data = resolved.resolution_result or {}
        data = winner_data.get("data", blurt_payload)

        operation = SyncOperation(
            sync_record_id=sync_record.id,
            provider=sync_record.provider,
            direction=SyncDirection.OUTBOUND,
            operation_type="update",
            payload=data,
            status=SyncStatus.IN_PROGRESS,
        )
        self._state.add_operation(operation)

        try:
            result = await adapter.push(operation)
            self._state.complete_operation(operation.id, result)
            self._state.mark_synced(
                sync_record.id,
                external_id=result.get("external_id"),
                external_version=result.get("external_version"),
            )
            return operation
        except Exception as exc:
            self._state.fail_operation(operation.id, str(exc))
            self._state.mark_failed(sync_record.id, str(exc))
            return operation

    # --- Inbound sync ---

    async def pull_inbound_changes(
        self, provider: SyncProvider | None = None
    ) -> list[SyncOperation]:
        """Pull changes from external services.

        If provider is None, pulls from all registered providers.
        """
        providers = [provider] if provider else list(self._providers.keys())
        operations: list[SyncOperation] = []

        for prov in providers:
            adapter = self._providers.get(prov)
            if adapter is None:
                continue

            # Determine since timestamp from last sync
            since = self._get_last_inbound_sync_time(prov)

            try:
                changes = await adapter.fetch_changes_since(since)
            except Exception as exc:
                logger.error(
                    "Failed to pull changes from %s: %s",
                    prov.value,
                    str(exc),
                )
                continue

            for change in changes:
                op = await self._process_inbound_change(prov, change)
                if op:
                    operations.append(op)

        return operations

    async def _process_inbound_change(
        self,
        provider: SyncProvider,
        change: dict[str, Any],
    ) -> SyncOperation | None:
        """Process a single inbound change from an external service."""
        external_id = change.get("external_id") or change.get("id")
        if not external_id:
            logger.warning("Inbound change missing external_id, skipping")
            return None

        # Find existing sync record
        sync_record = self._state.get_sync_record_by_external_id(external_id, provider)

        if sync_record is None:
            # New entity from external — create a sync record for it
            sync_record = SyncRecord(
                blurt_id=f"inbound_{external_id}",
                provider=provider,
                external_id=external_id,
                status=SyncStatus.PENDING,
                last_external_modified_at=datetime.now(timezone.utc),
            )
            self._state.upsert_sync_record(sync_record)
            operation_type = "create"
        else:
            sync_record.last_external_modified_at = datetime.now(timezone.utc)

            # Check for conflict
            if sync_record.has_conflict:
                self._state.mark_conflict(sync_record.id)
                logger.info("Conflict detected for %s from %s", external_id, provider.value)
                return None

            self._state.upsert_sync_record(sync_record)
            operation_type = "update"

        operation = SyncOperation(
            sync_record_id=sync_record.id,
            provider=provider,
            direction=SyncDirection.INBOUND,
            operation_type=operation_type,
            payload=change,
            status=SyncStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )
        self._state.add_operation(operation)
        self._state.mark_synced(sync_record.id, external_id=external_id)

        # Notify listeners about inbound change
        for callback in self._inbound_callbacks:
            try:
                callback(sync_record.blurt_id, provider, change)
            except Exception:
                logger.exception("Inbound callback failed")

        return operation

    def _get_last_inbound_sync_time(self, provider: SyncProvider) -> datetime:
        """Get the most recent inbound sync timestamp for a provider."""
        latest = datetime(2000, 1, 1, tzinfo=timezone.utc)
        for record in self._state._sync_records.values():
            if record.provider == provider and record.last_synced_at:
                if record.last_synced_at > latest:
                    latest = record.last_synced_at
        return latest

    # --- Retry failed syncs ---

    async def retry_failed(
        self, provider: SyncProvider | None = None
    ) -> list[SyncOperation]:
        """Retry all failed sync operations that haven't exceeded max retries."""
        pending = self._state.get_pending_records(provider)
        operations: list[SyncOperation] = []

        for record in pending:
            if record.status != SyncStatus.FAILED or not record.can_retry:
                continue

            adapter = self._providers.get(record.provider)
            if adapter is None:
                continue

            # Get the last operation's payload
            past_ops = self._state.get_operations_for_record(record.id)
            if not past_ops:
                continue

            last_op = past_ops[-1]
            operation = SyncOperation(
                sync_record_id=record.id,
                provider=record.provider,
                direction=last_op.direction,
                operation_type=last_op.operation_type,
                payload=last_op.payload,
                status=SyncStatus.IN_PROGRESS,
            )
            self._state.add_operation(operation)

            try:
                result = await adapter.push(operation)
                self._state.complete_operation(operation.id, result)
                self._state.mark_synced(
                    record.id,
                    external_id=result.get("external_id"),
                    external_version=result.get("external_version"),
                )
                operations.append(operation)
            except Exception as exc:
                self._state.fail_operation(operation.id, str(exc))
                self._state.mark_failed(record.id, str(exc))
                operations.append(operation)

        return operations

    # --- Health & Stats ---

    async def health(self) -> dict[str, Any]:
        """Check health of all registered providers."""
        results: dict[str, bool] = {}
        for provider, adapter in self._providers.items():
            try:
                results[provider.value] = await adapter.health_check()
            except Exception:
                results[provider.value] = False
        return {
            "providers": results,
            "state": self._state.stats(),
        }

    def stats(self) -> dict[str, Any]:
        """Get sync orchestrator statistics."""
        return {
            "registered_providers": [p.value for p in self._providers],
            "state": self._state.stats(),
        }
