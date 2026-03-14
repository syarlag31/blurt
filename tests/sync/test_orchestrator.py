"""Tests for the bidirectional sync orchestrator."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any

from blurt.models.sync import (
    ConflictResolutionStrategy,
    SyncDirection,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
)
from blurt.sync.orchestrator import SyncOrchestrator
from blurt.sync.conflict import ConflictResolver
from blurt.sync.providers import (
    SyncProviderAdapter,
)
from blurt.sync.state import SyncStateStore


# --- Fixtures ---


class MockAdapter(SyncProviderAdapter):
    """A mock adapter for testing."""

    def __init__(self, provider_type: SyncProvider) -> None:
        self._provider_type = provider_type
        self.push_calls: list[SyncOperation] = []
        self.pull_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.push_result: dict[str, Any] = {
            "external_id": "ext_123",
            "external_version": "v1",
        }
        self.pull_result: dict[str, Any] = {
            "external_id": "ext_123",
            "title": "Test",
            "modified_at": datetime.now(timezone.utc).isoformat(),
        }
        self.changes_result: list[dict[str, Any]] = []
        self.should_fail = False

    @property
    def provider(self) -> SyncProvider:
        return self._provider_type

    async def push(self, operation: SyncOperation) -> dict[str, Any]:
        self.push_calls.append(operation)
        if self.should_fail:
            raise RuntimeError("Push failed")
        return self.push_result

    async def pull(self, external_id: str) -> dict[str, Any]:
        self.pull_calls.append(external_id)
        return self.pull_result

    async def fetch_changes_since(self, since: datetime) -> list[dict[str, Any]]:
        return self.changes_result

    async def delete(self, external_id: str) -> bool:
        self.delete_calls.append(external_id)
        return True

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def gcal_adapter():
    return MockAdapter(SyncProvider.GOOGLE_CALENDAR)


@pytest.fixture
def notion_adapter():
    return MockAdapter(SyncProvider.NOTION)


@pytest.fixture
def orchestrator(gcal_adapter, notion_adapter):
    orch = SyncOrchestrator()
    orch.register_provider(gcal_adapter)
    orch.register_provider(notion_adapter)
    return orch


# --- Intent routing tests ---


class TestShouldSync:
    """Tests for intent-to-sync routing."""

    def test_event_intent_triggers_sync(self, orchestrator):
        assert orchestrator.should_sync("event", 0.9) is True

    def test_reminder_intent_triggers_sync(self, orchestrator):
        assert orchestrator.should_sync("reminder", 0.9) is True

    def test_task_intent_triggers_sync(self, orchestrator):
        assert orchestrator.should_sync("task", 0.9) is True

    def test_update_intent_triggers_sync(self, orchestrator):
        assert orchestrator.should_sync("update", 0.9) is True

    def test_journal_intent_does_not_trigger_sync(self, orchestrator):
        assert orchestrator.should_sync("journal", 0.9) is False

    def test_idea_intent_does_not_trigger_sync(self, orchestrator):
        assert orchestrator.should_sync("idea", 0.9) is False

    def test_question_intent_does_not_trigger_sync(self, orchestrator):
        assert orchestrator.should_sync("question", 0.9) is False

    def test_low_confidence_does_not_trigger_sync(self, orchestrator):
        assert orchestrator.should_sync("event", 0.3) is False

    def test_unknown_intent_does_not_trigger_sync(self, orchestrator):
        assert orchestrator.should_sync("unknown_thing", 0.9) is False

    def test_boundary_confidence_triggers_sync(self, orchestrator):
        assert orchestrator.should_sync("event", 0.5) is True

    def test_get_target_providers_event(self, orchestrator):
        providers = orchestrator.get_target_providers("event")
        assert SyncProvider.GOOGLE_CALENDAR in providers

    def test_get_target_providers_task(self, orchestrator):
        providers = orchestrator.get_target_providers("task")
        assert SyncProvider.NOTION in providers

    def test_get_target_providers_update_includes_both(self, orchestrator):
        providers = orchestrator.get_target_providers("update")
        assert SyncProvider.GOOGLE_CALENDAR in providers
        assert SyncProvider.NOTION in providers


# --- Outbound sync tests ---


class TestOutboundSync:
    """Tests for Blurt -> external sync."""

    @pytest.mark.asyncio
    async def test_event_creates_outbound_sync(self, orchestrator, gcal_adapter):
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt_1",
            intent="event",
            confidence=0.9,
            payload={"title": "Team standup", "datetime": "2026-03-14T10:00:00Z"},
        )
        assert len(triggers) == 1
        assert triggers[0].provider == SyncProvider.GOOGLE_CALENDAR
        assert len(gcal_adapter.push_calls) == 1

    @pytest.mark.asyncio
    async def test_task_creates_notion_sync(self, orchestrator, notion_adapter):
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt_2",
            intent="task",
            confidence=0.85,
            payload={"title": "Review PR"},
        )
        assert len(triggers) == 1
        assert triggers[0].provider == SyncProvider.NOTION
        assert len(notion_adapter.push_calls) == 1

    @pytest.mark.asyncio
    async def test_update_syncs_to_both_providers(self, orchestrator, gcal_adapter, notion_adapter):
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt_3",
            intent="update",
            confidence=0.9,
            payload={"title": "Meeting moved to 3pm"},
        )
        assert len(triggers) == 2
        assert len(gcal_adapter.push_calls) == 1
        assert len(notion_adapter.push_calls) == 1

    @pytest.mark.asyncio
    async def test_non_syncable_intent_no_trigger(self, orchestrator, gcal_adapter, notion_adapter):
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt_4",
            intent="journal",
            confidence=0.95,
            payload={"text": "Had a great day"},
        )
        assert len(triggers) == 0
        assert len(gcal_adapter.push_calls) == 0
        assert len(notion_adapter.push_calls) == 0

    @pytest.mark.asyncio
    async def test_sync_record_created_on_first_sync(self, orchestrator):
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_5",
            intent="event",
            confidence=0.9,
            payload={"title": "Lunch"},
        )
        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_5", SyncProvider.GOOGLE_CALENDAR
        )
        assert record is not None
        assert record.status == SyncStatus.COMPLETED
        assert record.external_id is not None

    @pytest.mark.asyncio
    async def test_sync_record_version_increments_on_update(self, orchestrator):
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_6",
            intent="event",
            confidence=0.9,
            payload={"title": "Lunch v1"},
        )
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_6",
            intent="event",
            confidence=0.9,
            payload={"title": "Lunch v2"},
        )
        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_6", SyncProvider.GOOGLE_CALENDAR
        )
        assert record is not None
        assert record.blurt_version == 2

    @pytest.mark.asyncio
    async def test_failed_sync_records_error(self, orchestrator, gcal_adapter):
        gcal_adapter.should_fail = True
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_7",
            intent="event",
            confidence=0.9,
            payload={"title": "Will fail"},
        )
        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_7", SyncProvider.GOOGLE_CALENDAR
        )
        assert record is not None
        assert record.status == SyncStatus.FAILED
        assert record.error_message is not None

    @pytest.mark.asyncio
    async def test_entities_passed_to_trigger(self, orchestrator):
        entities = [{"type": "person", "name": "Alice"}]
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt_8",
            intent="event",
            confidence=0.9,
            payload={"title": "Meeting with Alice"},
            entities=entities,
        )
        assert triggers[0].entities == entities


# --- Inbound sync tests ---


class TestInboundSync:
    """Tests for external -> Blurt sync."""

    @pytest.mark.asyncio
    async def test_inbound_creates_sync_record(self, orchestrator, gcal_adapter):
        gcal_adapter.changes_result = [
            {"external_id": "gcal_abc", "title": "External event", "start": "2026-03-15T14:00:00Z"}
        ]
        ops = await orchestrator.pull_inbound_changes(SyncProvider.GOOGLE_CALENDAR)
        assert len(ops) == 1
        assert ops[0].direction == SyncDirection.INBOUND

        record = orchestrator.state_store.get_sync_record_by_external_id(
            "gcal_abc", SyncProvider.GOOGLE_CALENDAR
        )
        assert record is not None

    @pytest.mark.asyncio
    async def test_inbound_callback_called(self, orchestrator, gcal_adapter):
        received = []

        def on_change(blurt_id, provider, data):
            received.append((blurt_id, provider, data))

        orchestrator.register_inbound_callback(on_change)
        gcal_adapter.changes_result = [
            {"external_id": "gcal_xyz", "title": "New external event"}
        ]
        await orchestrator.pull_inbound_changes(SyncProvider.GOOGLE_CALENDAR)
        assert len(received) == 1
        assert received[0][1] == SyncProvider.GOOGLE_CALENDAR

    @pytest.mark.asyncio
    async def test_inbound_no_changes_returns_empty(self, orchestrator):
        ops = await orchestrator.pull_inbound_changes()
        assert ops == []

    @pytest.mark.asyncio
    async def test_inbound_skips_missing_external_id(self, orchestrator, gcal_adapter):
        gcal_adapter.changes_result = [{"title": "No ID event"}]
        ops = await orchestrator.pull_inbound_changes(SyncProvider.GOOGLE_CALENDAR)
        assert len(ops) == 0


# --- Conflict resolution tests ---


class TestConflictResolution:
    """Tests for conflict detection and resolution."""

    @pytest.mark.asyncio
    async def test_conflict_detected_on_both_sides_modified(self, orchestrator, gcal_adapter):
        """When both Blurt and external modified since last sync, conflict is detected."""
        # First sync
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_conflict",
            intent="event",
            confidence=0.9,
            payload={"title": "Original event"},
        )

        # Simulate both sides modified after last sync
        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_conflict", SyncProvider.GOOGLE_CALENDAR
        )
        assert record is not None
        # Set both modifications after last_synced_at
        future = datetime.now(timezone.utc) + timedelta(seconds=5)
        record.last_blurt_modified_at = future
        record.last_external_modified_at = future + timedelta(seconds=1)
        orchestrator.state_store.upsert_sync_record(record)

        assert record.has_conflict is True

    @pytest.mark.asyncio
    async def test_latest_wins_strategy(self):
        resolver = ConflictResolver(ConflictResolutionStrategy.LATEST_WINS)
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(hours=1)

        from blurt.models.sync import ConflictRecord

        conflict = ConflictRecord(
            sync_record_id="test",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={"title": "Blurt version", "modified_at": now.isoformat()},
            external_data={"title": "External version", "modified_at": earlier.isoformat()},
        )
        resolved = await resolver.resolve(conflict, ConflictResolutionStrategy.LATEST_WINS)
        assert resolved.resolved is True
        assert resolved.resolution_result is not None
        assert resolved.resolution_result["winner"] == "blurt"

    @pytest.mark.asyncio
    async def test_blurt_wins_strategy(self):
        resolver = ConflictResolver()
        from blurt.models.sync import ConflictRecord

        conflict = ConflictRecord(
            sync_record_id="test",
            provider=SyncProvider.NOTION,
            blurt_data={"title": "Blurt version"},
            external_data={"title": "External version"},
        )
        resolved = await resolver.resolve(conflict, ConflictResolutionStrategy.BLURT_WINS)
        assert resolved.resolved is True
        assert resolved.resolution_result is not None
        assert resolved.resolution_result["winner"] == "blurt"

    @pytest.mark.asyncio
    async def test_external_wins_strategy(self):
        resolver = ConflictResolver()
        from blurt.models.sync import ConflictRecord

        conflict = ConflictRecord(
            sync_record_id="test",
            provider=SyncProvider.NOTION,
            blurt_data={"title": "Blurt version"},
            external_data={"title": "External version"},
        )
        resolved = await resolver.resolve(conflict, ConflictResolutionStrategy.EXTERNAL_WINS)
        assert resolved.resolved is True
        assert resolved.resolution_result is not None
        assert resolved.resolution_result["winner"] == "external"

    @pytest.mark.asyncio
    async def test_merge_strategy_clean(self):
        """Non-overlapping changes merge cleanly."""
        resolver = ConflictResolver()
        from blurt.models.sync import ConflictRecord

        conflict = ConflictRecord(
            sync_record_id="test",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={
                "title": "Updated title",
                "location": "Office",
                "_base": {"title": "Original", "location": "Office"},
            },
            external_data={
                "title": "Original",
                "location": "Home",
                "_base": {"title": "Original", "location": "Office"},
            },
        )
        resolved = await resolver.resolve(conflict, ConflictResolutionStrategy.MERGE)
        assert resolved.resolved is True
        result = resolved.resolution_result
        assert result is not None
        assert result["winner"] == "merged"
        assert result["data"]["title"] == "Updated title"
        assert result["data"]["location"] == "Home"
        assert result["clean_merge"] is True

    @pytest.mark.asyncio
    async def test_manual_strategy_not_resolved(self):
        resolver = ConflictResolver()
        from blurt.models.sync import ConflictRecord

        conflict = ConflictRecord(
            sync_record_id="test",
            provider=SyncProvider.NOTION,
            blurt_data={"title": "A"},
            external_data={"title": "B"},
        )
        resolved = await resolver.resolve(conflict, ConflictResolutionStrategy.MANUAL)
        assert resolved.resolved is False
        assert resolved.resolution_result is not None
        assert resolved.resolution_result["requires_manual_resolution"] is True


# --- Retry tests ---


class TestRetryFailed:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_failed_operations(self, orchestrator, gcal_adapter):
        # First attempt fails
        gcal_adapter.should_fail = True
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_retry",
            intent="event",
            confidence=0.9,
            payload={"title": "Retry me"},
        )

        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_retry", SyncProvider.GOOGLE_CALENDAR
        )
        assert record.status == SyncStatus.FAILED
        assert record.retry_count == 1

        # Retry succeeds
        gcal_adapter.should_fail = False
        retried = await orchestrator.retry_failed()
        assert len(retried) >= 1

        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_retry", SyncProvider.GOOGLE_CALENDAR
        )
        assert record.status == SyncStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, orchestrator, gcal_adapter):
        gcal_adapter.should_fail = True

        # Fail initial + 3 retries (max_retries=3)
        await orchestrator.on_blurt_classified(
            blurt_id="blurt_maxretry",
            intent="event",
            confidence=0.9,
            payload={"title": "Always fails"},
        )
        for _ in range(3):
            await orchestrator.retry_failed()

        record = orchestrator.state_store.get_sync_record_by_blurt_id(
            "blurt_maxretry", SyncProvider.GOOGLE_CALENDAR
        )
        assert record.can_retry is False

        # No more retries should happen
        retried = await orchestrator.retry_failed()
        assert len(retried) == 0


# --- State store tests ---


class TestSyncStateStore:
    """Tests for the sync state store."""

    def test_upsert_and_retrieve(self):
        store = SyncStateStore()
        record = SyncRecord(
            blurt_id="b1",
            provider=SyncProvider.GOOGLE_CALENDAR,
        )
        store.upsert_sync_record(record)
        retrieved = store.get_sync_record(record.id)
        assert retrieved is not None
        assert retrieved.blurt_id == "b1"

    def test_lookup_by_blurt_id(self):
        store = SyncStateStore()
        record = SyncRecord(
            blurt_id="b2",
            provider=SyncProvider.NOTION,
        )
        store.upsert_sync_record(record)
        found = store.get_sync_record_by_blurt_id("b2", SyncProvider.NOTION)
        assert found is not None
        assert found.id == record.id

    def test_lookup_by_external_id(self):
        store = SyncStateStore()
        record = SyncRecord(
            blurt_id="b3",
            provider=SyncProvider.GOOGLE_CALENDAR,
            external_id="gcal_999",
        )
        store.upsert_sync_record(record)
        found = store.get_sync_record_by_external_id("gcal_999", SyncProvider.GOOGLE_CALENDAR)
        assert found is not None

    def test_mark_synced(self):
        store = SyncStateStore()
        record = SyncRecord(blurt_id="b4", provider=SyncProvider.NOTION)
        store.upsert_sync_record(record)
        store.mark_synced(record.id, external_id="notion_1", external_version="v2")
        updated = store.get_sync_record(record.id)
        assert updated is not None
        assert updated.status == SyncStatus.COMPLETED
        assert updated.external_id == "notion_1"
        assert updated.last_synced_at is not None

    def test_mark_failed_increments_retry(self):
        store = SyncStateStore()
        record = SyncRecord(blurt_id="b5", provider=SyncProvider.NOTION)
        store.upsert_sync_record(record)
        store.mark_failed(record.id, "timeout")
        updated = store.get_sync_record(record.id)
        assert updated is not None
        assert updated.status == SyncStatus.FAILED
        assert updated.retry_count == 1
        assert updated.error_message == "timeout"

    def test_stats(self):
        store = SyncStateStore()
        store.upsert_sync_record(SyncRecord(blurt_id="s1", provider=SyncProvider.NOTION))
        store.upsert_sync_record(SyncRecord(blurt_id="s2", provider=SyncProvider.GOOGLE_CALENDAR))
        stats = store.stats()
        assert stats["total_records"] == 2


# --- Health check tests ---


class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_provider_status(self, orchestrator):
        health = await orchestrator.health()
        assert "providers" in health
        assert "state" in health

    def test_stats(self, orchestrator):
        stats = orchestrator.stats()
        assert "registered_providers" in stats
        assert len(stats["registered_providers"]) == 2


# --- Sync model tests ---


class TestSyncRecord:
    """Tests for SyncRecord model properties."""

    def test_needs_outbound_when_never_synced(self):
        record = SyncRecord(blurt_id="x", provider=SyncProvider.NOTION)
        assert record.needs_outbound_sync is True

    def test_needs_outbound_when_modified_after_sync(self):
        now = datetime.now(timezone.utc)
        record = SyncRecord(
            blurt_id="x",
            provider=SyncProvider.NOTION,
            last_synced_at=now - timedelta(minutes=5),
            last_blurt_modified_at=now,
        )
        assert record.needs_outbound_sync is True

    def test_no_outbound_when_synced_after_modification(self):
        now = datetime.now(timezone.utc)
        record = SyncRecord(
            blurt_id="x",
            provider=SyncProvider.NOTION,
            last_synced_at=now,
            last_blurt_modified_at=now - timedelta(minutes=5),
        )
        assert record.needs_outbound_sync is False

    def test_has_conflict_when_both_modified(self):
        now = datetime.now(timezone.utc)
        record = SyncRecord(
            blurt_id="x",
            provider=SyncProvider.NOTION,
            last_synced_at=now - timedelta(minutes=10),
            last_blurt_modified_at=now,
            last_external_modified_at=now - timedelta(minutes=1),
        )
        assert record.has_conflict is True

    def test_no_conflict_when_only_blurt_modified(self):
        now = datetime.now(timezone.utc)
        record = SyncRecord(
            blurt_id="x",
            provider=SyncProvider.NOTION,
            last_synced_at=now - timedelta(minutes=10),
            last_blurt_modified_at=now,
        )
        assert record.has_conflict is False
