"""Sync-related domain models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SyncProvider(str, Enum):
    """External services Blurt syncs with."""

    GOOGLE_CALENDAR = "google_calendar"
    NOTION = "notion"


class SyncDirection(str, Enum):
    """Direction of a sync operation."""

    OUTBOUND = "outbound"  # Blurt -> external
    INBOUND = "inbound"  # external -> Blurt
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(str, Enum):
    """Current status of a sync operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"
    SKIPPED = "skipped"


class ConflictResolutionStrategy(str, Enum):
    """How to resolve sync conflicts."""

    LATEST_WINS = "latest_wins"  # Most recently modified version wins
    BLURT_WINS = "blurt_wins"  # Blurt's version always wins
    EXTERNAL_WINS = "external_wins"  # External service's version wins
    MERGE = "merge"  # Attempt to merge changes
    MANUAL = "manual"  # Flag for manual resolution


class SyncRecord(BaseModel):
    """Tracks the sync state of a single entity between Blurt and an external service."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    blurt_id: str = Field(description="ID of the Blurt entity (task, event, etc.)")
    provider: SyncProvider
    external_id: str | None = Field(
        default=None,
        description="ID in the external system (e.g., Google Calendar event ID)",
    )
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    status: SyncStatus = SyncStatus.PENDING
    blurt_version: int = Field(
        default=1,
        description="Version counter for the Blurt-side entity",
    )
    external_version: str | None = Field(
        default=None,
        description="Version/etag from external system",
    )
    last_synced_at: datetime | None = None
    last_blurt_modified_at: datetime | None = None
    last_external_modified_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def needs_outbound_sync(self) -> bool:
        """Check if Blurt changes need pushing to external."""
        if self.last_synced_at is None:
            return True
        if self.last_blurt_modified_at and self.last_blurt_modified_at > self.last_synced_at:
            return True
        return False

    @property
    def needs_inbound_sync(self) -> bool:
        """Check if external changes need pulling to Blurt."""
        if self.last_external_modified_at is None:
            return False
        if self.last_synced_at is None:
            return True
        return self.last_external_modified_at > self.last_synced_at

    @property
    def has_conflict(self) -> bool:
        """Check if both sides changed since last sync."""
        return self.needs_outbound_sync and self.needs_inbound_sync

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries


class SyncOperation(BaseModel):
    """A single sync operation to be executed."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sync_record_id: str
    provider: SyncProvider
    direction: SyncDirection
    operation_type: str = Field(description="create, update, delete")
    payload: dict[str, Any] = Field(default_factory=dict)
    status: SyncStatus = SyncStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None


class ConflictRecord(BaseModel):
    """Records a detected conflict between Blurt and an external service."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sync_record_id: str
    provider: SyncProvider
    blurt_data: dict[str, Any]
    external_data: dict[str, Any]
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS
    resolved: bool = False
    resolved_at: datetime | None = None
    resolution_result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SyncTrigger(BaseModel):
    """Describes what triggered a sync operation."""

    blurt_id: str
    intent: str
    confidence: float
    entities: list[dict[str, Any]] = Field(default_factory=list)
    provider: SyncProvider
    direction: SyncDirection = SyncDirection.OUTBOUND
    payload: dict[str, Any] = Field(default_factory=dict)
