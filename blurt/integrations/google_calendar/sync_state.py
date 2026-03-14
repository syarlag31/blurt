"""Sync state tracking for Google Calendar change detection.

Tracks sync tokens, per-event content hashes, and poll timing
to enable efficient incremental polling for bidirectional sync.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from blurt.core.models import BlurtEvent


class ChangeType(str, Enum):
    """Type of change detected during polling."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class EventChange(BaseModel):
    """Represents a detected change in a calendar event.

    Contains the change type, the mapped BlurtEvent (if available),
    the Google Calendar event ID, and content hashes for audit.
    """
    change_type: ChangeType
    event: BlurtEvent | None = None
    google_calendar_id: str = ""
    previous_hash: str | None = None
    current_hash: str | None = None
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}


class SyncState(BaseModel):
    """Tracks the state of calendar sync for incremental polling.

    Uses Google Calendar's sync tokens for efficient incremental sync.
    Stores per-event content hashes for real change detection.
    """
    user_id: str
    calendar_id: str = "primary"
    sync_token: str | None = None
    last_poll_at: datetime | None = None
    last_full_sync_at: datetime | None = None
    event_hashes: dict[str, str] = Field(default_factory=dict)  # gcal_event_id -> content_hash
    poll_interval_seconds: int = 300  # 5 minutes default
    consecutive_errors: int = 0

    def remove_event(self, event_id: str) -> bool:
        """Remove an event from tracking. Returns True if it was tracked."""
        return self.event_hashes.pop(event_id, None) is not None

    def reset(self) -> None:
        """Reset sync state for a full re-sync."""
        self.sync_token = None
        self.event_hashes.clear()
        self.consecutive_errors = 0

    @property
    def tracked_event_count(self) -> int:
        return len(self.event_hashes)


class PollResult(BaseModel):
    """Result of a polling operation with detected changes."""
    changes: list[EventChange] = Field(default_factory=list)
    new_sync_token: str | None = None
    polled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_full_sync: bool = False
    events_checked: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    @property
    def created_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.CREATED)

    @property
    def updated_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.UPDATED)

    @property
    def deleted_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.DELETED)
