"""Local Google Calendar client for local-only mode.

Provides a file-backed calendar implementation that mirrors the
GoogleCalendarClient interface. Events are stored locally as JSON,
ensuring full feature parity with no data leakage to Google.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalCalendarClient:
    """File-backed local calendar client for local-only mode.

    Stores events as JSON files in the data directory. Implements the
    same interface as GoogleCalendarClient so the sync pipeline works
    unchanged in local mode.
    """

    def __init__(self, data_dir: Path | None = None, calendar_id: str = "local") -> None:
        self._data_dir = (data_dir or Path.home() / ".blurt") / "calendar"
        self._calendar_id = calendar_id
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def events_file(self) -> Path:
        return self._data_dir / f"{self._calendar_id}_events.json"

    def _load_events(self) -> dict[str, Any]:
        """Load events from local JSON file."""
        if self.events_file.exists():
            return json.loads(self.events_file.read_text())
        return {}

    def _save_events(self, events: dict[str, Any]) -> None:
        """Save events to local JSON file."""
        self.events_file.write_text(json.dumps(events, indent=2, default=str))

    def create_event(self, event: Any, **kwargs: Any) -> Any:
        """Create an event in the local calendar store.

        Args:
            event: A BlurtEvent instance (or any object with title, start_time, etc.)

        Returns:
            The event with google_calendar_id populated (local UUID).
        """
        events = self._load_events()
        local_id = f"local_{uuid.uuid4().hex[:12]}"

        event_data = {
            "id": local_id,
            "title": getattr(event, "title", "Untitled"),
            "start_time": str(getattr(event, "start_time", "")),
            "end_time": str(getattr(event, "end_time", "")),
            "description": getattr(event, "description", ""),
            "status": "confirmed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        events[local_id] = event_data
        self._save_events(events)

        # Populate the event's external ID
        if hasattr(event, "google_calendar_id"):
            event.google_calendar_id = local_id
        if hasattr(event, "synced_at"):
            event.synced_at = datetime.now(timezone.utc)

        logger.info("Created local calendar event: %s (id=%s)", event_data["title"], local_id)
        return event

    def update_event(self, event: Any, **kwargs: Any) -> Any:
        """Update an event in the local calendar store."""
        gcal_id = getattr(event, "google_calendar_id", None)
        if not gcal_id:
            raise ValueError("Cannot update event without google_calendar_id")

        events = self._load_events()
        if gcal_id not in events:
            raise KeyError(f"Event not found: {gcal_id}")

        events[gcal_id].update({
            "title": getattr(event, "title", events[gcal_id].get("title")),
            "start_time": str(getattr(event, "start_time", "")),
            "end_time": str(getattr(event, "end_time", "")),
            "description": getattr(event, "description", ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save_events(events)

        if hasattr(event, "synced_at"):
            event.synced_at = datetime.now(timezone.utc)

        logger.info("Updated local calendar event: %s (id=%s)", events[gcal_id]["title"], gcal_id)
        return event

    def create_or_update_event(self, event: Any, **kwargs: Any) -> Any:
        """Create or update depending on whether event has an external ID."""
        if getattr(event, "google_calendar_id", None):
            return self.update_event(event, **kwargs)
        return self.create_event(event, **kwargs)

    def list_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """List events from the local store, optionally filtered by time range."""
        events = self._load_events()
        result = list(events.values())

        # Simple time filtering
        if time_min or time_max:
            filtered = []
            for ev in result:
                start = ev.get("start_time", "")
                if start and time_min and start < time_min.isoformat():
                    continue
                if start and time_max and start > time_max.isoformat():
                    continue
                filtered.append(ev)
            result = filtered

        return result

    def delete_event(self, event_id: str) -> bool:
        """Delete an event from the local store."""
        events = self._load_events()
        if event_id in events:
            del events[event_id]
            self._save_events(events)
            logger.info("Deleted local calendar event: %s", event_id)
            return True
        return False
