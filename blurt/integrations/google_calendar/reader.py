"""Google Calendar read and polling operations for bidirectional sync.

Implements the read path:
- List events with time range filtering and pagination
- Get individual events by ID
- Incremental sync using Google Calendar's sync tokens
- Polling with content-hash-based change detection
- Automatic full re-sync on invalidated sync tokens

Uses the googleapiclient Resource (consistent with the existing write path)
for all API interactions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from blurt.core.models import BlurtEvent
from blurt.integrations.google_calendar.auth import GoogleCalendarAuth
from blurt.integrations.google_calendar.client import (
    EventNotFoundError,
    GoogleCalendarError,
)
from blurt.integrations.google_calendar.mapper import gcal_to_blurt_event
from blurt.integrations.google_calendar.sync_state import (
    ChangeType,
    EventChange,
    PollResult,
    SyncState,
)

logger = logging.getLogger(__name__)

# Google Calendar API error code for expired sync tokens
_SYNC_TOKEN_GONE = 410


def _content_hash(gcal_event: dict[str, Any]) -> str:
    """Generate a content hash from mutable Google Calendar event fields.

    Compares summary, description, location, start, end, attendees, status.
    Ignores etag, updated timestamp, and other metadata-only fields.
    This lets us distinguish real content changes from metadata-only updates.
    """
    content = {
        "summary": gcal_event.get("summary", ""),
        "description": gcal_event.get("description"),
        "location": gcal_event.get("location"),
        "status": gcal_event.get("status", "confirmed"),
        "start": gcal_event.get("start", {}),
        "end": gcal_event.get("end", {}),
        "attendees": [
            {"email": a.get("email"), "responseStatus": a.get("responseStatus")}
            for a in gcal_event.get("attendees", [])
        ],
        "recurrence": gcal_event.get("recurrence"),
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, default=str).encode()
    ).hexdigest()


class GoogleCalendarReader:
    """Read-path client for Google Calendar with change detection.

    Complements the existing GoogleCalendarClient (write path) to enable
    full bidirectional sync. Uses sync tokens for efficient incremental
    polling and content hashing to detect actual changes.

    Usage:
        auth = GoogleCalendarAuth()
        reader = GoogleCalendarReader(auth)

        # List upcoming events
        events = reader.list_events(user_id="u1")

        # Poll for changes
        state = SyncState(user_id="u1")
        result = reader.poll_changes(state)
    """

    def __init__(
        self,
        auth: GoogleCalendarAuth,
        calendar_id: str = "primary",
        service: Any | None = None,
        full_sync_lookback_days: int = 30,
        full_sync_lookahead_days: int = 90,
        max_results_per_page: int = 250,
    ):
        """Initialize the reader.

        Args:
            auth: Authentication manager for Google API credentials.
            calendar_id: Default calendar to operate on.
            service: Optional pre-built Google Calendar API service (for testing).
            full_sync_lookback_days: How far back to look on full sync.
            full_sync_lookahead_days: How far forward to look on full sync.
            max_results_per_page: Max events per API page request.
        """
        self._auth = auth
        self._calendar_id = calendar_id
        self._service = service
        self._lookback_days = full_sync_lookback_days
        self._lookahead_days = full_sync_lookahead_days
        self._max_per_page = max_results_per_page

    def _get_service(self) -> Any:
        """Get or build the Google Calendar API service."""
        if self._service is not None:
            return self._service

        from googleapiclient.discovery import build

        credentials = self._auth.get_credentials()
        self._service = build("calendar", "v3", credentials=credentials)
        return self._service

    # ── Single Event Read ───────────────────────────────────────────

    def get_event(
        self,
        event_id: str,
        calendar_id: str | None = None,
        user_id: str = "",
    ) -> BlurtEvent:
        """Read a single calendar event by its Google Calendar ID.

        Args:
            event_id: The Google Calendar event ID.
            calendar_id: Calendar to read from (defaults to instance calendar_id).
            user_id: Blurt user ID for the returned BlurtEvent.

        Returns:
            The event mapped to a BlurtEvent.

        Raises:
            EventNotFoundError: If the event does not exist.
            GoogleCalendarError: On other API errors.
        """
        from googleapiclient.errors import HttpError

        service = self._get_service()
        cal_id = calendar_id or self._calendar_id

        try:
            result = (
                service.events()
                .get(calendarId=cal_id, eventId=event_id)
                .execute()
            )
            event = gcal_to_blurt_event(result, user_id=user_id)
            event.last_modified = datetime.now(timezone.utc)
            return event

        except HttpError as e:
            status = e.resp.status if e.resp else None
            if status == 404:
                raise EventNotFoundError(
                    f"Event not found: {event_id}",
                    status_code=404,
                ) from e
            raise GoogleCalendarError(
                f"Failed to get event {event_id}",
                status_code=status,
                details=str(e),
            ) from e

    # ── Event Listing ───────────────────────────────────────────────

    def list_events(
        self,
        calendar_id: str | None = None,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        max_results: int | None = None,
        single_events: bool = True,
        order_by: str = "startTime",
        query: str | None = None,
        page_token: str | None = None,
        show_deleted: bool = False,
        user_id: str = "",
    ) -> dict[str, Any]:
        """List calendar events with filtering and pagination.

        Args:
            calendar_id: Calendar to list from (defaults to instance calendar_id).
            time_min: Lower bound for event start (inclusive).
            time_max: Upper bound for event start (exclusive).
            max_results: Max events per page.
            single_events: If True, expand recurring events into instances.
            order_by: "startTime" or "updated".
            query: Free-text search query.
            page_token: Pagination token from previous response.
            show_deleted: Include cancelled/deleted events.
            user_id: Blurt user ID for the returned BlurtEvents.

        Returns:
            Dict with keys:
                events: list[BlurtEvent]
                next_page_token: str | None
                next_sync_token: str | None
                raw_items: list[dict] — raw API responses for hash computation
        """
        from googleapiclient.errors import HttpError

        service = self._get_service()
        cal_id = calendar_id or self._calendar_id

        kwargs: dict[str, Any] = {"calendarId": cal_id}

        if time_min:
            kwargs["timeMin"] = time_min.isoformat()
        if time_max:
            kwargs["timeMax"] = time_max.isoformat()
        if max_results:
            kwargs["maxResults"] = max_results
        if single_events:
            kwargs["singleEvents"] = True
        if order_by and single_events:
            kwargs["orderBy"] = order_by
        if query:
            kwargs["q"] = query
        if page_token:
            kwargs["pageToken"] = page_token
        if show_deleted:
            kwargs["showDeleted"] = True

        try:
            response = service.events().list(**kwargs).execute()
        except HttpError as e:
            status = e.resp.status if e.resp else None
            raise GoogleCalendarError(
                f"Failed to list events from {cal_id}",
                status_code=status,
                details=str(e),
            ) from e

        raw_items = response.get("items", [])
        events = [gcal_to_blurt_event(item, user_id=user_id) for item in raw_items]

        return {
            "events": events,
            "next_page_token": response.get("nextPageToken"),
            "next_sync_token": response.get("nextSyncToken"),
            "raw_items": raw_items,
        }

    def list_all_events(
        self,
        calendar_id: str | None = None,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        query: str | None = None,
        show_deleted: bool = False,
        user_id: str = "",
    ) -> dict[str, Any]:
        """List all events, automatically handling pagination.

        Returns all events across all pages plus the final sync token.
        """
        all_events: list[BlurtEvent] = []
        all_raw_items: list[dict[str, Any]] = []
        page_token: str | None = None
        sync_token: str | None = None

        while True:
            result = self.list_events(
                calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                max_results=self._max_per_page,
                query=query,
                page_token=page_token,
                show_deleted=show_deleted,
                user_id=user_id,
            )
            all_events.extend(result["events"])
            all_raw_items.extend(result["raw_items"])
            sync_token = result["next_sync_token"]
            page_token = result["next_page_token"]
            if not page_token:
                break

        return {
            "events": all_events,
            "next_sync_token": sync_token,
            "raw_items": all_raw_items,
        }

    def _list_with_sync_token(
        self,
        sync_token: str,
        calendar_id: str | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List events using a sync token for incremental sync.

        Sync token mode is mutually exclusive with time range filters.
        Returns only events that changed since the sync token was issued.

        Raises:
            SyncTokenExpiredError: If the sync token is invalid (HTTP 410).
        """
        from googleapiclient.errors import HttpError

        service = self._get_service()
        cal_id = calendar_id or self._calendar_id

        kwargs: dict[str, Any] = {
            "calendarId": cal_id,
            "showDeleted": True,  # Must show deleted for change detection
        }

        if page_token:
            kwargs["pageToken"] = page_token
        else:
            kwargs["syncToken"] = sync_token

        try:
            response = service.events().list(**kwargs).execute()
        except HttpError as e:
            status = e.resp.status if e.resp else None
            if status == _SYNC_TOKEN_GONE:
                raise SyncTokenExpiredError(
                    "Sync token expired — full re-sync required",
                    status_code=status,
                ) from e
            raise GoogleCalendarError(
                f"Incremental sync failed for {cal_id}",
                status_code=status,
                details=str(e),
            ) from e

        raw_items = response.get("items", [])

        return {
            "raw_items": raw_items,
            "next_page_token": response.get("nextPageToken"),
            "next_sync_token": response.get("nextSyncToken"),
        }

    # ── Polling & Change Detection ──────────────────────────────────

    def poll_changes(self, sync_state: SyncState) -> PollResult:
        """Poll for changes since the last sync.

        This is the core change detection method for bidirectional sync.

        Strategy:
        1. If we have a sync token, use incremental sync (efficient).
        2. If sync token expired (410), fall back to full sync.
        3. If no sync token exists, do initial full sync.
        4. Compare content hashes to detect actual vs. metadata-only changes.

        Args:
            sync_state: Current sync state with token and event hashes.

        Returns:
            PollResult with detected changes and new sync token.
        """
        try:
            if sync_state.sync_token:
                return self._incremental_sync(sync_state)
            else:
                return self._full_sync(sync_state)
        except SyncTokenExpiredError:
            logger.info(
                "Sync token expired for user=%s calendar=%s — full re-sync",
                sync_state.user_id,
                sync_state.calendar_id,
            )
            sync_state.reset()
            return self._full_sync(sync_state)

    def _incremental_sync(self, sync_state: SyncState) -> PollResult:
        """Perform incremental sync using the stored sync token."""
        changes: list[EventChange] = []
        page_token: str | None = None
        new_sync_token: str | None = None
        events_checked = 0

        while True:
            result = self._list_with_sync_token(
                sync_token=sync_state.sync_token or "" if not page_token else "",
                calendar_id=sync_state.calendar_id,
                page_token=page_token,
            )

            for raw_item in result["raw_items"]:
                events_checked += 1
                change = self._detect_change(raw_item, sync_state)
                if change:
                    changes.append(change)

            new_sync_token = result["next_sync_token"]
            page_token = result["next_page_token"]
            if not page_token:
                break

        now = datetime.now(timezone.utc)
        if new_sync_token:
            sync_state.sync_token = new_sync_token
        sync_state.last_poll_at = now
        sync_state.consecutive_errors = 0

        return PollResult(
            changes=changes,
            new_sync_token=new_sync_token,
            polled_at=now,
            is_full_sync=False,
            events_checked=events_checked,
        )

    def _full_sync(self, sync_state: SyncState) -> PollResult:
        """Perform a full sync — fetches all events in the configured range.

        Used for initial sync or when a sync token has expired.
        Rebuilds the event hash map from scratch.
        """
        now = datetime.now(timezone.utc)
        time_min = now - timedelta(days=self._lookback_days)
        time_max = now + timedelta(days=self._lookahead_days)

        old_hashes = dict(sync_state.event_hashes)
        seen_event_ids: set[str] = set()
        changes: list[EventChange] = []

        result = self.list_all_events(
            calendar_id=sync_state.calendar_id,
            time_min=time_min,
            time_max=time_max,
            show_deleted=True,
            user_id=sync_state.user_id,
        )

        for raw_item in result["raw_items"]:
            event_id = raw_item.get("id", "")
            seen_event_ids.add(event_id)

            status = raw_item.get("status", "confirmed")
            if status == "cancelled":
                if sync_state.remove_event(event_id):
                    blurt_event = gcal_to_blurt_event(raw_item, user_id=sync_state.user_id)
                    changes.append(EventChange(
                        change_type=ChangeType.DELETED,
                        event=blurt_event,
                        previous_hash=old_hashes.get(event_id),
                        google_calendar_id=event_id,
                    ))
            else:
                change = self._detect_change(raw_item, sync_state)
                if change:
                    changes.append(change)

        # Events in our hash map but not in full sync results are deleted
        for event_id in set(old_hashes.keys()) - seen_event_ids:
            sync_state.remove_event(event_id)
            changes.append(EventChange(
                change_type=ChangeType.DELETED,
                event=None,  # We don't have the full event data
                google_calendar_id=event_id,
                previous_hash=old_hashes.get(event_id),
            ))

        if result["next_sync_token"]:
            sync_state.sync_token = result["next_sync_token"]
        sync_state.last_poll_at = now
        sync_state.last_full_sync_at = now
        sync_state.consecutive_errors = 0

        return PollResult(
            changes=changes,
            new_sync_token=result["next_sync_token"],
            polled_at=now,
            is_full_sync=True,
            events_checked=len(result["raw_items"]),
        )

    def _detect_change(
        self,
        raw_item: dict[str, Any],
        sync_state: SyncState,
    ) -> EventChange | None:
        """Compare an event against stored hashes to detect actual changes.

        Returns an EventChange if the event is new, modified, or deleted.
        Returns None if the event content hasn't actually changed.
        """
        event_id = raw_item.get("id", "")
        status = raw_item.get("status", "confirmed")

        # Handle deletions
        if status == "cancelled":
            previous_hash = sync_state.event_hashes.get(event_id)
            if sync_state.remove_event(event_id):
                blurt_event = gcal_to_blurt_event(raw_item, user_id=sync_state.user_id)
                return EventChange(
                    change_type=ChangeType.DELETED,
                    event=blurt_event,
                    previous_hash=previous_hash,
                    google_calendar_id=event_id,
                )
            return None  # Not in our tracking

        # Compute content hash
        new_hash = _content_hash(raw_item)
        old_hash = sync_state.event_hashes.get(event_id)

        if old_hash is None:
            # New event
            sync_state.event_hashes[event_id] = new_hash
            blurt_event = gcal_to_blurt_event(raw_item, user_id=sync_state.user_id)
            return EventChange(
                change_type=ChangeType.CREATED,
                event=blurt_event,
                current_hash=new_hash,
                google_calendar_id=event_id,
            )
        elif old_hash != new_hash:
            # Content actually changed
            sync_state.event_hashes[event_id] = new_hash
            blurt_event = gcal_to_blurt_event(raw_item, user_id=sync_state.user_id)
            return EventChange(
                change_type=ChangeType.UPDATED,
                event=blurt_event,
                previous_hash=old_hash,
                current_hash=new_hash,
                google_calendar_id=event_id,
            )

        # Content unchanged — metadata-only update, ignore
        return None


class SyncTokenExpiredError(GoogleCalendarError):
    """Raised when a sync token is no longer valid (HTTP 410)."""
    pass
