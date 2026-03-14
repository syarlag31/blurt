"""Google Calendar API client for Blurt — write path (create/update).

Wraps the Google Calendar API v3 to provide create and update operations
for calendar events, mapping between Blurt's internal event model and
the Google Calendar API format.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from googleapiclient.discovery import build, Resource  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError

from blurt.core.models import BlurtEvent
from blurt.integrations.google_calendar.auth import GoogleCalendarAuth
from blurt.integrations.google_calendar.mapper import (
    blurt_event_to_gcal,
    gcal_to_blurt_event,
)

logger = logging.getLogger(__name__)


class GoogleCalendarError(Exception):
    """Base exception for Google Calendar operations."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class EventNotFoundError(GoogleCalendarError):
    """Raised when a calendar event is not found."""
    pass


class ConflictError(GoogleCalendarError):
    """Raised when an update conflicts (etag mismatch)."""
    pass


class GoogleCalendarClient:
    """Client for Google Calendar event create/update operations.

    Handles authentication, API calls, error handling, and mapping between
    Blurt's internal event model and the Google Calendar API.

    Usage:
        auth = GoogleCalendarAuth()
        client = GoogleCalendarClient(auth)

        # Create an event
        event = BlurtEvent(title="Dentist", start_time=datetime(...))
        created = await client.create_event(event)

        # Update an event
        created.title = "Dentist - Dr. Smith"
        updated = await client.update_event(created)
    """

    def __init__(
        self,
        auth: GoogleCalendarAuth,
        calendar_id: str = "primary",
        service: Optional[Resource] = None,
    ):
        """Initialize the Google Calendar client.

        Args:
            auth: Authentication manager for Google API credentials.
            calendar_id: The calendar to operate on. Defaults to "primary".
            service: Optional pre-built Google Calendar API service resource
                     (useful for testing / dependency injection).
        """
        self._auth = auth
        self._calendar_id = calendar_id
        self._service = service

    def _get_service(self) -> Any:
        """Get or build the Google Calendar API service."""
        if self._service is not None:
            return self._service

        credentials = self._auth.get_credentials()
        self._service = build("calendar", "v3", credentials=credentials)
        return self._service

    def create_event(
        self,
        event: BlurtEvent,
        send_updates: str = "none",
        conference_data_version: int = 0,
    ) -> BlurtEvent:
        """Create a new event in Google Calendar.

        Args:
            event: The Blurt event to create.
            send_updates: Whether to send notifications. One of:
                "all" — send to all attendees
                "externalOnly" — only non-Google Calendar attendees
                "none" — no notifications (default, least intrusive)
            conference_data_version: Set to 1 to enable conference data
                (Google Meet links).

        Returns:
            The created BlurtEvent with google_calendar_id populated.

        Raises:
            GoogleCalendarError: If the API call fails.
        """
        service = self._get_service()
        body = blurt_event_to_gcal(event)

        try:
            result = (
                service.events()
                .insert(
                    calendarId=self._calendar_id,
                    body=body,
                    sendUpdates=send_updates,
                    conferenceDataVersion=conference_data_version,
                )
                .execute()
            )

            logger.info(
                "Created Google Calendar event: %s (gcal_id=%s)",
                event.title,
                result.get("id"),
            )

            # Map the response back to a BlurtEvent, preserving Blurt-specific fields
            created_event = gcal_to_blurt_event(result, user_id=event.user_id)

            # Preserve the original Blurt event ID and blurt_id
            created_event.id = event.id
            created_event.blurt_id = event.blurt_id
            created_event.linked_entity_ids = event.linked_entity_ids
            created_event.source = event.source

            return created_event

        except HttpError as e:
            status = e.resp.status if e.resp else None
            logger.error(
                "Failed to create Google Calendar event: %s (status=%s)",
                event.title,
                status,
            )
            raise GoogleCalendarError(
                f"Failed to create event '{event.title}'",
                status_code=status,
                details=str(e),
            ) from e

    def update_event(
        self,
        event: BlurtEvent,
        send_updates: str = "none",
    ) -> BlurtEvent:
        """Update an existing event in Google Calendar.

        The event must have a google_calendar_id set from a previous
        create or read operation.

        Args:
            event: The Blurt event with updated fields.
            send_updates: Whether to send notifications to attendees.

        Returns:
            The updated BlurtEvent with fresh etag and synced_at.

        Raises:
            EventNotFoundError: If the event doesn't exist in Google Calendar.
            ConflictError: If the event was modified externally (etag mismatch).
            GoogleCalendarError: If the API call fails for other reasons.
            ValueError: If the event has no google_calendar_id.
        """
        if not event.google_calendar_id:
            raise ValueError(
                "Cannot update event without google_calendar_id. "
                "Create the event first or set the ID from a previous sync."
            )

        service = self._get_service()
        body = blurt_event_to_gcal(event)

        try:
            request_kwargs: dict[str, Any] = {
                "calendarId": self._calendar_id,
                "eventId": event.google_calendar_id,
                "body": body,
                "sendUpdates": send_updates,
            }

            result = service.events().update(**request_kwargs).execute()

            logger.info(
                "Updated Google Calendar event: %s (gcal_id=%s)",
                event.title,
                event.google_calendar_id,
            )

            # Map back preserving Blurt-specific fields
            updated_event = gcal_to_blurt_event(result, user_id=event.user_id)
            updated_event.id = event.id
            updated_event.blurt_id = event.blurt_id
            updated_event.linked_entity_ids = event.linked_entity_ids
            updated_event.source = event.source

            return updated_event

        except HttpError as e:
            status = e.resp.status if e.resp else None

            if status == 404:
                raise EventNotFoundError(
                    f"Event '{event.title}' not found in Google Calendar "
                    f"(gcal_id={event.google_calendar_id})",
                    status_code=404,
                ) from e

            if status == 409:
                raise ConflictError(
                    f"Event '{event.title}' was modified externally. "
                    "Re-read the event and retry.",
                    status_code=409,
                ) from e

            logger.error(
                "Failed to update Google Calendar event: %s (status=%s)",
                event.title,
                status,
            )
            raise GoogleCalendarError(
                f"Failed to update event '{event.title}'",
                status_code=status,
                details=str(e),
            ) from e

    def patch_event(
        self,
        event_gcal_id: str,
        updates: dict[str, Any],
        send_updates: str = "none",
    ) -> dict[str, Any]:
        """Partially update an event using Google Calendar's patch API.

        This is useful for quick updates (e.g., changing just the title
        or status) without sending the full event body.

        Args:
            event_gcal_id: The Google Calendar event ID.
            updates: A dict of fields to update (in Google Calendar format).
            send_updates: Whether to send notifications.

        Returns:
            The raw Google Calendar API response dict.

        Raises:
            EventNotFoundError: If the event doesn't exist.
            GoogleCalendarError: If the API call fails.
        """
        service = self._get_service()

        try:
            result = (
                service.events()
                .patch(
                    calendarId=self._calendar_id,
                    eventId=event_gcal_id,
                    body=updates,
                    sendUpdates=send_updates,
                )
                .execute()
            )

            logger.info("Patched Google Calendar event: %s", event_gcal_id)
            return result

        except HttpError as e:
            status = e.resp.status if e.resp else None

            if status == 404:
                raise EventNotFoundError(
                    f"Event not found (gcal_id={event_gcal_id})",
                    status_code=404,
                ) from e

            raise GoogleCalendarError(
                f"Failed to patch event (gcal_id={event_gcal_id})",
                status_code=status,
                details=str(e),
            ) from e

    def cancel_event(
        self,
        event: BlurtEvent,
        send_updates: str = "none",
    ) -> BlurtEvent:
        """Cancel (soft-delete) an event in Google Calendar.

        Sets the event status to 'cancelled' rather than hard-deleting it.
        This preserves the event in the calendar's history.

        Args:
            event: The Blurt event to cancel.
            send_updates: Whether to send notifications.

        Returns:
            The updated BlurtEvent with cancelled status.
        """
        if not event.google_calendar_id:
            raise ValueError("Cannot cancel event without google_calendar_id.")

        self.patch_event(
            event.google_calendar_id,
            {"status": "cancelled"},
            send_updates=send_updates,
        )

        event.status = __import__("blurt.core.models", fromlist=["EventStatus"]).EventStatus.CANCELLED
        event.synced_at = datetime.now(timezone.utc)
        event.last_modified = datetime.now(timezone.utc)
        return event

    def create_or_update_event(
        self,
        event: BlurtEvent,
        send_updates: str = "none",
    ) -> BlurtEvent:
        """Create or update an event based on whether it has a Google Calendar ID.

        This is the primary entry point for the sync pipeline — it handles
        the routing automatically.

        Args:
            event: The Blurt event to sync.
            send_updates: Whether to send notifications.

        Returns:
            The synced BlurtEvent.
        """
        if event.google_calendar_id:
            return self.update_event(event, send_updates=send_updates)
        return self.create_event(event, send_updates=send_updates)
