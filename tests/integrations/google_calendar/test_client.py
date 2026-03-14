"""Tests for Google Calendar client — create and update operations."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from blurt.core.models import (
    BlurtEvent,
    EventAttendee,
    EventRecurrence,
    EventStatus,
)
from blurt.integrations.google_calendar.client import (
    ConflictError,
    EventNotFoundError,
    GoogleCalendarClient,
    GoogleCalendarError,
)
from blurt.integrations.google_calendar.auth import GoogleCalendarAuth


def _make_mock_service(insert_return=None, update_return=None, patch_return=None):
    """Create a mock Google Calendar API service."""
    service = MagicMock()
    events = MagicMock()
    service.events.return_value = events

    if insert_return is not None:
        events.insert.return_value.execute.return_value = insert_return

    if update_return is not None:
        events.update.return_value.execute.return_value = update_return

    if patch_return is not None:
        events.patch.return_value.execute.return_value = patch_return

    return service


def _make_gcal_response(
    gcal_id: str = "gcal-created-1",
    summary: str = "Test Event",
    start_dt: str = "2026-03-15T14:00:00Z",
    end_dt: str = "2026-03-15T15:00:00Z",
    etag: str = '"etag-1"',
    ext_props: dict | None = None,
) -> dict:
    """Build a mock Google Calendar API event response."""
    response = {
        "id": gcal_id,
        "etag": etag,
        "summary": summary,
        "start": {"dateTime": start_dt, "timeZone": "UTC"},
        "end": {"dateTime": end_dt, "timeZone": "UTC"},
        "status": "confirmed",
    }
    if ext_props:
        response["extendedProperties"] = {"private": ext_props}
    return response


def _make_client(service=None) -> GoogleCalendarClient:
    """Create a GoogleCalendarClient with a mock auth and optional mock service."""
    auth = MagicMock(spec=GoogleCalendarAuth)
    return GoogleCalendarClient(auth=auth, service=service)


class TestCreateEvent:
    """Test event creation via Google Calendar API."""

    def test_create_basic_event(self):
        """Creates an event and returns a BlurtEvent with google_calendar_id set."""
        gcal_response = _make_gcal_response(
            gcal_id="gcal-new-123",
            summary="Dentist",
            ext_props={
                "blurt_event_id": "evt-1",
                "blurt_source": "blurt",
                "blurt_id": "blurt-original",
            },
        )
        service = _make_mock_service(insert_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            id="evt-1",
            title="Dentist",
            start_time=datetime(2026, 3, 15, 14, 0),
            end_time=datetime(2026, 3, 15, 15, 0),
            timezone="UTC",
            user_id="user-1",
            blurt_id="blurt-original",
        )

        result = client.create_event(event)

        assert result.google_calendar_id == "gcal-new-123"
        assert result.google_calendar_etag == '"etag-1"'
        assert result.id == "evt-1"  # Preserves Blurt event ID
        assert result.blurt_id == "blurt-original"
        assert result.synced_at is not None

        # Verify the API was called correctly
        service.events().insert.assert_called_once()
        call_kwargs = service.events().insert.call_args
        assert call_kwargs.kwargs["calendarId"] == "primary"
        assert call_kwargs.kwargs["sendUpdates"] == "none"

    def test_create_event_sends_correct_body(self):
        """The request body contains all mapped fields."""
        gcal_response = _make_gcal_response()
        service = _make_mock_service(insert_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            id="evt-2",
            title="Team sync",
            description="Weekly alignment",
            start_time=datetime(2026, 3, 16, 9, 0),
            duration_minutes=30,
            location="Room A",
            timezone="America/New_York",
            attendees=[EventAttendee(email="alice@co.com", name="Alice")],
            recurrence=EventRecurrence.WEEKLY,
            user_id="user-1",
            blurt_id="blurt-2",
            linked_entity_ids=["ent-alice"],
        )

        client.create_event(event)

        call_kwargs = service.events().insert.call_args
        body = call_kwargs.kwargs["body"]

        assert body["summary"] == "Team sync"
        assert "Weekly alignment" in body["description"]
        assert body["location"] == "Room A"
        assert body["start"]["timeZone"] == "America/New_York"
        assert body["recurrence"] == ["RRULE:FREQ=WEEKLY"]
        assert len(body["attendees"]) == 1
        assert body["attendees"][0]["email"] == "alice@co.com"
        assert body["extendedProperties"]["private"]["blurt_id"] == "blurt-2"
        assert body["extendedProperties"]["private"]["blurt_linked_entities"] == "ent-alice"

    def test_create_event_api_error(self):
        """Raises GoogleCalendarError on API failure."""
        service = MagicMock()
        from googleapiclient.errors import HttpError
        import httplib2

        resp = httplib2.Response({"status": 403})
        error = HttpError(resp, b"Forbidden")
        service.events().insert.return_value.execute.side_effect = error

        client = _make_client(service=service)
        event = BlurtEvent(
            title="Fail",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
        )

        with pytest.raises(GoogleCalendarError) as exc_info:
            client.create_event(event)

        assert exc_info.value.status_code == 403

    def test_create_event_preserves_linked_entities(self):
        """Linked entity IDs are preserved through create."""
        gcal_response = _make_gcal_response(
            ext_props={"blurt_linked_entities": "ent-a,ent-b"}
        )
        service = _make_mock_service(insert_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            title="With entities",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
            linked_entity_ids=["ent-a", "ent-b"],
        )

        result = client.create_event(event)
        assert result.linked_entity_ids == ["ent-a", "ent-b"]


class TestUpdateEvent:
    """Test event update via Google Calendar API."""

    def test_update_basic_event(self):
        """Updates an event and returns fresh etag."""
        gcal_response = _make_gcal_response(
            gcal_id="gcal-existing",
            summary="Updated title",
            etag='"etag-2"',
        )
        service = _make_mock_service(update_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            id="evt-3",
            title="Updated title",
            start_time=datetime(2026, 3, 15, 14, 0),
            end_time=datetime(2026, 3, 15, 15, 0),
            user_id="user-1",
            google_calendar_id="gcal-existing",
            google_calendar_etag='"etag-1"',
        )

        result = client.update_event(event)

        assert result.google_calendar_id == "gcal-existing"
        assert result.google_calendar_etag == '"etag-2"'
        assert result.id == "evt-3"

        service.events().update.assert_called_once()
        call_kwargs = service.events().update.call_args
        assert call_kwargs.kwargs["eventId"] == "gcal-existing"

    def test_update_without_gcal_id_raises(self):
        """Raises ValueError if event has no google_calendar_id."""
        client = _make_client()

        event = BlurtEvent(
            title="No GCal ID",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
        )

        with pytest.raises(ValueError, match="google_calendar_id"):
            client.update_event(event)

    def test_update_event_not_found(self):
        """Raises EventNotFoundError on 404."""
        service = MagicMock()
        from googleapiclient.errors import HttpError
        import httplib2

        resp = httplib2.Response({"status": 404})
        error = HttpError(resp, b"Not Found")
        service.events().update.return_value.execute.side_effect = error

        client = _make_client(service=service)
        event = BlurtEvent(
            title="Ghost",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
            google_calendar_id="gcal-gone",
        )

        with pytest.raises(EventNotFoundError):
            client.update_event(event)

    def test_update_conflict_error(self):
        """Raises ConflictError on 409."""
        service = MagicMock()
        from googleapiclient.errors import HttpError
        import httplib2

        resp = httplib2.Response({"status": 409})
        error = HttpError(resp, b"Conflict")
        service.events().update.return_value.execute.side_effect = error

        client = _make_client(service=service)
        event = BlurtEvent(
            title="Stale",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
            google_calendar_id="gcal-stale",
        )

        with pytest.raises(ConflictError):
            client.update_event(event)


class TestPatchEvent:
    """Test partial event updates."""

    def test_patch_event(self):
        """Patch updates specific fields."""
        gcal_response = _make_gcal_response(summary="New title")
        service = _make_mock_service(patch_return=gcal_response)
        client = _make_client(service=service)

        result = client.patch_event("gcal-123", {"summary": "New title"})

        assert result["summary"] == "New title"
        service.events().patch.assert_called_once()

    def test_patch_not_found(self):
        """Raises EventNotFoundError on 404."""
        service = MagicMock()
        from googleapiclient.errors import HttpError
        import httplib2

        resp = httplib2.Response({"status": 404})
        error = HttpError(resp, b"Not Found")
        service.events().patch.return_value.execute.side_effect = error

        client = _make_client(service=service)

        with pytest.raises(EventNotFoundError):
            client.patch_event("gcal-gone", {"summary": "nope"})


class TestCancelEvent:
    """Test event cancellation (soft delete)."""

    def test_cancel_event(self):
        """Cancel sets status to cancelled via patch."""
        gcal_response = _make_gcal_response()
        service = _make_mock_service(patch_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            title="To cancel",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
            google_calendar_id="gcal-cancel",
        )

        result = client.cancel_event(event)

        assert result.status == EventStatus.CANCELLED
        service.events().patch.assert_called_once()
        call_kwargs = service.events().patch.call_args
        assert call_kwargs.kwargs["body"] == {"status": "cancelled"}

    def test_cancel_without_gcal_id_raises(self):
        """Raises ValueError if no google_calendar_id."""
        client = _make_client()
        event = BlurtEvent(
            title="No ID",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
        )

        with pytest.raises(ValueError):
            client.cancel_event(event)


class TestCreateOrUpdateEvent:
    """Test the unified create-or-update method."""

    def test_routes_to_create_when_no_gcal_id(self):
        """Routes to create_event when google_calendar_id is None."""
        gcal_response = _make_gcal_response(gcal_id="gcal-new")
        service = _make_mock_service(insert_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            title="New event",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
        )

        result = client.create_or_update_event(event)

        assert result.google_calendar_id == "gcal-new"
        service.events().insert.assert_called_once()

    def test_routes_to_update_when_gcal_id_exists(self):
        """Routes to update_event when google_calendar_id is set."""
        gcal_response = _make_gcal_response(gcal_id="gcal-existing")
        service = _make_mock_service(update_return=gcal_response)
        client = _make_client(service=service)

        event = BlurtEvent(
            title="Existing event",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
            google_calendar_id="gcal-existing",
        )

        result = client.create_or_update_event(event)

        assert result.google_calendar_id == "gcal-existing"
        service.events().update.assert_called_once()
