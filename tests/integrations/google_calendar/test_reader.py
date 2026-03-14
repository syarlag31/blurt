"""Tests for Google Calendar reader — read and polling operations with change detection.

Covers:
- Single event retrieval
- Event listing with time range, search, pagination
- Multi-page listing
- Incremental sync via sync tokens
- Full sync with time range
- Change detection (created, updated, deleted events)
- Content hash change detection (ignores metadata-only changes)
- Sync token expiration and automatic recovery
- Poller lifecycle (start/stop, callbacks)
- Edge cases
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from blurt.core.models import BlurtEvent
from blurt.integrations.google_calendar.client import (
    EventNotFoundError,
    GoogleCalendarError,
)
from blurt.integrations.google_calendar.reader import (
    GoogleCalendarReader,
    _content_hash,
)
from blurt.integrations.google_calendar.sync_state import (
    ChangeType,
    EventChange,
    PollResult,
    SyncState,
)
from blurt.integrations.google_calendar.poller import CalendarPoller


# ── Test Helpers ────────────────────────────────────────────────────

def _make_gcal_event_dict(
    event_id: str = "gcal_1",
    summary: str = "Team Standup",
    description: str | None = "Daily standup",
    location: str | None = "Room 42",
    status: str = "confirmed",
    start_dt: str = "2026-03-14T10:00:00Z",
    end_dt: str = "2026-03-14T10:30:00Z",
    attendees: list | None = None,
    etag: str = '"abc123"',
    updated: str = "2026-03-13T12:00:00Z",
) -> dict:
    """Create a Google Calendar API event response dict."""
    event = {
        "id": event_id,
        "summary": summary,
        "status": status,
        "start": {"dateTime": start_dt, "timeZone": "UTC"},
        "end": {"dateTime": end_dt, "timeZone": "UTC"},
        "etag": etag,
        "updated": updated,
        "htmlLink": f"https://calendar.google.com/event?eid={event_id}",
        "created": "2026-03-13T10:00:00Z",
        "creator": {"email": "user@example.com"},
        "organizer": {"email": "user@example.com"},
    }
    if description:
        event["description"] = description
    if location:
        event["location"] = location
    if attendees:
        event["attendees"] = attendees
    return event


def _make_mock_service(
    list_responses: list[dict] | None = None,
    get_response: dict | None = None,
):
    """Create a mock Google Calendar API service.

    list_responses: list of dicts that service.events().list().execute() returns sequentially.
    get_response: dict that service.events().get().execute() returns.
    """
    service = MagicMock()
    events_resource = MagicMock()
    service.events.return_value = events_resource

    if list_responses is not None:
        list_mock = MagicMock()
        list_mock.execute = MagicMock(side_effect=list_responses)
        events_resource.list.return_value = list_mock

    if get_response is not None:
        get_mock = MagicMock()
        get_mock.execute.return_value = get_response
        events_resource.get.return_value = get_mock

    return service


def _make_reader(service=None, **kwargs) -> GoogleCalendarReader:
    """Create a GoogleCalendarReader with a mock auth and optional mock service."""
    auth = MagicMock()
    return GoogleCalendarReader(auth=auth, service=service, **kwargs)


# ── Content Hash Tests ──────────────────────────────────────────────

class TestContentHash:
    def test_deterministic(self):
        event = _make_gcal_event_dict()
        assert _content_hash(event) == _content_hash(event)

    def test_changes_with_summary(self):
        e1 = _make_gcal_event_dict(summary="Meeting A")
        e2 = _make_gcal_event_dict(summary="Meeting B")
        assert _content_hash(e1) != _content_hash(e2)

    def test_changes_with_time(self):
        e1 = _make_gcal_event_dict(start_dt="2026-03-14T10:00:00Z")
        e2 = _make_gcal_event_dict(start_dt="2026-03-14T11:00:00Z")
        assert _content_hash(e1) != _content_hash(e2)

    def test_changes_with_location(self):
        e1 = _make_gcal_event_dict(location="Room A")
        e2 = _make_gcal_event_dict(location="Room B")
        assert _content_hash(e1) != _content_hash(e2)

    def test_changes_with_description(self):
        e1 = _make_gcal_event_dict(description="v1")
        e2 = _make_gcal_event_dict(description="v2")
        assert _content_hash(e1) != _content_hash(e2)

    def test_changes_with_status(self):
        e1 = _make_gcal_event_dict(status="confirmed")
        e2 = _make_gcal_event_dict(status="tentative")
        assert _content_hash(e1) != _content_hash(e2)

    def test_ignores_etag(self):
        """Etag changes should NOT affect the content hash."""
        e1 = _make_gcal_event_dict(etag='"v1"')
        e2 = _make_gcal_event_dict(etag='"v2"')
        assert _content_hash(e1) == _content_hash(e2)

    def test_ignores_updated_timestamp(self):
        """The 'updated' timestamp should NOT affect the content hash."""
        e1 = _make_gcal_event_dict(updated="2026-03-13T12:00:00Z")
        e2 = _make_gcal_event_dict(updated="2026-03-14T08:00:00Z")
        assert _content_hash(e1) == _content_hash(e2)

    def test_attendee_response_changes(self):
        e1 = _make_gcal_event_dict(attendees=[
            {"email": "a@x.com", "responseStatus": "needsAction"}
        ])
        e2 = _make_gcal_event_dict(attendees=[
            {"email": "a@x.com", "responseStatus": "accepted"}
        ])
        assert _content_hash(e1) != _content_hash(e2)


# ── SyncState Tests ─────────────────────────────────────────────────

class TestSyncState:
    def test_remove_event_tracked(self):
        state = SyncState(user_id="u1")
        state.event_hashes["evt_1"] = "hash1"
        assert state.remove_event("evt_1") is True
        assert "evt_1" not in state.event_hashes

    def test_remove_event_not_tracked(self):
        state = SyncState(user_id="u1")
        assert state.remove_event("nonexistent") is False

    def test_reset(self):
        state = SyncState(user_id="u1", sync_token="tok", consecutive_errors=3)
        state.event_hashes = {"e1": "h1", "e2": "h2"}
        state.reset()
        assert state.sync_token is None
        assert state.event_hashes == {}
        assert state.consecutive_errors == 0

    def test_tracked_event_count(self):
        state = SyncState(user_id="u1")
        state.event_hashes = {"e1": "h1", "e2": "h2", "e3": "h3"}
        assert state.tracked_event_count == 3


# ── PollResult Tests ────────────────────────────────────────────────

class TestPollResult:
    def test_has_changes_true(self):
        result = PollResult(changes=[
            EventChange(change_type=ChangeType.CREATED, google_calendar_id="e1"),
        ])
        assert result.has_changes is True

    def test_has_changes_false(self):
        assert PollResult().has_changes is False

    def test_change_counts(self):
        result = PollResult(changes=[
            EventChange(change_type=ChangeType.CREATED, google_calendar_id="e1"),
            EventChange(change_type=ChangeType.CREATED, google_calendar_id="e2"),
            EventChange(change_type=ChangeType.UPDATED, google_calendar_id="e3"),
            EventChange(change_type=ChangeType.DELETED, google_calendar_id="e4"),
        ])
        assert result.created_count == 2
        assert result.updated_count == 1
        assert result.deleted_count == 1


# ── Reader: Get Event Tests ─────────────────────────────────────────

class TestGetEvent:
    def test_get_event_success(self):
        event_data = _make_gcal_event_dict(event_id="gcal_42", summary="Dentist")
        service = _make_mock_service(get_response=event_data)
        reader = _make_reader(service=service)

        event = reader.get_event("gcal_42", user_id="u1")

        assert isinstance(event, BlurtEvent)
        assert event.google_calendar_id == "gcal_42"
        assert event.title == "Dentist"
        assert event.user_id == "u1"

    def test_get_event_404(self):
        from googleapiclient.errors import HttpError
        import httplib2

        service = MagicMock()
        resp = httplib2.Response({"status": 404})
        service.events().get.return_value.execute.side_effect = HttpError(resp, b"Not Found")

        reader = _make_reader(service=service)

        with pytest.raises(EventNotFoundError):
            reader.get_event("nonexistent")

    def test_get_event_api_error(self):
        from googleapiclient.errors import HttpError
        import httplib2

        service = MagicMock()
        resp = httplib2.Response({"status": 500})
        service.events().get.return_value.execute.side_effect = HttpError(resp, b"Server Error")

        reader = _make_reader(service=service)

        with pytest.raises(GoogleCalendarError) as exc_info:
            reader.get_event("bad")
        assert exc_info.value.status_code == 500


# ── Reader: List Events Tests ───────────────────────────────────────

class TestListEvents:
    def test_list_events_basic(self):
        response = {
            "items": [
                _make_gcal_event_dict("e1", summary="Event 1"),
                _make_gcal_event_dict("e2", summary="Event 2"),
            ],
            "nextSyncToken": "sync_abc",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        result = reader.list_events(user_id="u1")

        assert len(result["events"]) == 2
        assert result["events"][0].title == "Event 1"
        assert result["next_sync_token"] == "sync_abc"
        assert result["next_page_token"] is None

    def test_list_events_with_time_range(self):
        response = {"items": [], "nextSyncToken": "tok"}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        time_min = datetime(2026, 3, 1, tzinfo=timezone.utc)
        time_max = datetime(2026, 3, 31, tzinfo=timezone.utc)
        reader.list_events(time_min=time_min, time_max=time_max)

        call_kwargs = service.events().list.call_args
        assert "timeMin" in call_kwargs.kwargs
        assert "timeMax" in call_kwargs.kwargs

    def test_list_events_with_search_query(self):
        response = {"items": []}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        reader.list_events(query="standup")

        call_kwargs = service.events().list.call_args
        assert call_kwargs.kwargs.get("q") == "standup"

    def test_list_events_with_pagination(self):
        response = {
            "items": [_make_gcal_event_dict()],
            "nextPageToken": "page2_tok",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        result = reader.list_events()
        assert result["next_page_token"] == "page2_tok"

    def test_list_events_show_deleted(self):
        response = {"items": []}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        reader.list_events(show_deleted=True)

        call_kwargs = service.events().list.call_args
        assert call_kwargs.kwargs.get("showDeleted") is True


class TestListAllEvents:
    def test_list_all_events_multi_page(self):
        """Handles multiple pages transparently."""
        service = MagicMock()
        events_resource = MagicMock()
        service.events.return_value = events_resource

        # Simulate two pages
        page1_mock = MagicMock()
        page1_mock.execute.return_value = {
            "items": [_make_gcal_event_dict("e1")],
            "nextPageToken": "page2",
        }
        page2_mock = MagicMock()
        page2_mock.execute.return_value = {
            "items": [_make_gcal_event_dict("e2")],
            "nextSyncToken": "sync_final",
        }
        events_resource.list.side_effect = [page1_mock, page2_mock]

        reader = _make_reader(service=service)
        result = reader.list_all_events(user_id="u1")

        assert len(result["events"]) == 2
        assert result["next_sync_token"] == "sync_final"
        assert events_resource.list.call_count == 2


# ── Reader: Change Detection Tests ──────────────────────────────────

class TestChangeDetection:
    def test_detect_new_event(self):
        reader = _make_reader()
        state = SyncState(user_id="u1")
        raw = _make_gcal_event_dict("e1")

        change = reader._detect_change(raw, state)

        assert change is not None
        assert change.change_type == ChangeType.CREATED
        assert change.google_calendar_id == "e1"
        assert "e1" in state.event_hashes

    def test_detect_unchanged_event(self):
        reader = _make_reader()
        state = SyncState(user_id="u1")
        raw = _make_gcal_event_dict("e1")

        # First time — created
        reader._detect_change(raw, state)
        # Second time — unchanged
        change = reader._detect_change(raw, state)

        assert change is None

    def test_detect_updated_event(self):
        reader = _make_reader()
        state = SyncState(user_id="u1")

        raw_v1 = _make_gcal_event_dict("e1", summary="V1")
        reader._detect_change(raw_v1, state)

        raw_v2 = _make_gcal_event_dict("e1", summary="V2")
        change = reader._detect_change(raw_v2, state)

        assert change is not None
        assert change.change_type == ChangeType.UPDATED
        assert change.event is not None
        assert change.event.title == "V2"

    def test_detect_deleted_event(self):
        reader = _make_reader()
        state = SyncState(user_id="u1")

        raw = _make_gcal_event_dict("e1")
        reader._detect_change(raw, state)

        raw_deleted = _make_gcal_event_dict("e1", status="cancelled")
        change = reader._detect_change(raw_deleted, state)

        assert change is not None
        assert change.change_type == ChangeType.DELETED
        assert "e1" not in state.event_hashes

    def test_detect_deleted_event_not_tracked(self):
        """Deletion of an event we're not tracking should return None."""
        reader = _make_reader()
        state = SyncState(user_id="u1")

        raw = _make_gcal_event_dict("unknown", status="cancelled")
        change = reader._detect_change(raw, state)

        assert change is None

    def test_metadata_only_change_ignored(self):
        """Event with different etag but same content should not be detected."""
        reader = _make_reader()
        state = SyncState(user_id="u1")

        raw_v1 = _make_gcal_event_dict("e1", etag='"etag_1"')
        reader._detect_change(raw_v1, state)

        raw_v2 = _make_gcal_event_dict("e1", etag='"etag_2"')
        change = reader._detect_change(raw_v2, state)

        assert change is None  # Content unchanged


# ── Reader: Polling Tests ───────────────────────────────────────────

class TestPollChanges:
    def test_initial_full_sync(self):
        """First poll with no sync token should do a full sync."""
        response = {
            "items": [
                _make_gcal_event_dict("e1"),
                _make_gcal_event_dict("e2", summary="Lunch"),
            ],
            "nextSyncToken": "sync_initial",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1")

        result = reader.poll_changes(state)

        assert result.is_full_sync is True
        assert result.created_count == 2
        assert state.sync_token == "sync_initial"
        assert len(state.event_hashes) == 2

    def test_incremental_poll_no_changes(self):
        """Incremental poll with no new events returns no changes."""
        response = {"items": [], "nextSyncToken": "sync_new"}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", sync_token="sync_old")

        result = reader.poll_changes(state)

        assert result.is_full_sync is False
        assert result.has_changes is False
        assert state.sync_token == "sync_new"

    def test_incremental_poll_detects_new_event(self):
        response = {
            "items": [_make_gcal_event_dict("e_new", summary="New Meeting")],
            "nextSyncToken": "sync_updated",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", sync_token="sync_old")

        result = reader.poll_changes(state)

        assert result.created_count == 1
        assert result.changes[0].change_type == ChangeType.CREATED
        assert result.changes[0].event is not None
        assert result.changes[0].event.title == "New Meeting"

    def test_incremental_poll_detects_updated_event(self):
        reader = _make_reader()
        state = SyncState(user_id="u1", sync_token="sync_old")
        # Pre-populate with existing event hash
        state.event_hashes["e1"] = _content_hash(
            _make_gcal_event_dict("e1", summary="V1")
        )

        response = {
            "items": [_make_gcal_event_dict("e1", summary="V2 Updated")],
            "nextSyncToken": "sync_updated",
        }
        service = _make_mock_service(list_responses=[response])
        reader._service = service

        result = reader.poll_changes(state)

        assert result.updated_count == 1
        assert result.changes[0].event is not None
        assert result.changes[0].event.title == "V2 Updated"

    def test_incremental_poll_detects_deleted_event(self):
        reader = _make_reader()
        state = SyncState(user_id="u1", sync_token="sync_old")
        state.event_hashes["e1"] = "somehash"

        response = {
            "items": [_make_gcal_event_dict("e1", status="cancelled")],
            "nextSyncToken": "sync_updated",
        }
        service = _make_mock_service(list_responses=[response])
        reader._service = service

        result = reader.poll_changes(state)

        assert result.deleted_count == 1
        assert "e1" not in state.event_hashes

    def test_incremental_poll_ignores_unchanged(self):
        """Unchanged events should not appear as changes."""
        raw = _make_gcal_event_dict("e1")
        state = SyncState(user_id="u1", sync_token="sync_old")
        state.event_hashes["e1"] = _content_hash(raw)

        response = {
            "items": [raw],  # Same content
            "nextSyncToken": "sync_new",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        result = reader.poll_changes(state)

        assert result.has_changes is False
        assert result.events_checked == 1

    def test_poll_updates_timestamps(self):
        response = {"items": [], "nextSyncToken": "sync_1"}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1")

        assert state.last_poll_at is None
        reader.poll_changes(state)
        assert state.last_poll_at is not None

    def test_poll_resets_error_count(self):
        response = {"items": [], "nextSyncToken": "sync_1"}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", consecutive_errors=3)

        reader.poll_changes(state)
        assert state.consecutive_errors == 0

    def test_full_sync_detects_implicit_deletions(self):
        """Full sync should detect events that disappeared from results."""
        state = SyncState(user_id="u1")
        state.event_hashes["e1"] = "hash1"
        state.event_hashes["e2"] = "hash2"

        # Only e1 returned — e2 implicitly deleted
        response = {
            "items": [_make_gcal_event_dict("e1")],
            "nextSyncToken": "sync_fresh",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        result = reader.poll_changes(state)

        assert result.is_full_sync is True
        deleted_ids = [c.google_calendar_id for c in result.changes if c.change_type == ChangeType.DELETED]
        assert "e2" in deleted_ids
        assert "e2" not in state.event_hashes

    def test_full_sync_handles_cancelled_events(self):
        """Full sync explicitly handles events with cancelled status."""
        state = SyncState(user_id="u1")
        state.event_hashes["e1"] = "hash1"

        response = {
            "items": [
                _make_gcal_event_dict("e1", status="cancelled"),
                _make_gcal_event_dict("e_new"),
            ],
            "nextSyncToken": "sync_new",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        result = reader.poll_changes(state)

        assert result.deleted_count == 1
        assert result.created_count == 1


class TestSyncTokenExpiration:
    def test_sync_token_expired_recovery(self):
        """When sync token expires (410), should fall back to full sync."""
        from googleapiclient.errors import HttpError
        import httplib2

        service = MagicMock()
        events_resource = MagicMock()
        service.events.return_value = events_resource

        # First call: 410 (expired token)
        expired_mock = MagicMock()
        resp_410 = httplib2.Response({"status": 410})
        expired_mock.execute.side_effect = HttpError(resp_410, b"Gone")

        # Second call: successful full sync
        success_mock = MagicMock()
        success_mock.execute.return_value = {
            "items": [_make_gcal_event_dict("e_fresh")],
            "nextSyncToken": "sync_fresh",
        }

        events_resource.list.side_effect = [expired_mock, success_mock]

        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", sync_token="expired_token")
        state.event_hashes = {"old_evt": "old_hash"}

        result = reader.poll_changes(state)

        assert result.is_full_sync is True
        assert result.created_count == 1
        assert state.sync_token == "sync_fresh"
        assert "old_evt" not in state.event_hashes


# ── Poller Tests ────────────────────────────────────────────────────

class TestCalendarPoller:
    def test_poll_once_with_changes(self):
        response = {
            "items": [_make_gcal_event_dict("e1")],
            "nextSyncToken": "sync_1",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1")

        callback = MagicMock()
        poller = CalendarPoller(reader, state, on_changes=callback)

        result = poller.poll_once()

        assert result.has_changes is True
        callback.assert_called_once()

    def test_poll_once_no_callback_when_no_changes(self):
        response = {"items": [], "nextSyncToken": "sync_1"}
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", sync_token="existing")

        callback = MagicMock()
        poller = CalendarPoller(reader, state, on_changes=callback)

        result = poller.poll_once()

        assert result.has_changes is False
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_poller_start_stop(self):
        response = {"items": [], "nextSyncToken": "sync_1"}
        service = _make_mock_service(list_responses=[response] * 100)  # Many responses
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", poll_interval_seconds=100)

        poller = CalendarPoller(reader, state)

        assert poller.is_running is False
        await poller.start()
        assert poller.is_running is True

        await asyncio.sleep(0.05)

        await poller.stop()
        assert poller.is_running is False

    @pytest.mark.asyncio
    async def test_poller_start_idempotent(self):
        response = {"items": [], "nextSyncToken": "sync_1"}
        service = _make_mock_service(list_responses=[response] * 100)
        reader = _make_reader(service=service)
        state = SyncState(user_id="u1", poll_interval_seconds=100)

        poller = CalendarPoller(reader, state)
        await poller.start()
        await poller.start()  # Should not raise
        assert poller.is_running is True
        await poller.stop()

    def test_poller_exposes_sync_state(self):
        reader = _make_reader()
        state = SyncState(user_id="u1")
        poller = CalendarPoller(reader, state)
        assert poller.sync_state is state


# ── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_summary_event(self):
        raw = _make_gcal_event_dict(summary="")
        h = _content_hash(raw)
        assert h is not None and len(h) > 0

    def test_event_without_optional_fields(self):
        raw = {
            "id": "minimal",
            "status": "confirmed",
            "start": {"date": "2026-03-20"},
            "end": {"date": "2026-03-21"},
        }
        h = _content_hash(raw)
        assert h is not None

    def test_multiple_changes_in_single_poll(self):
        """A single poll can return multiple types of changes."""
        state = SyncState(user_id="u1", sync_token="sync_old")
        state.event_hashes["e_existing"] = _content_hash(
            _make_gcal_event_dict("e_existing", summary="Old")
        )
        state.event_hashes["e_to_delete"] = "hash_del"

        response = {
            "items": [
                _make_gcal_event_dict("e_new", summary="Brand New"),
                _make_gcal_event_dict("e_existing", summary="Updated Title"),
                _make_gcal_event_dict("e_to_delete", status="cancelled"),
            ],
            "nextSyncToken": "sync_multi",
        }
        service = _make_mock_service(list_responses=[response])
        reader = _make_reader(service=service)

        result = reader.poll_changes(state)

        assert result.created_count == 1
        assert result.updated_count == 1
        assert result.deleted_count == 1
        assert result.events_checked == 3
