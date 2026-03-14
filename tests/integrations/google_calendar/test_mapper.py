"""Tests for Google Calendar mapper — Blurt event model ↔ GCal API format."""

from datetime import datetime

import pytest

from blurt.core.models import (
    BlurtEvent,
    EventAttendee,
    EventRecurrence,
    EventReminder,
    EventStatus,
)
from blurt.integrations.google_calendar.mapper import (
    blurt_event_to_gcal,
    gcal_to_blurt_event,
    _parse_recurrence_rule,
    _strip_blurt_metadata_footer,
)


class TestBlurtEventToGcal:
    """Test conversion from BlurtEvent to Google Calendar API format."""

    def test_basic_timed_event(self):
        """A simple event with title, start, and end times."""
        event = BlurtEvent(
            id="evt-001",
            title="Dentist appointment",
            start_time=datetime(2026, 3, 15, 14, 0),
            end_time=datetime(2026, 3, 15, 15, 0),
            timezone="America/New_York",
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["summary"] == "Dentist appointment"
        assert body["start"]["dateTime"] == "2026-03-15T14:00:00"
        assert body["start"]["timeZone"] == "America/New_York"
        assert body["end"]["dateTime"] == "2026-03-15T15:00:00"
        assert body["end"]["timeZone"] == "America/New_York"
        assert body["status"] == "confirmed"
        assert body["reminders"] == {"useDefault": True}

    def test_all_day_event(self):
        """An all-day event uses date instead of dateTime."""
        event = BlurtEvent(
            id="evt-002",
            title="Company holiday",
            start_time=datetime(2026, 7, 4),
            end_time=datetime(2026, 7, 5),
            all_day=True,
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["start"] == {"date": "2026-07-04"}
        assert body["end"] == {"date": "2026-07-05"}
        assert "dateTime" not in body["start"]
        assert "dateTime" not in body["end"]

    def test_event_with_duration_only(self):
        """When only duration_minutes is set, end_time is computed."""
        event = BlurtEvent(
            title="Quick sync",
            start_time=datetime(2026, 3, 15, 10, 0),
            duration_minutes=30,
            timezone="UTC",
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["end"]["dateTime"] == "2026-03-15T10:30:00"

    def test_default_duration_one_hour(self):
        """When neither end_time nor duration is set, defaults to 1 hour."""
        event = BlurtEvent(
            title="Meeting",
            start_time=datetime(2026, 3, 15, 9, 0),
            timezone="UTC",
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["end"]["dateTime"] == "2026-03-15T10:00:00"

    def test_event_with_description(self):
        """Description is included in the body."""
        event = BlurtEvent(
            title="Planning",
            description="Discuss Q2 roadmap with team",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert "Discuss Q2 roadmap" in body["description"]

    def test_event_with_location(self):
        """Location is passed through."""
        event = BlurtEvent(
            title="Lunch",
            location="Cafe Nero, Main St",
            start_time=datetime(2026, 3, 15, 12, 0),
            end_time=datetime(2026, 3, 15, 13, 0),
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["location"] == "Cafe Nero, Main St"

    def test_event_without_location(self):
        """No location field when not set."""
        event = BlurtEvent(
            title="Call",
            start_time=datetime(2026, 3, 15, 12, 0),
            end_time=datetime(2026, 3, 15, 12, 30),
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert "location" not in body

    def test_event_with_attendees(self):
        """Attendees are mapped correctly."""
        event = BlurtEvent(
            title="Team sync",
            start_time=datetime(2026, 3, 15, 14, 0),
            end_time=datetime(2026, 3, 15, 14, 30),
            attendees=[
                EventAttendee(
                    email="sarah@example.com",
                    name="Sarah",
                    response_status="accepted",
                ),
                EventAttendee(
                    email="jake@example.com",
                    name="Jake",
                    optional=True,
                ),
            ],
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert len(body["attendees"]) == 2
        assert body["attendees"][0]["email"] == "sarah@example.com"
        assert body["attendees"][0]["displayName"] == "Sarah"
        assert body["attendees"][0]["responseStatus"] == "accepted"
        assert body["attendees"][1]["optional"] is True

    def test_event_with_custom_reminders(self):
        """Custom reminders override defaults."""
        event = BlurtEvent(
            title="Interview",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            use_default_reminders=False,
            reminders=[
                EventReminder(method="popup", minutes_before=30),
                EventReminder(method="email", minutes_before=60),
            ],
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["reminders"]["useDefault"] is False
        assert len(body["reminders"]["overrides"]) == 2
        assert body["reminders"]["overrides"][0]["method"] == "popup"
        assert body["reminders"]["overrides"][0]["minutes"] == 30

    def test_weekly_recurrence(self):
        """Weekly recurrence generates RRULE."""
        event = BlurtEvent(
            title="Standup",
            start_time=datetime(2026, 3, 16, 9, 0),
            duration_minutes=15,
            recurrence=EventRecurrence.WEEKLY,
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert "recurrence" in body
        assert body["recurrence"] == ["RRULE:FREQ=WEEKLY"]

    def test_biweekly_recurrence(self):
        """Biweekly produces WEEKLY with INTERVAL=2."""
        event = BlurtEvent(
            title="1:1",
            start_time=datetime(2026, 3, 16, 10, 0),
            duration_minutes=30,
            recurrence=EventRecurrence.BIWEEKLY,
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["recurrence"] == ["RRULE:FREQ=WEEKLY;INTERVAL=2"]

    def test_custom_recurrence_rule(self):
        """Explicit recurrence_rule takes precedence."""
        event = BlurtEvent(
            title="Every MWF",
            start_time=datetime(2026, 3, 16, 8, 0),
            duration_minutes=60,
            recurrence=EventRecurrence.WEEKLY,
            recurrence_rule="RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR",
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["recurrence"] == ["RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"]

    def test_no_recurrence(self):
        """No recurrence field when NONE."""
        event = BlurtEvent(
            title="One-off",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body.get("recurrence", []) == []

    def test_extended_properties_contain_blurt_id(self):
        """Extended properties store Blurt tracking data."""
        event = BlurtEvent(
            id="evt-123",
            title="From blurt",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            blurt_id="blurt-456",
            linked_entity_ids=["entity-a", "entity-b"],
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        ext = body["extendedProperties"]["private"]
        assert ext["blurt_event_id"] == "evt-123"
        assert ext["blurt_id"] == "blurt-456"
        assert ext["blurt_linked_entities"] == "entity-a,entity-b"
        assert ext["blurt_source"] == "blurt"

    def test_status_tentative(self):
        """Tentative status maps correctly."""
        event = BlurtEvent(
            title="Maybe",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            status=EventStatus.TENTATIVE,
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["status"] == "tentative"

    def test_color_id(self):
        """Color ID is passed through."""
        event = BlurtEvent(
            title="Colorful",
            start_time=datetime(2026, 3, 15, 10, 0),
            end_time=datetime(2026, 3, 15, 11, 0),
            color_id="5",
            user_id="user-1",
        )
        body = blurt_event_to_gcal(event)

        assert body["colorId"] == "5"


class TestGcalToBlurtEvent:
    """Test conversion from Google Calendar API response to BlurtEvent."""

    def test_basic_timed_event(self):
        """Parse a simple timed event from GCal response."""
        gcal_event = {
            "id": "gcal-abc",
            "etag": '"etag-123"',
            "summary": "Team meeting",
            "start": {
                "dateTime": "2026-03-15T14:00:00-04:00",
                "timeZone": "America/New_York",
            },
            "end": {
                "dateTime": "2026-03-15T15:00:00-04:00",
                "timeZone": "America/New_York",
            },
            "status": "confirmed",
        }

        event = gcal_to_blurt_event(gcal_event, user_id="user-1")

        assert event.title == "Team meeting"
        assert event.google_calendar_id == "gcal-abc"
        assert event.google_calendar_etag == '"etag-123"'
        assert event.timezone == "America/New_York"
        assert event.all_day is False
        assert event.status == EventStatus.CONFIRMED
        assert event.user_id == "user-1"
        assert event.source == "google_calendar"
        assert event.synced_at is not None

    def test_all_day_event(self):
        """Parse an all-day event."""
        gcal_event = {
            "id": "gcal-allday",
            "summary": "Holiday",
            "start": {"date": "2026-07-04"},
            "end": {"date": "2026-07-05"},
            "status": "confirmed",
        }

        event = gcal_to_blurt_event(gcal_event)

        assert event.all_day is True
        assert event.start_time == datetime(2026, 7, 4)

    def test_preserves_blurt_metadata_from_extended_properties(self):
        """Blurt metadata round-trips through extended properties."""
        gcal_event = {
            "id": "gcal-xyz",
            "summary": "From Blurt",
            "start": {"dateTime": "2026-03-15T10:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-03-15T11:00:00Z", "timeZone": "UTC"},
            "status": "confirmed",
            "extendedProperties": {
                "private": {
                    "blurt_event_id": "evt-original",
                    "blurt_id": "blurt-789",
                    "blurt_source": "blurt",
                    "blurt_linked_entities": "ent-1,ent-2",
                },
            },
        }

        event = gcal_to_blurt_event(gcal_event, user_id="user-1")

        assert event.id == "evt-original"
        assert event.blurt_id == "blurt-789"
        assert event.source == "blurt"
        assert event.linked_entity_ids == ["ent-1", "ent-2"]

    def test_attendees_parsed(self):
        """Attendees are parsed from GCal response."""
        gcal_event = {
            "id": "gcal-attend",
            "summary": "Sync",
            "start": {"dateTime": "2026-03-15T10:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-03-15T10:30:00Z", "timeZone": "UTC"},
            "attendees": [
                {
                    "email": "alice@example.com",
                    "displayName": "Alice",
                    "responseStatus": "accepted",
                },
                {
                    "email": "bob@example.com",
                    "responseStatus": "tentative",
                    "optional": True,
                },
            ],
        }

        event = gcal_to_blurt_event(gcal_event)

        assert len(event.attendees) == 2
        assert event.attendees[0].email == "alice@example.com"
        assert event.attendees[0].name == "Alice"
        assert event.attendees[0].response_status == "accepted"
        assert event.attendees[1].optional is True

    def test_custom_reminders_parsed(self):
        """Custom reminder overrides are parsed."""
        gcal_event = {
            "id": "gcal-remind",
            "summary": "Important",
            "start": {"dateTime": "2026-03-15T10:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-03-15T11:00:00Z", "timeZone": "UTC"},
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 30},
                    {"method": "email", "minutes": 1440},
                ],
            },
        }

        event = gcal_to_blurt_event(gcal_event)

        assert event.use_default_reminders is False
        assert len(event.reminders) == 2
        assert event.reminders[0].minutes_before == 30
        assert event.reminders[1].method == "email"
        assert event.reminders[1].minutes_before == 1440

    def test_recurrence_parsed(self):
        """Recurrence rules are parsed from GCal response."""
        gcal_event = {
            "id": "gcal-recur",
            "summary": "Weekly standup",
            "start": {"dateTime": "2026-03-16T09:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-03-16T09:15:00Z", "timeZone": "UTC"},
            "recurrence": ["RRULE:FREQ=WEEKLY"],
        }

        event = gcal_to_blurt_event(gcal_event)

        assert event.recurrence == EventRecurrence.WEEKLY
        assert event.recurrence_rule == "RRULE:FREQ=WEEKLY"

    def test_duration_computed(self):
        """Duration in minutes is computed from start/end."""
        gcal_event = {
            "id": "gcal-dur",
            "summary": "Quick call",
            "start": {"dateTime": "2026-03-15T10:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-03-15T10:45:00Z", "timeZone": "UTC"},
        }

        event = gcal_to_blurt_event(gcal_event)

        assert event.duration_minutes == 45

    def test_location_parsed(self):
        """Location is preserved from GCal response."""
        gcal_event = {
            "id": "gcal-loc",
            "summary": "Lunch",
            "location": "Cafe Nero, Main St",
            "start": {"dateTime": "2026-03-15T12:00:00Z", "timeZone": "UTC"},
            "end": {"dateTime": "2026-03-15T13:00:00Z", "timeZone": "UTC"},
        }

        event = gcal_to_blurt_event(gcal_event)

        assert event.location == "Cafe Nero, Main St"


class TestRoundTrip:
    """Test that BlurtEvent → GCal → BlurtEvent preserves data."""

    def test_full_round_trip(self):
        """A rich event should survive a round trip through the mapper."""
        original = BlurtEvent(
            id="evt-round",
            title="Product review",
            description="Review Q2 roadmap and deliverables",
            start_time=datetime(2026, 3, 20, 14, 0),
            end_time=datetime(2026, 3, 20, 15, 30),
            timezone="America/Los_Angeles",
            location="Room 4B",
            status=EventStatus.CONFIRMED,
            recurrence=EventRecurrence.WEEKLY,
            attendees=[
                EventAttendee(email="sarah@co.com", name="Sarah", response_status="accepted"),
            ],
            use_default_reminders=False,
            reminders=[EventReminder(method="popup", minutes_before=10)],
            blurt_id="blurt-abc",
            user_id="user-1",
            linked_entity_ids=["ent-sarah", "ent-q2"],
            source="blurt",
            color_id="3",
        )

        # Convert to GCal format
        gcal_body = blurt_event_to_gcal(original)

        # Simulate GCal response (add id, etag)
        gcal_response = {**gcal_body, "id": "gcal-new-id", "etag": '"etag-new"'}

        # Convert back
        result = gcal_to_blurt_event(gcal_response, user_id="user-1")

        assert result.title == original.title
        assert result.location == original.location
        assert result.status == original.status
        assert result.recurrence == original.recurrence
        assert result.color_id == original.color_id
        assert result.blurt_id == original.blurt_id
        assert result.linked_entity_ids == original.linked_entity_ids
        assert result.source == "blurt"
        assert result.google_calendar_id == "gcal-new-id"
        assert len(result.attendees) == 1
        assert result.attendees[0].email == "sarah@co.com"
        assert len(result.reminders) == 1
        assert result.reminders[0].minutes_before == 10


class TestParseRecurrenceRule:
    """Test RRULE string parsing."""

    @pytest.mark.parametrize(
        "rrule,expected",
        [
            (None, EventRecurrence.NONE),
            ("", EventRecurrence.NONE),
            ("RRULE:FREQ=DAILY", EventRecurrence.DAILY),
            ("RRULE:FREQ=WEEKLY", EventRecurrence.WEEKLY),
            ("RRULE:FREQ=WEEKLY;INTERVAL=2", EventRecurrence.BIWEEKLY),
            ("RRULE:FREQ=MONTHLY", EventRecurrence.MONTHLY),
            ("RRULE:FREQ=YEARLY", EventRecurrence.YEARLY),
            ("RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR", EventRecurrence.WEEKLY),
            ("RRULE:FREQ=UNKNOWN", EventRecurrence.NONE),
        ],
    )
    def test_parse_recurrence(self, rrule, expected):
        assert _parse_recurrence_rule(rrule) == expected


class TestStripBlurtMetadata:
    """Test metadata footer stripping."""

    def test_no_metadata(self):
        assert _strip_blurt_metadata_footer("Just a description") == "Just a description"

    def test_with_blurt_tag(self):
        text = "Description\n\n[blurt:abc-123]"
        assert _strip_blurt_metadata_footer(text) == "Description"

    def test_empty(self):
        assert _strip_blurt_metadata_footer("") == ""

    def test_none_like(self):
        assert _strip_blurt_metadata_footer("") == ""
