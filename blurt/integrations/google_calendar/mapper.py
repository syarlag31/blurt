"""Mapper between Blurt internal event model and Google Calendar API format.

Handles bidirectional conversion:
- BlurtEvent → Google Calendar API event body (for create/update)
- Google Calendar API response → BlurtEvent (for read)

Google Calendar API reference:
https://developers.google.com/calendar/api/v3/reference/events
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from blurt.core.models import (
    BlurtEvent,
    EventAttendee,
    EventRecurrence,
    EventReminder,
    EventStatus,
)


# Mapping from BlurtEvent status to Google Calendar status strings
_STATUS_TO_GCAL: dict[EventStatus, str] = {
    EventStatus.CONFIRMED: "confirmed",
    EventStatus.TENTATIVE: "tentative",
    EventStatus.CANCELLED: "cancelled",
}

_STATUS_FROM_GCAL: dict[str, EventStatus] = {v: k for k, v in _STATUS_TO_GCAL.items()}

# Mapping from EventRecurrence to RRULE frequency strings
_RECURRENCE_TO_RRULE: dict[EventRecurrence, Optional[str]] = {
    EventRecurrence.NONE: None,
    EventRecurrence.DAILY: "RRULE:FREQ=DAILY",
    EventRecurrence.WEEKLY: "RRULE:FREQ=WEEKLY",
    EventRecurrence.BIWEEKLY: "RRULE:FREQ=WEEKLY;INTERVAL=2",
    EventRecurrence.MONTHLY: "RRULE:FREQ=MONTHLY",
    EventRecurrence.YEARLY: "RRULE:FREQ=YEARLY",
}


def _format_datetime_for_gcal(dt: datetime, timezone_str: str) -> dict[str, str]:
    """Format a datetime for the Google Calendar API.

    Returns the appropriate dict with either 'dateTime' + 'timeZone'
    for timed events, or 'date' for all-day events.
    """
    return {
        "dateTime": dt.isoformat(),
        "timeZone": timezone_str,
    }


def _format_date_for_gcal(dt: datetime) -> dict[str, str]:
    """Format a date for all-day Google Calendar events."""
    return {
        "date": dt.strftime("%Y-%m-%d"),
    }


def blurt_event_to_gcal(event: BlurtEvent) -> dict[str, Any]:
    """Convert a BlurtEvent to a Google Calendar API event body.

    This produces the JSON body for the Google Calendar Events.insert()
    or Events.update() API call.

    Args:
        event: The Blurt internal event model.

    Returns:
        A dict suitable for passing to the Google Calendar API.
    """
    body: dict[str, Any] = {}

    # Summary (title)
    body["summary"] = event.title

    # Description — append Blurt metadata as invisible footer
    description_parts = []
    if event.description:
        description_parts.append(event.description)

    # Add Blurt tracking metadata to description
    blurt_meta = _build_blurt_metadata_footer(event)
    if blurt_meta:
        description_parts.append(blurt_meta)

    if description_parts:
        body["description"] = "\n\n".join(description_parts)

    # Start and end times
    end_time = event.compute_end_time()

    if event.all_day:
        body["start"] = _format_date_for_gcal(event.start_time)
        body["end"] = _format_date_for_gcal(end_time)
    else:
        body["start"] = _format_datetime_for_gcal(event.start_time, event.timezone)
        body["end"] = _format_datetime_for_gcal(end_time, event.timezone)

    # Location
    if event.location:
        body["location"] = event.location

    # Status
    body["status"] = _STATUS_TO_GCAL.get(event.status, "confirmed")

    # Recurrence
    recurrence_rules = _build_recurrence(event)
    if recurrence_rules:
        body["recurrence"] = recurrence_rules

    # Attendees
    if event.attendees:
        body["attendees"] = [_attendee_to_gcal(a) for a in event.attendees]

    # Reminders
    body["reminders"] = _build_reminders(event)

    # Color
    if event.color_id:
        body["colorId"] = event.color_id

    # Extended properties for Blurt-specific data
    body["extendedProperties"] = {
        "private": _build_extended_properties(event),
    }

    return body


def gcal_to_blurt_event(
    gcal_event: dict[str, Any],
    user_id: str = "",
) -> BlurtEvent:
    """Convert a Google Calendar API event response to a BlurtEvent.

    Args:
        gcal_event: The raw event dict from the Google Calendar API.
        user_id: The Blurt user ID who owns this event.

    Returns:
        A BlurtEvent populated from the Google Calendar data.
    """
    # Parse start time
    start_data = gcal_event.get("start", {})
    all_day = "date" in start_data and "dateTime" not in start_data

    if all_day:
        start_time = datetime.strptime(start_data["date"], "%Y-%m-%d")
    else:
        start_time = datetime.fromisoformat(start_data.get("dateTime", ""))

    # Parse end time
    end_data = gcal_event.get("end", {})
    if all_day:
        end_time = datetime.strptime(end_data.get("date", start_data.get("date", "")), "%Y-%m-%d")
    else:
        end_time = datetime.fromisoformat(end_data.get("dateTime", ""))

    # Parse timezone
    tz = start_data.get("timeZone", end_data.get("timeZone", "UTC"))

    # Parse status
    status = _STATUS_FROM_GCAL.get(
        gcal_event.get("status", "confirmed"),
        EventStatus.CONFIRMED,
    )

    # Parse attendees
    attendees = [
        _attendee_from_gcal(a) for a in gcal_event.get("attendees", [])
    ]

    # Parse reminders
    reminders_data = gcal_event.get("reminders", {})
    use_default = reminders_data.get("useDefault", True)
    reminders = []
    for override in reminders_data.get("overrides", []):
        reminders.append(
            EventReminder(
                method=override.get("method", "popup"),
                minutes_before=override.get("minutes", 15),
            )
        )

    # Parse recurrence
    recurrence = EventRecurrence.NONE
    recurrence_rule = None
    recurrence_list = gcal_event.get("recurrence", [])
    if recurrence_list:
        recurrence_rule = recurrence_list[0] if recurrence_list else None
        recurrence = _parse_recurrence_rule(recurrence_rule)

    # Extract Blurt metadata from extended properties
    ext_props = (
        gcal_event.get("extendedProperties", {}).get("private", {})
    )
    blurt_id = ext_props.get("blurt_id")
    source = ext_props.get("blurt_source", "google_calendar")
    linked_entities_raw = ext_props.get("blurt_linked_entities", "")
    linked_entity_ids = [e for e in linked_entities_raw.split(",") if e]

    # Parse description — strip Blurt metadata footer if present
    raw_description = gcal_event.get("description", "")
    description = _strip_blurt_metadata_footer(raw_description)

    # Compute duration in minutes
    duration_minutes = None
    if start_time and end_time:
        delta = end_time - start_time
        duration_minutes = int(delta.total_seconds() / 60)

    return BlurtEvent(
        id=ext_props.get("blurt_event_id", gcal_event.get("id", "")),
        title=gcal_event.get("summary", ""),
        description=description if description else None,
        start_time=start_time,
        end_time=end_time,
        duration_minutes=duration_minutes,
        all_day=all_day,
        location=gcal_event.get("location"),
        timezone=tz,
        status=status,
        recurrence=recurrence,
        recurrence_rule=recurrence_rule,
        attendees=attendees,
        reminders=reminders,
        use_default_reminders=use_default,
        blurt_id=blurt_id,
        user_id=user_id,
        linked_entity_ids=linked_entity_ids,
        source=source,
        color_id=gcal_event.get("colorId"),
        google_calendar_id=gcal_event.get("id"),
        google_calendar_etag=gcal_event.get("etag"),
        synced_at=datetime.now(timezone.utc),
    )


def _attendee_to_gcal(attendee: EventAttendee) -> dict[str, Any]:
    """Convert a BlurtEvent attendee to Google Calendar format."""
    result: dict[str, Any] = {}
    if attendee.email:
        result["email"] = attendee.email
    if attendee.name:
        result["displayName"] = attendee.name
    result["responseStatus"] = attendee.response_status
    if attendee.optional:
        result["optional"] = True
    return result


def _attendee_from_gcal(gcal_attendee: dict[str, Any]) -> EventAttendee:
    """Convert a Google Calendar attendee to BlurtEvent format."""
    return EventAttendee(
        email=gcal_attendee.get("email"),
        name=gcal_attendee.get("displayName"),
        response_status=gcal_attendee.get("responseStatus", "needsAction"),
        optional=gcal_attendee.get("optional", False),
    )


def _build_recurrence(event: BlurtEvent) -> list[str]:
    """Build the recurrence list for Google Calendar API."""
    rules = []

    # Use explicit recurrence_rule if provided
    if event.recurrence_rule:
        rules.append(event.recurrence_rule)
    elif event.recurrence != EventRecurrence.NONE:
        rrule = _RECURRENCE_TO_RRULE.get(event.recurrence)
        if rrule:
            rules.append(rrule)

    return rules


def _build_reminders(event: BlurtEvent) -> dict[str, Any]:
    """Build the reminders object for Google Calendar API."""
    if event.use_default_reminders and not event.reminders:
        return {"useDefault": True}

    overrides = []
    for reminder in event.reminders:
        overrides.append({
            "method": reminder.method,
            "minutes": reminder.minutes_before,
        })

    return {
        "useDefault": False,
        "overrides": overrides,
    }


def _build_extended_properties(event: BlurtEvent) -> dict[str, str]:
    """Build extended properties for Blurt tracking data.

    These are stored as private extended properties on the Google Calendar
    event and are only visible to the Blurt application.
    """
    props: dict[str, str] = {
        "blurt_event_id": event.id,
        "blurt_source": event.source,
    }

    if event.blurt_id:
        props["blurt_id"] = event.blurt_id

    if event.linked_entity_ids:
        props["blurt_linked_entities"] = ",".join(event.linked_entity_ids)

    return props


def _build_blurt_metadata_footer(event: BlurtEvent) -> str:
    """Build a metadata footer to append to event descriptions.

    This is a fallback for preserving Blurt context in event descriptions
    when extended properties aren't sufficient.
    """
    parts = []
    if event.blurt_id:
        parts.append(f"[blurt:{event.blurt_id}]")
    return ""  # Currently using extended properties instead


def _strip_blurt_metadata_footer(description: str) -> str:
    """Strip any Blurt metadata footer from a description."""
    if not description:
        return ""
    # Remove any [blurt:...] tags at the end
    lines = description.rstrip().split("\n")
    cleaned = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("[blurt:") and stripped.endswith("]"):
            continue
        cleaned.insert(0, line)
    return "\n".join(cleaned).strip()


def _parse_recurrence_rule(rrule: Optional[str]) -> EventRecurrence:
    """Parse an RRULE string into an EventRecurrence enum value."""
    if not rrule:
        return EventRecurrence.NONE

    rrule_upper = rrule.upper()

    if "FREQ=DAILY" in rrule_upper:
        return EventRecurrence.DAILY
    if "FREQ=YEARLY" in rrule_upper:
        return EventRecurrence.YEARLY
    if "FREQ=MONTHLY" in rrule_upper:
        return EventRecurrence.MONTHLY
    if "FREQ=WEEKLY" in rrule_upper:
        if "INTERVAL=2" in rrule_upper:
            return EventRecurrence.BIWEEKLY
        return EventRecurrence.WEEKLY

    return EventRecurrence.NONE
