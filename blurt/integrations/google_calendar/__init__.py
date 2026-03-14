"""Google Calendar integration for Blurt.

Provides bidirectional sync between Blurt's internal event model
and Google Calendar via the Google Calendar API v3.

Write path: GoogleCalendarClient (create/update/patch/cancel)
Read path: GoogleCalendarReader (get/list/poll with change detection)
Polling: CalendarPoller (background change detection loop)
"""

from blurt.integrations.google_calendar.auth import GoogleCalendarAuth
from blurt.integrations.google_calendar.client import (
    ConflictError,
    EventNotFoundError,
    GoogleCalendarClient,
    GoogleCalendarError,
)
from blurt.integrations.google_calendar.mapper import (
    blurt_event_to_gcal,
    gcal_to_blurt_event,
)
from blurt.integrations.google_calendar.poller import CalendarPoller
from blurt.integrations.google_calendar.reader import (
    GoogleCalendarReader,
    SyncTokenExpiredError,
)
from blurt.integrations.google_calendar.sync_state import (
    ChangeType,
    EventChange,
    PollResult,
    SyncState,
)

__all__ = [
    # Auth
    "GoogleCalendarAuth",
    # Write path
    "GoogleCalendarClient",
    "GoogleCalendarError",
    "EventNotFoundError",
    "ConflictError",
    # Read path
    "GoogleCalendarReader",
    "SyncTokenExpiredError",
    # Polling & change detection
    "CalendarPoller",
    "ChangeType",
    "EventChange",
    "PollResult",
    "SyncState",
    # Mappers
    "blurt_event_to_gcal",
    "gcal_to_blurt_event",
]
