"""Blurt core domain models.

These models define the internal representation of all Blurt data types.
They are the source of truth — external integrations map to/from these.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from blurt.models.intents import BlurtIntent

# Re-export BlurtIntent as IntentType for backward compatibility.
# New code should use BlurtIntent directly from blurt.models.intents.
IntentType = BlurtIntent


class EntityType(str, Enum):
    """Types of entities extracted from blurts.

    The four core types (PERSON, PLACE, PROJECT, ORGANIZATION) have dedicated
    typed attribute schemas in ``blurt.models.entities``.  TOPIC and TOOL are
    also valid entity types but use free-form attribute dicts.
    """

    PERSON = "person"
    PLACE = "place"
    PROJECT = "project"
    ORGANIZATION = "organization"
    TOPIC = "topic"
    TOOL = "tool"


class Entity(BaseModel):
    """An extracted entity from a blurt.

    This is the lightweight extraction-time representation.  Once an entity is
    persisted to the knowledge graph it becomes an ``EntityNode`` (see
    ``blurt.models.entities``) with typed attribute schemas, embeddings,
    mention tracking, and relationship edges.

    Fields:
        id: Auto-generated UUID.
        name: Entity name as mentioned in speech.
        entity_type: Classification of the entity.
        metadata: Free-form dict captured during extraction.  For the four core
            types this can later be parsed into a typed schema via
            ``parse_typed_attributes``.
        confidence: Extraction confidence score (0.0–1.0).
        aliases: Alternate names / references found in context.
        source_blurt_id: ID of the blurt that produced this entity.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entity_type: EntityType
    metadata: dict = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    source_blurt_id: Optional[str] = None

    def to_entity_node_kwargs(self, user_id: str) -> dict:
        """Convert to kwargs suitable for creating an ``EntityNode``.

        This bridges the extraction-time ``Entity`` to the persistence-layer
        ``EntityNode`` used in the knowledge graph, ensuring no data is lost
        in the transition.
        """
        from blurt.models.entities import EntityType as GraphEntityType

        return {
            "user_id": user_id,
            "name": self.name,
            "entity_type": GraphEntityType(self.entity_type.value),
            "aliases": list(self.aliases),
            "attributes": dict(self.metadata),
        }


class EventRecurrence(str, Enum):
    """Recurrence patterns for events."""

    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class EventStatus(str, Enum):
    """Status of an event."""

    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventAttendee(BaseModel):
    """An attendee of an event."""

    email: Optional[str] = None
    name: Optional[str] = None
    response_status: str = "needsAction"  # needsAction, accepted, declined, tentative
    optional: bool = False
    entity_id: Optional[str] = None  # Link to Blurt entity graph


class EventReminder(BaseModel):
    """A reminder for an event."""

    method: str = "popup"  # popup, email
    minutes_before: int = 15


class BlurtEvent(BaseModel):
    """Blurt's internal event model.

    This is the canonical representation of a calendar event within Blurt.
    It is richer than Google Calendar's model — it includes entity links,
    the originating blurt ID, and emotional context.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    all_day: bool = False
    location: Optional[str] = None
    timezone: str = "UTC"

    # Status and recurrence
    status: EventStatus = EventStatus.CONFIRMED
    recurrence: EventRecurrence = EventRecurrence.NONE
    recurrence_rule: Optional[str] = None  # Raw RRULE for complex patterns

    # Attendees
    attendees: list[EventAttendee] = Field(default_factory=list)

    # Reminders
    reminders: list[EventReminder] = Field(default_factory=list)
    use_default_reminders: bool = True

    # Blurt-specific metadata
    blurt_id: Optional[str] = None  # The blurt that created this event
    user_id: str = ""
    linked_entity_ids: list[str] = Field(default_factory=list)
    source: str = "blurt"  # blurt | google_calendar | manual
    color_id: Optional[str] = None  # Google Calendar color ID (1-11)

    # Sync tracking
    google_calendar_id: Optional[str] = None  # Google Calendar event ID
    google_calendar_etag: Optional[str] = None  # For conflict detection
    synced_at: Optional[datetime] = None
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def compute_end_time(self) -> datetime:
        """Compute end_time from duration if not explicitly set."""
        if self.end_time:
            return self.end_time
        if self.duration_minutes:
            return self.start_time + timedelta(minutes=self.duration_minutes)
        # Default to 1 hour
        return self.start_time + timedelta(hours=1)
