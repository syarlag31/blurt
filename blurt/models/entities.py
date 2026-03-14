"""Entity and knowledge graph data models for the semantic memory tier.

The semantic memory tier stores long-term knowledge: entities (people, places,
projects, organizations), their relationships, facts, preferences, and
learned patterns. Every entity gets a vector embedding for semantic search.

Specialized attribute schemas exist for each core entity type (person, place,
project, organization) so the knowledge graph captures domain-specific fields
(e.g. a person's role, a project's status) rather than only free-form dicts.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Entity Types ─────────────────────────────────────────────────────


class EntityType(str, Enum):
    """Types of entities that can exist in the knowledge graph."""

    PERSON = "person"
    PLACE = "place"
    PROJECT = "project"
    ORGANIZATION = "organization"
    TOPIC = "topic"
    TOOL = "tool"


# ── Relationship Types ───────────────────────────────────────────────


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    # People
    WORKS_WITH = "works_with"
    MANAGES = "manages"
    MANAGED_BY = "managed_by"
    REPORTS_TO = "reports_to"
    KNOWS = "knows"
    FRIEND_OF = "friend_of"
    FAMILY_OF = "family_of"

    # Organizations
    MEMBER_OF = "member_of"
    EMPLOYED_BY = "employed_by"
    FOUNDED = "founded"

    # Projects
    COLLABORATES_ON = "collaborates_on"
    OWNS = "owns"
    CONTRIBUTES_TO = "contributes_to"
    DEPENDS_ON = "depends_on"
    BLOCKED_BY = "blocked_by"

    # Structural
    PART_OF = "part_of"
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    LOCATED_AT = "located_at"
    BASED_IN = "based_in"

    # Co-mention (auto-created)
    MENTIONED_WITH = "mentioned_with"


# ── Typed Attribute Schemas ──────────────────────────────────────────


class PersonAttributes(BaseModel):
    """Typed attributes specific to a PERSON entity.

    Captures relationship to the user, role, contact info, and
    interaction patterns. All fields optional — populated incrementally
    as Blurt learns from natural speech.
    """

    entity_schema: Literal["person"] = "person"

    # Identity
    first_name: str | None = None
    last_name: str | None = None
    nickname: str | None = None

    # Relationship to user
    relationship_to_user: str | None = None  # e.g. "manager", "friend", "partner"
    closeness: str | None = None  # "close", "acquaintance", "professional"

    # Professional
    role: str | None = None  # e.g. "engineering manager", "designer"
    title: str | None = None  # e.g. "VP of Engineering"
    company: str | None = None
    team: str | None = None
    department: str | None = None

    # Contact
    email: str | None = None
    phone: str | None = None
    timezone: str | None = None  # e.g. "America/New_York"

    # Preferences & patterns (learned over time)
    preferred_contact_method: str | None = None
    availability_notes: str | None = None
    communication_style: str | None = None  # e.g. "direct", "detailed"
    birthday: date | None = None

    # Interaction tracking
    last_interaction_context: str | None = None
    interaction_frequency: str | None = None  # "daily", "weekly", "monthly", "rare"
    sentiment_trend: str | None = None  # "positive", "neutral", "tense"

    # Free-form overflow for anything that doesn't fit typed fields
    extra: dict[str, Any] = Field(default_factory=dict)


class PlaceAttributes(BaseModel):
    """Typed attributes specific to a PLACE entity.

    Captures location details, how the user relates to the place, and
    contextual information about what happens there.
    """

    entity_schema: Literal["place"] = "place"

    # Location
    address: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    coordinates: tuple[float, float] | None = None  # (lat, lng)
    timezone: str | None = None

    # Classification
    place_type: str | None = None  # "office", "home", "restaurant", "city", "country"
    category: str | None = None  # "work", "personal", "travel"

    # User relationship
    relationship_to_user: str | None = None  # "workplace", "home", "favorite café"
    visit_frequency: str | None = None  # "daily", "weekly", "occasionally"
    last_visit_context: str | None = None

    # Context
    associated_activities: list[str] = Field(default_factory=list)
    notes: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


class ProjectStatus(str, Enum):
    """Status of a project entity."""

    IDEA = "idea"
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ProjectAttributes(BaseModel):
    """Typed attributes specific to a PROJECT entity.

    Captures project lifecycle, stakeholders, deadlines, and how the
    user feels about the project. Status is shame-free — no overdue
    counters or forced urgency.
    """

    entity_schema: Literal["project"] = "project"

    # Lifecycle
    status: ProjectStatus = ProjectStatus.ACTIVE
    description: str | None = None
    start_date: date | None = None
    target_date: date | None = None  # NOT "deadline" — shame-free language
    completed_date: date | None = None

    # Ownership
    owner: str | None = None  # person name or entity reference
    stakeholders: list[str] = Field(default_factory=list)
    team: str | None = None
    organization: str | None = None

    # Classification
    category: str | None = None  # "work", "personal", "side-project"
    tags: list[str] = Field(default_factory=list)
    priority_hint: str | None = None  # "high", "medium", "low" — inferred, never forced

    # External links
    notion_page_id: str | None = None
    calendar_event_ids: list[str] = Field(default_factory=list)
    external_urls: list[str] = Field(default_factory=list)

    # User sentiment (detected, not set)
    user_energy_association: str | None = None  # "energizing", "draining", "neutral"
    momentum: str | None = None  # "gaining", "steady", "stalled"
    notes: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


class OrganizationAttributes(BaseModel):
    """Typed attributes specific to an ORGANIZATION entity.

    Captures the organization's relationship to the user, its people,
    and structural information. Works for companies, teams, communities,
    or any named group.
    """

    entity_schema: Literal["organization"] = "organization"

    # Identity
    full_name: str | None = None  # official name if the entity.name is shorthand
    org_type: str | None = None  # "company", "team", "community", "department", "client"
    industry: str | None = None
    website: str | None = None

    # User relationship
    relationship_to_user: str | None = None  # "employer", "client", "partner"
    user_role: str | None = None  # user's role within this org
    user_department: str | None = None
    user_team: str | None = None

    # People
    key_contacts: list[str] = Field(default_factory=list)  # person names/entity refs
    manager_name: str | None = None
    team_size: int | None = None

    # Location
    headquarters: str | None = None
    office_locations: list[str] = Field(default_factory=list)

    # External links
    notion_workspace_id: str | None = None
    calendar_id: str | None = None

    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ── Attribute schema registry ────────────────────────────────────────

# Maps EntityType to the typed attribute schema class.
# Entity types without a specialized schema (TOPIC, TOOL) use raw dicts.
ENTITY_ATTRIBUTE_SCHEMAS: dict[
    EntityType,
    type[PersonAttributes | PlaceAttributes | ProjectAttributes | OrganizationAttributes],
] = {
    EntityType.PERSON: PersonAttributes,
    EntityType.PLACE: PlaceAttributes,
    EntityType.PROJECT: ProjectAttributes,
    EntityType.ORGANIZATION: OrganizationAttributes,
}


def parse_typed_attributes(
    entity_type: EntityType,
    raw_attributes: dict[str, Any],
) -> PersonAttributes | PlaceAttributes | ProjectAttributes | OrganizationAttributes | None:
    """Parse a raw attributes dict into a typed schema, if one exists.

    Returns None for entity types that don't have a specialized schema
    (e.g. TOPIC, TOOL). Unknown keys are captured in the ``extra`` field
    so no data is lost.
    """
    schema_cls = ENTITY_ATTRIBUTE_SCHEMAS.get(entity_type)
    if schema_cls is None:
        return None

    # Separate known fields from unknown ones
    known_fields = set(schema_cls.model_fields.keys()) - {"extra"}
    known: dict[str, Any] = {}
    overflow: dict[str, Any] = {}

    for k, v in raw_attributes.items():
        if k in known_fields:
            known[k] = v
        else:
            overflow[k] = v

    known["extra"] = overflow
    return schema_cls.model_validate(known)


def typed_attributes_to_dict(
    attrs: PersonAttributes | PlaceAttributes | ProjectAttributes | OrganizationAttributes,
) -> dict[str, Any]:
    """Serialize typed attributes back into a flat dict for storage.

    Merges the ``extra`` overflow dict into the top-level result so that
    round-tripping through parse -> serialize is lossless.
    """
    data = attrs.model_dump(exclude_none=True, exclude={"extra", "entity_schema"})
    data.update(attrs.extra)
    return data


# ── Valid relationship matrix ────────────────────────────────────────
# Defines which relationship types are semantically valid between entity
# type pairs. Used for validation and to guide the extraction pipeline.

ValidRelationshipPair = tuple[EntityType, EntityType]

VALID_RELATIONSHIPS: dict[RelationshipType, list[ValidRelationshipPair]] = {
    # Person ↔ Person
    RelationshipType.WORKS_WITH: [
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationshipType.MANAGES: [
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PROJECT),
    ],
    RelationshipType.MANAGED_BY: [
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PROJECT, EntityType.PERSON),
    ],
    RelationshipType.REPORTS_TO: [
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationshipType.KNOWS: [
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationshipType.FRIEND_OF: [
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationshipType.FAMILY_OF: [
        (EntityType.PERSON, EntityType.PERSON),
    ],

    # Person ↔ Organization
    RelationshipType.MEMBER_OF: [
        (EntityType.PERSON, EntityType.ORGANIZATION),
    ],
    RelationshipType.EMPLOYED_BY: [
        (EntityType.PERSON, EntityType.ORGANIZATION),
    ],
    RelationshipType.FOUNDED: [
        (EntityType.PERSON, EntityType.ORGANIZATION),
    ],

    # Person / Org ↔ Project
    RelationshipType.COLLABORATES_ON: [
        (EntityType.PERSON, EntityType.PROJECT),
        (EntityType.ORGANIZATION, EntityType.PROJECT),
    ],
    RelationshipType.OWNS: [
        (EntityType.PERSON, EntityType.PROJECT),
        (EntityType.ORGANIZATION, EntityType.PROJECT),
    ],
    RelationshipType.CONTRIBUTES_TO: [
        (EntityType.PERSON, EntityType.PROJECT),
    ],
    RelationshipType.DEPENDS_ON: [
        (EntityType.PROJECT, EntityType.PROJECT),
    ],
    RelationshipType.BLOCKED_BY: [
        (EntityType.PROJECT, EntityType.PROJECT),
        (EntityType.PROJECT, EntityType.PERSON),
    ],

    # Structural / Location
    RelationshipType.PART_OF: [
        (EntityType.PERSON, EntityType.ORGANIZATION),
        (EntityType.PROJECT, EntityType.PROJECT),
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION),
        (EntityType.PLACE, EntityType.PLACE),
    ],
    RelationshipType.CONTAINS: [
        (EntityType.ORGANIZATION, EntityType.PERSON),
        (EntityType.PROJECT, EntityType.PROJECT),
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION),
        (EntityType.PLACE, EntityType.PLACE),
    ],
    RelationshipType.LOCATED_AT: [
        (EntityType.PERSON, EntityType.PLACE),
        (EntityType.ORGANIZATION, EntityType.PLACE),
        (EntityType.PROJECT, EntityType.PLACE),
    ],
    RelationshipType.BASED_IN: [
        (EntityType.PERSON, EntityType.PLACE),
        (EntityType.ORGANIZATION, EntityType.PLACE),
    ],

    # Universal
    RelationshipType.RELATED_TO: [
        (et1, et2)
        for et1 in EntityType
        for et2 in EntityType
    ],
    RelationshipType.MENTIONED_WITH: [
        (et1, et2)
        for et1 in EntityType
        for et2 in EntityType
    ],
}


def is_valid_relationship(
    source_type: EntityType,
    target_type: EntityType,
    relationship_type: RelationshipType,
) -> bool:
    """Check whether a relationship type is valid between two entity types."""
    valid_pairs = VALID_RELATIONSHIPS.get(relationship_type, [])
    return (source_type, target_type) in valid_pairs


class FactType(str, Enum):
    """Types of facts stored about entities or the user."""

    ATTRIBUTE = "attribute"      # "Sarah is my manager"
    PREFERENCE = "preference"    # "I prefer morning meetings"
    HABIT = "habit"              # "I usually skip tasks after 4pm"
    ASSOCIATION = "association"  # "Q2 deck = the launch project"
    ALIAS = "alias"              # "the deck" -> "Q2 planning deck"


class PatternType(str, Enum):
    """Types of learned behavioral patterns."""

    TIME_OF_DAY = "time_of_day"
    DAY_OF_WEEK = "day_of_week"
    MOOD_CYCLE = "mood_cycle"
    COMPLETION_SIGNAL = "completion_signal"
    SKIP_SIGNAL = "skip_signal"
    ENERGY_RHYTHM = "energy_rhythm"
    ENTITY_PATTERN = "entity_pattern"


class EntityNode(BaseModel):
    """A node in the personal knowledge graph.

    Represents a person, place, project, organization, topic, or tool
    that the user has mentioned. Tracks co-mention strength, last seen
    time, and an embedding vector for semantic search.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    normalized_name: str = ""
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    mention_count: int = 0
    first_seen: datetime = Field(default_factory=_utcnow)
    last_seen: datetime = Field(default_factory=_utcnow)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def model_post_init(self, __context: Any) -> None:
        if not self.normalized_name:
            self.normalized_name = self.name.lower().strip()


class RelationshipEdge(BaseModel):
    """An edge connecting two entities in the knowledge graph.

    Strength increases with co-mentions and decays over time with absence.
    This enables the compounding knowledge property.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    strength: float = Field(default=1.0, ge=0.0, le=100.0)
    co_mention_count: int = 1
    context_snippets: list[str] = Field(default_factory=list)
    first_seen: datetime = Field(default_factory=_utcnow)
    last_seen: datetime = Field(default_factory=_utcnow)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Fact(BaseModel):
    """A fact or preference learned about the user's world.

    Facts are extracted from natural speech and stored with embeddings
    for semantic retrieval. Examples:
    - "Sarah is my manager" (ATTRIBUTE)
    - "I prefer morning meetings" (PREFERENCE)
    - "the deck" means "Q2 planning deck" (ALIAS)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    fact_type: FactType
    subject_entity_id: str | None = None
    content: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_blurt_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    is_active: bool = True
    superseded_by: str | None = None
    first_learned: datetime = Field(default_factory=_utcnow)
    last_confirmed: datetime = Field(default_factory=_utcnow)
    confirmation_count: int = 1
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class LearnedPattern(BaseModel):
    """A behavioral pattern detected from accumulated observations.

    Patterns emerge from the episodic memory tier and get promoted to
    semantic memory once they reach sufficient confidence.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    pattern_type: PatternType
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observation_count: int = 0
    supporting_evidence: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    is_active: bool = True
    first_detected: datetime = Field(default_factory=_utcnow)
    last_confirmed: datetime = Field(default_factory=_utcnow)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class SemanticSearchResult(BaseModel):
    """Result from a semantic search over the knowledge graph."""

    item_type: str  # "entity", "fact", "pattern", "relationship"
    item_id: str
    content: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
