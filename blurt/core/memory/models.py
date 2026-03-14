"""Memory tier models for the 3-tier memory system.

Working memory: session-scoped, ephemeral, recent context.
Episodic memory: append-only observations with timestamps, compressed over time.
Semantic memory: entity graph nodes with relationships, co-mention strength, embeddings.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class MemoryTier(str, enum.Enum):
    """The three tiers of memory."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class IntentType(str, enum.Enum):
    """Blurt intent classification types."""

    TASK = "task"
    EVENT = "event"
    REMINDER = "reminder"
    IDEA = "idea"
    JOURNAL = "journal"
    UPDATE = "update"
    QUESTION = "question"


@dataclass
class EmotionState:
    """Plutchik-based emotion detection result."""

    primary: str  # e.g., "joy", "anger", "fear", "surprise", etc.
    intensity: float  # 0.0–3.0
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0


@dataclass
class Entity:
    """An extracted entity (person, place, project, organization)."""

    name: str
    entity_type: str  # "person", "place", "project", "organization"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemoryItem:
    """Session-scoped working memory item.

    Short-lived, holds the most recent context for active processing.
    Candidates for promotion to episodic memory when session ends or
    importance threshold is crossed.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    intent: IntentType | None = None
    confidence: float = 0.0
    entities: list[Entity] = field(default_factory=list)
    emotion: EmotionState | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 1
    source_blurt_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Seconds since this item was created."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class EpisodicMemoryItem:
    """Append-only episodic observation.

    Stores a concrete observation with full context. Compressed over time.
    Can be promoted to semantic memory when patterns emerge.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    intent: IntentType | None = None
    entities: list[Entity] = field(default_factory=list)
    emotion: EmotionState | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_working_id: str | None = None
    access_count: int = 1
    mention_count: int = 1
    importance_score: float = 0.0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipEdge:
    """An edge in the semantic entity graph."""

    target_entity_id: str
    relationship_type: str  # e.g., "works_with", "located_at", "part_of"
    co_mention_count: int = 1
    strength: float = 0.0  # 0.0–1.0, computed from co-mentions and recency
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SemanticMemoryItem:
    """Semantic memory node — entity graph entry.

    Represents a consolidated understanding of an entity with relationship
    edges, embedding, and accumulated knowledge.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity: Entity | None = None
    summary: str = ""
    relationships: list[RelationshipEdge] = field(default_factory=list)
    source_episodic_ids: list[str] = field(default_factory=list)
    mention_count: int = 1
    importance_score: float = 0.0
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionEvent:
    """Record of a memory promotion from one tier to another."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    source_tier: MemoryTier = MemoryTier.WORKING
    target_tier: MemoryTier = MemoryTier.EPISODIC
    target_id: str = ""
    reason: str = ""
    score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
