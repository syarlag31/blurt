"""Working memory tier - short-term conversation buffer with TTL-based expiration.

Working memory holds the active context from the current session. Entries
automatically expire after a configurable TTL so stale context never leaks
into future reasoning. The buffer is bounded in size; when capacity is
exceeded the oldest entries are evicted first (regardless of TTL).

Key design decisions:
- Thread-safe via asyncio.Lock (async-first, no threading locks)
- TTL is per-entry, not global, allowing different lifetimes for different content
- Session context aggregates mood, energy, active task, and recent entities
- All timestamps use UTC
- No persistence — working memory is ephemeral by design
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EmotionLabel(str, Enum):
    """Plutchik's 8 primary emotions."""

    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class IntentType(str, Enum):
    """The 7 classification intents for a blurt."""

    TASK = "task"
    EVENT = "event"
    REMINDER = "reminder"
    IDEA = "idea"
    JOURNAL = "journal"
    UPDATE = "update"
    QUESTION = "question"


@dataclass
class EmotionState:
    """Detected emotional state using Plutchik's model."""

    primary: EmotionLabel = EmotionLabel.TRUST
    intensity: float = 0.0  # 0.0–3.0
    valence: float = 0.0  # -1.0 to 1.0 (negative to positive)
    arousal: float = 0.0  # 0.0 to 1.0 (calm to excited)


@dataclass
class WorkingMemoryEntry:
    """A single entry in working memory.

    Represents a processed blurt or system event that is part of the
    current active context. Each entry has its own TTL (time-to-live)
    in seconds after which it will be automatically expired.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    intent: IntentType | None = None
    confidence: float = 0.0
    entities: list[dict[str, Any]] = field(default_factory=list)
    emotion: EmotionState = field(default_factory=EmotionState)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # default 5 minutes
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "voice"  # voice | text | system

    @property
    def expires_at(self) -> float:
        """Unix timestamp when this entry expires."""
        return self.created_at + self.ttl_seconds

    @property
    def is_expired(self) -> bool:
        """Check whether this entry has passed its TTL."""
        return time.time() > self.expires_at

    @property
    def remaining_ttl(self) -> float:
        """Seconds remaining before expiration. Returns 0 if already expired."""
        return max(0.0, self.expires_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for API responses."""
        return {
            "id": self.id,
            "content": self.content,
            "intent": self.intent.value if self.intent else None,
            "confidence": self.confidence,
            "entities": self.entities,
            "emotion": {
                "primary": self.emotion.primary.value,
                "intensity": self.emotion.intensity,
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
            },
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "remaining_ttl": self.remaining_ttl,
            "is_expired": self.is_expired,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class SessionContext:
    """Aggregated session-level context derived from working memory.

    This is a rolling summary that gets rebuilt each time working memory
    is queried — it reflects the *current* state of the conversation.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_mood: EmotionState = field(default_factory=EmotionState)
    energy_level: float = 0.5  # 0.0–1.0
    active_task_id: str | None = None
    recent_intents: list[IntentType] = field(default_factory=list)
    recent_entities: list[dict[str, Any]] = field(default_factory=list)
    entry_count: int = 0
    last_interaction_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for API responses."""
        return {
            "session_id": self.session_id,
            "current_mood": {
                "primary": self.current_mood.primary.value,
                "intensity": self.current_mood.intensity,
                "valence": self.current_mood.valence,
                "arousal": self.current_mood.arousal,
            },
            "energy_level": self.energy_level,
            "active_task_id": self.active_task_id,
            "recent_intents": [i.value for i in self.recent_intents],
            "recent_entities": self.recent_entities,
            "entry_count": self.entry_count,
            "last_interaction_at": self.last_interaction_at,
        }


class WorkingMemory:
    """Short-term conversation buffer with TTL-based expiration.

    Thread-safe (via asyncio.Lock), bounded-size buffer that holds the
    active context for a single session. Entries expire individually
    based on their TTL; the buffer is also capped at ``max_entries``
    with oldest-first eviction.

    Usage::

        wm = WorkingMemory(session_id="abc", max_entries=50)
        entry = await wm.add("I need to call Sarah tomorrow", intent=IntentType.TASK)
        context = await wm.get_context()
        active = await wm.get_active_entries()
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_entries: int = 100,
        default_ttl: float = 300.0,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._entries: list[WorkingMemoryEntry] = []
        self._lock = asyncio.Lock()
        self._active_task_id: str | None = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def add(
        self,
        content: str,
        *,
        intent: IntentType | None = None,
        confidence: float = 0.0,
        entities: list[dict[str, Any]] | None = None,
        emotion: EmotionState | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "voice",
    ) -> WorkingMemoryEntry:
        """Add a new entry to working memory.

        If the buffer is at capacity, the oldest entry is evicted first.
        Expired entries are also pruned on every add.
        """
        entry = WorkingMemoryEntry(
            content=content,
            intent=intent,
            confidence=confidence,
            entities=entities or [],
            emotion=emotion or EmotionState(),
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self.default_ttl,
            metadata=metadata or {},
            source=source,
        )

        async with self._lock:
            # Prune expired entries first
            self._prune_expired()

            # Evict oldest if at capacity
            while len(self._entries) >= self.max_entries:
                self._entries.pop(0)

            self._entries.append(entry)

        return entry

    async def get_active_entries(self) -> list[WorkingMemoryEntry]:
        """Return all non-expired entries, ordered oldest-first."""
        async with self._lock:
            self._prune_expired()
            return list(self._entries)

    async def get_entry(self, entry_id: str) -> WorkingMemoryEntry | None:
        """Look up a single entry by ID. Returns None if not found or expired."""
        async with self._lock:
            self._prune_expired()
            for entry in self._entries:
                if entry.id == entry_id:
                    return entry
        return None

    async def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if it was found and removed."""
        async with self._lock:
            for i, entry in enumerate(self._entries):
                if entry.id == entry_id:
                    self._entries.pop(i)
                    return True
        return False

    async def clear(self) -> int:
        """Remove all entries. Returns the count of entries that were cleared."""
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._active_task_id = None
            return count

    async def size(self) -> int:
        """Return the number of active (non-expired) entries."""
        async with self._lock:
            self._prune_expired()
            return len(self._entries)

    async def set_active_task(self, task_id: str | None) -> None:
        """Set or clear the currently active task for this session."""
        async with self._lock:
            self._active_task_id = task_id

    # ------------------------------------------------------------------
    # Context aggregation
    # ------------------------------------------------------------------

    async def get_context(self) -> SessionContext:
        """Build a SessionContext snapshot from current working memory state.

        Aggregates mood (most recent emotion), energy, active intents,
        and mentioned entities into a single summary object.
        """
        async with self._lock:
            self._prune_expired()
            entries = list(self._entries)

        ctx = SessionContext(session_id=self.session_id)
        ctx.entry_count = len(entries)

        if not entries:
            return ctx

        # Most recent emotion becomes current mood
        ctx.current_mood = entries[-1].emotion

        # Energy estimated from recent valence + arousal (simple heuristic)
        recent = entries[-5:]  # last 5 entries
        if recent:
            avg_valence = sum(e.emotion.valence for e in recent) / len(recent)
            avg_arousal = sum(e.emotion.arousal for e in recent) / len(recent)
            # Map valence [-1,1] and arousal [0,1] to energy [0,1]
            ctx.energy_level = round(
                max(0.0, min(1.0, 0.5 + avg_valence * 0.3 + avg_arousal * 0.2)),
                3,
            )

        ctx.active_task_id = self._active_task_id

        # Collect recent intents (deduplicated, preserving order)
        seen_intents: set[IntentType] = set()
        for entry in reversed(entries):
            if entry.intent and entry.intent not in seen_intents:
                ctx.recent_intents.append(entry.intent)
                seen_intents.add(entry.intent)
            if len(ctx.recent_intents) >= 7:
                break
        ctx.recent_intents.reverse()

        # Collect recent unique entities
        seen_entity_names: set[str] = set()
        for entry in reversed(entries):
            for entity in entry.entities:
                name = entity.get("name", "")
                if name and name not in seen_entity_names:
                    ctx.recent_entities.append(entity)
                    seen_entity_names.add(name)
                if len(ctx.recent_entities) >= 20:
                    break
            if len(ctx.recent_entities) >= 20:
                break
        ctx.recent_entities.reverse()

        ctx.last_interaction_at = entries[-1].created_at

        return ctx

    async def get_recent_content(self, limit: int = 10) -> list[str]:
        """Return the text content of the most recent N active entries.

        Useful for building prompts with conversational context.
        """
        async with self._lock:
            self._prune_expired()
            recent = self._entries[-limit:]
        return [e.content for e in recent]

    async def get_entries_by_intent(self, intent: IntentType) -> list[WorkingMemoryEntry]:
        """Return active entries matching a specific intent type."""
        async with self._lock:
            self._prune_expired()
            return [e for e in self._entries if e.intent == intent]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_expired(self) -> int:
        """Remove expired entries. Returns count removed.

        Must be called while holding ``self._lock``.
        """
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired]
        return before - len(self._entries)
