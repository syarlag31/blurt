"""Episodic memory store — append-only storage and retrieval of conversation episodes.

Tier 2 of Blurt's 3-tier memory system. Every blurt creates an episode that
records what was said, the classified intent, detected emotion, extracted entities,
session context, and behavioral signals (task completed/skipped/dismissed).

Key properties:
- Append-only: episodes are never modified or deleted
- Timestamped: every episode has precise UTC temporal context
- Emotionally tagged: Plutchik emotion + intensity + valence + arousal
- Entity-linked: extracted entities for knowledge graph
- Behaviorally tracked: user action on surfaced tasks feeds learning
- Compressible: older episodes can be summarized for efficiency
- Searchable: by time range, entity, emotion, intent, and semantic similarity

Storage backend is pluggable (in-memory for tests, persistent for production).
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class InputModality(str, Enum):
    """How the blurt was received."""

    VOICE = "voice"
    TEXT = "text"


class BehavioralSignal(str, Enum):
    """User action on a surfaced task — feeds behavioral learning."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    DISMISSED = "dismissed"
    POSTPONED = "postponed"
    BROKEN_DOWN = "broken_down"
    NONE = "none"


@dataclass
class EmotionSnapshot:
    """Emotion state captured at episode time. Uses Plutchik's 8 primary emotions."""

    primary: str = "trust"  # joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    intensity: float = 0.0  # 0.0–3.0
    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.0  # 0.0 (calm) to 1.0 (excited)
    secondary: str | None = None  # optional secondary emotion

    @property
    def is_negative(self) -> bool:
        return self.valence < -0.2

    @property
    def is_high_arousal(self) -> bool:
        return self.arousal > 0.6


@dataclass
class EntityRef:
    """Reference to an entity mentioned in an episode."""

    name: str
    entity_type: str  # person, place, project, organization, topic
    entity_id: str | None = None  # resolved ID from knowledge graph
    confidence: float = 1.0


@dataclass
class EpisodeContext:
    """Contextual metadata captured at episode creation time."""

    time_of_day: str = "morning"  # morning, afternoon, evening, night
    day_of_week: str = "monday"  # monday..sunday
    session_id: str = ""
    preceding_episode_id: str | None = None
    active_task_id: str | None = None


@dataclass
class Episode:
    """A single episodic memory entry — one observation from a blurt.

    Append-only: once created, episodes are never modified or deleted.
    They may be compressed into summaries over time, but raw data is preserved.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # What was said
    raw_text: str = ""
    modality: InputModality = InputModality.VOICE

    # Classification
    intent: str = "task"  # task, event, reminder, idea, journal, update, question
    intent_confidence: float = 0.0

    # Emotional context
    emotion: EmotionSnapshot = field(default_factory=EmotionSnapshot)

    # Entity associations
    entities: list[EntityRef] = field(default_factory=list)

    # Behavioral signals
    behavioral_signal: BehavioralSignal = BehavioralSignal.NONE
    surfaced_task_id: str | None = None

    # Session context
    context: EpisodeContext = field(default_factory=EpisodeContext)

    # Compression tracking
    is_compressed: bool = False
    compressed_into_id: str | None = None

    # Embedding for semantic search (Gemini 2 embeddings)
    embedding: list[float] | None = None

    # Link back to source working memory / core memory item
    source_working_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "raw_text": self.raw_text,
            "modality": self.modality.value,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "emotion": {
                "primary": self.emotion.primary,
                "intensity": self.emotion.intensity,
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "secondary": self.emotion.secondary,
            },
            "entities": [
                {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "entity_id": e.entity_id,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "behavioral_signal": self.behavioral_signal.value,
            "surfaced_task_id": self.surfaced_task_id,
            "context": {
                "time_of_day": self.context.time_of_day,
                "day_of_week": self.context.day_of_week,
                "session_id": self.context.session_id,
                "preceding_episode_id": self.context.preceding_episode_id,
                "active_task_id": self.context.active_task_id,
            },
            "is_compressed": self.is_compressed,
            "compressed_into_id": self.compressed_into_id,
        }


@dataclass
class EpisodeSummary:
    """Compressed summary of multiple episodes over a time range.

    Raw episodes are preserved but marked as compressed. Summaries capture
    aggregated patterns for efficient long-term recall.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_episode_ids: list[str] = field(default_factory=list)
    episode_count: int = 0
    summary_text: str = ""
    dominant_emotions: list[EmotionSnapshot] = field(default_factory=list)
    entity_mentions: dict[str, int] = field(default_factory=dict)
    intent_distribution: dict[str, int] = field(default_factory=dict)
    behavioral_signals: dict[str, int | str] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "source_episode_ids": self.source_episode_ids,
            "episode_count": self.episode_count,
            "summary_text": self.summary_text,
            "dominant_emotions": [
                {"primary": e.primary, "intensity": e.intensity,
                 "valence": e.valence, "arousal": e.arousal}
                for e in self.dominant_emotions
            ],
            "entity_mentions": self.entity_mentions,
            "intent_distribution": self.intent_distribution,
            "behavioral_signals": self.behavioral_signals,
        }


# ---------------------------------------------------------------------------
# Query filters — composable predicates for episode retrieval
# ---------------------------------------------------------------------------


class TimeRangeFilter:
    """Filter episodes by time range."""

    def __init__(self, start: datetime | None = None, end: datetime | None = None):
        self.start = start
        self.end = end

    def matches(self, episode: Episode) -> bool:
        if self.start and episode.timestamp < self.start:
            return False
        if self.end and episode.timestamp > self.end:
            return False
        return True


class EntityFilter:
    """Filter episodes mentioning a specific entity."""

    def __init__(self, entity_name: str | None = None, entity_id: str | None = None):
        self.entity_name = entity_name.lower() if entity_name else None
        self.entity_id = entity_id

    def matches(self, episode: Episode) -> bool:
        for entity in episode.entities:
            if self.entity_id and entity.entity_id == self.entity_id:
                return True
            if self.entity_name and entity.name.lower() == self.entity_name:
                return True
        return False


class EmotionFilter:
    """Filter episodes by emotional state."""

    def __init__(
        self,
        primary: str | None = None,
        min_intensity: float = 0.0,
        valence_range: tuple[float, float] | None = None,
    ):
        self.primary = primary
        self.min_intensity = min_intensity
        self.valence_range = valence_range

    def matches(self, episode: Episode) -> bool:
        if self.primary and episode.emotion.primary != self.primary:
            return False
        if episode.emotion.intensity < self.min_intensity:
            return False
        if self.valence_range:
            lo, hi = self.valence_range
            if not (lo <= episode.emotion.valence <= hi):
                return False
        return True


class IntentFilter:
    """Filter episodes by intent type."""

    def __init__(self, intent: str):
        self.intent = intent

    def matches(self, episode: Episode) -> bool:
        return episode.intent == self.intent


class BehavioralFilter:
    """Filter episodes by behavioral signal."""

    def __init__(self, signal: BehavioralSignal):
        self.signal = signal

    def matches(self, episode: Episode) -> bool:
        return episode.behavioral_signal == self.signal


class SessionFilter:
    """Filter episodes within a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def matches(self, episode: Episode) -> bool:
        return episode.context.session_id == self.session_id


# ---------------------------------------------------------------------------
# Abstract store interface
# ---------------------------------------------------------------------------


class EpisodicMemoryStore(ABC):
    """Abstract interface for episodic memory storage.

    All implementations must guarantee:
    - Append-only semantics (no update/delete of episodes)
    - Chronological ordering by timestamp
    - Efficient filtering by time, entity, emotion, intent
    """

    @abstractmethod
    async def append(self, episode: Episode) -> Episode:
        """Store a new episode. Returns the stored episode."""
        ...

    @abstractmethod
    async def get(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID."""
        ...

    @abstractmethod
    async def query(
        self,
        user_id: str,
        *,
        time_range: TimeRangeFilter | None = None,
        entity_filter: EntityFilter | None = None,
        emotion_filter: EmotionFilter | None = None,
        intent_filter: IntentFilter | None = None,
        behavioral_filter: BehavioralFilter | None = None,
        session_filter: SessionFilter | None = None,
        limit: int = 50,
        offset: int = 0,
        include_compressed: bool = False,
    ) -> list[Episode]:
        """Query episodes with composable filters. Returns newest first."""
        ...

    @abstractmethod
    async def count(self, user_id: str) -> int:
        """Total episode count for a user."""
        ...

    @abstractmethod
    async def get_session_episodes(self, session_id: str) -> list[Episode]:
        """Get all episodes in a session, ordered chronologically."""
        ...

    @abstractmethod
    async def semantic_search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[tuple[Episode, float]]:
        """Search by embedding similarity. Returns (episode, score) pairs."""
        ...

    @abstractmethod
    async def get_entity_timeline(
        self,
        user_id: str,
        entity_name: str,
        limit: int = 20,
    ) -> list[Episode]:
        """Get episodes mentioning a specific entity, newest first."""
        ...

    @abstractmethod
    async def get_emotion_timeline(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[Episode]:
        """Get episodes in a time range for emotion pattern analysis."""
        ...

    @abstractmethod
    async def mark_compressed(
        self, episode_ids: list[str], summary_id: str
    ) -> int:
        """Mark episodes as compressed into a summary. Returns count updated."""
        ...

    @abstractmethod
    async def store_summary(self, summary: EpisodeSummary) -> EpisodeSummary:
        """Store an episode summary."""
        ...

    @abstractmethod
    async def get_summaries(
        self,
        user_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EpisodeSummary]:
        """Retrieve summaries for a time range."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_summary(
    user_id: str,
    episodes: list[Episode],
    summary_text: str,
    embedding: list[float] | None = None,
) -> EpisodeSummary:
    """Build a summary from a list of episodes.

    Args:
        user_id: Owner of the episodes.
        episodes: Episodes to compress. Must be non-empty.
        summary_text: LLM-generated natural language summary.
        embedding: Optional embedding vector for semantic search.
    """
    if not episodes:
        raise ValueError("Cannot build summary from empty episode list")

    sorted_eps = sorted(episodes, key=lambda e: e.timestamp)

    # Aggregate emotions — most common primary emotions
    emotion_counter: Counter[str] = Counter()
    for ep in episodes:
        emotion_counter[ep.emotion.primary] += 1

    dominant_emotions = []
    for primary_val, _count in emotion_counter.most_common(3):
        for ep in episodes:
            if ep.emotion.primary == primary_val:
                dominant_emotions.append(ep.emotion)
                break

    # Entity mention counts
    entity_counts: Counter[str] = Counter()
    for ep in episodes:
        for ent in ep.entities:
            entity_counts[ent.name] += 1

    # Intent distribution
    intent_counts: Counter[str] = Counter()
    for ep in episodes:
        intent_counts[ep.intent] += 1

    # Behavioral signal counts
    signal_counts: Counter[str] = Counter()
    for ep in episodes:
        if ep.behavioral_signal != BehavioralSignal.NONE:
            signal_counts[ep.behavioral_signal.value] += 1

    return EpisodeSummary(
        user_id=user_id,
        period_start=sorted_eps[0].timestamp,
        period_end=sorted_eps[-1].timestamp,
        source_episode_ids=[ep.id for ep in episodes],
        episode_count=len(episodes),
        summary_text=summary_text,
        dominant_emotions=dominant_emotions,
        entity_mentions=dict(entity_counts),
        intent_distribution=dict(intent_counts),
        behavioral_signals=dict(signal_counts),
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryEpisodicStore(EpisodicMemoryStore):
    """In-memory implementation for testing and local-only mode.

    Uses dict-based indexes for efficient filtering. Production backends
    (SQLite, cloud DB) would use SQL indexes instead.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        self._user_index: dict[str, list[str]] = defaultdict(list)
        self._session_index: dict[str, list[str]] = defaultdict(list)
        self._entity_index: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )  # user_id -> entity_name_lower -> [episode_ids]
        self._summaries: dict[str, EpisodeSummary] = {}
        self._user_summaries: dict[str, list[str]] = defaultdict(list)

    async def append(self, episode: Episode) -> Episode:
        if episode.id in self._episodes:
            raise ValueError(f"Episode {episode.id} already exists (append-only)")

        self._episodes[episode.id] = episode
        self._user_index[episode.user_id].append(episode.id)
        self._session_index[episode.context.session_id].append(episode.id)

        for entity in episode.entities:
            key = entity.name.lower()
            self._entity_index[episode.user_id][key].append(episode.id)

        return episode

    async def get(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)

    async def query(
        self,
        user_id: str,
        *,
        time_range: TimeRangeFilter | None = None,
        entity_filter: EntityFilter | None = None,
        emotion_filter: EmotionFilter | None = None,
        intent_filter: IntentFilter | None = None,
        behavioral_filter: BehavioralFilter | None = None,
        session_filter: SessionFilter | None = None,
        limit: int = 50,
        offset: int = 0,
        include_compressed: bool = False,
    ) -> list[Episode]:
        episode_ids = self._user_index.get(user_id, [])
        candidates = [self._episodes[eid] for eid in episode_ids]

        filters: list[Any] = []
        if time_range:
            filters.append(time_range.matches)
        if entity_filter:
            filters.append(entity_filter.matches)
        if emotion_filter:
            filters.append(emotion_filter.matches)
        if intent_filter:
            filters.append(intent_filter.matches)
        if behavioral_filter:
            filters.append(behavioral_filter.matches)
        if session_filter:
            filters.append(session_filter.matches)
        if not include_compressed:
            filters.append(lambda ep: not ep.is_compressed)

        for f in filters:
            candidates = [ep for ep in candidates if f(ep)]

        candidates.sort(key=lambda ep: ep.timestamp, reverse=True)
        return candidates[offset : offset + limit]

    async def count(self, user_id: str) -> int:
        return len(self._user_index.get(user_id, []))

    async def get_session_episodes(self, session_id: str) -> list[Episode]:
        episode_ids = self._session_index.get(session_id, [])
        episodes = [self._episodes[eid] for eid in episode_ids]
        episodes.sort(key=lambda ep: ep.timestamp)
        return episodes

    async def semantic_search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[tuple[Episode, float]]:
        episode_ids = self._user_index.get(user_id, [])
        scored: list[tuple[Episode, float]] = []

        for eid in episode_ids:
            ep = self._episodes[eid]
            if ep.is_compressed or ep.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, ep.embedding)
            if sim >= min_similarity:
                scored.append((ep, sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    async def get_entity_timeline(
        self,
        user_id: str,
        entity_name: str,
        limit: int = 20,
    ) -> list[Episode]:
        key = entity_name.lower()
        episode_ids = self._entity_index.get(user_id, {}).get(key, [])
        episodes = [
            self._episodes[eid]
            for eid in episode_ids
            if not self._episodes[eid].is_compressed
        ]
        episodes.sort(key=lambda ep: ep.timestamp, reverse=True)
        return episodes[:limit]

    async def get_emotion_timeline(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[Episode]:
        episode_ids = self._user_index.get(user_id, [])
        episodes = [
            self._episodes[eid]
            for eid in episode_ids
            if start <= self._episodes[eid].timestamp <= end
            and not self._episodes[eid].is_compressed
        ]
        episodes.sort(key=lambda ep: ep.timestamp)
        return episodes

    async def mark_compressed(
        self, episode_ids: list[str], summary_id: str
    ) -> int:
        count = 0
        for eid in episode_ids:
            ep = self._episodes.get(eid)
            if ep and not ep.is_compressed:
                ep.is_compressed = True
                ep.compressed_into_id = summary_id
                count += 1
        return count

    async def store_summary(self, summary: EpisodeSummary) -> EpisodeSummary:
        self._summaries[summary.id] = summary
        self._user_summaries[summary.user_id].append(summary.id)
        return summary

    async def get_summaries(
        self,
        user_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EpisodeSummary]:
        summary_ids = self._user_summaries.get(user_id, [])
        summaries = [self._summaries[sid] for sid in summary_ids]

        if start:
            summaries = [s for s in summaries if s.period_end >= start]
        if end:
            summaries = [s for s in summaries if s.period_start <= end]

        summaries.sort(key=lambda s: s.period_start, reverse=True)
        return summaries


async def compress_episodes(
    store: EpisodicMemoryStore,
    user_id: str,
    episodes: list[Episode],
    summary_text: str,
    embedding: list[float] | None = None,
) -> EpisodeSummary:
    """Full compression pipeline: build summary, store it, mark episodes."""
    summary = build_summary(user_id, episodes, summary_text, embedding)
    stored_summary = await store.store_summary(summary)
    await store.mark_compressed(
        [ep.id for ep in episodes], stored_summary.id
    )
    return stored_summary
