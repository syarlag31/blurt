"""Observation model — the raw timestamped record of what a user said/did.

An Observation is the input unit to the episodic memory system. Every blurt
(voice or text) creates an Observation that captures:
- The raw content exactly as received
- Input modality (voice/text)
- Precise UTC timestamp
- Session and contextual metadata
- Pipeline results (intent, emotion, entities) once classification runs

The Observation is then stored as an Episode in the append-only episodic store.
This separation keeps the ingestion contract clean: callers create Observations,
the memory system persists Episodes.

Design:
- Immutable after creation (append-only semantics flow from here)
- All metadata captured at observation time, not retroactively
- Factory methods for common creation patterns (from voice, from text, from pipeline)
- Conversion to Episode is explicit and lossless
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodicMemoryStore,
    InputModality,
)


@dataclass(frozen=True)
class ObservationMetadata:
    """Contextual metadata captured at the moment of observation.

    Frozen to enforce immutability — metadata is fixed at capture time.
    """

    session_id: str = ""
    time_of_day: str = "morning"  # morning, afternoon, evening, night
    day_of_week: str = "monday"  # monday..sunday
    preceding_episode_id: str | None = None
    active_task_id: str | None = None
    device_type: str | None = None  # mobile, desktop, wearable
    locale: str | None = None  # e.g. "en-US"
    audio_duration_ms: int | None = None  # duration of voice input
    transcription_confidence: float | None = None  # STT confidence if voice

    def to_episode_context(self) -> EpisodeContext:
        """Convert to EpisodeContext for storage."""
        return EpisodeContext(
            time_of_day=self.time_of_day,
            day_of_week=self.day_of_week,
            session_id=self.session_id,
            preceding_episode_id=self.preceding_episode_id,
            active_task_id=self.active_task_id,
        )


@dataclass(frozen=True)
class Observation:
    """A single timestamped observation from a user blurt.

    Immutable record of exactly what was captured and when. Created at
    the moment the user speaks or types, enriched by the classification
    pipeline, then persisted as an Episode.

    Fields:
        id: Unique observation identifier (UUID4).
        user_id: Owner of this observation.
        timestamp: Precise UTC time of capture.
        raw_text: The transcribed or typed text content.
        modality: How the input was received (voice or text).
        metadata: Contextual metadata from the capture moment.
        intent: Classified intent (set by pipeline, None before classification).
        intent_confidence: Classification confidence score.
        emotion: Detected emotion state (set by pipeline).
        entities: Extracted entity references (set by pipeline).
        behavioral_signal: User action on surfaced task (if applicable).
        surfaced_task_id: ID of the task that was surfaced when this was captured.
        embedding: Semantic embedding vector (set by embedding pipeline).
        source_working_id: Link to working memory item if promoted.
        extra: Extensible metadata bag for pipeline-specific data.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_text: str = ""
    modality: InputModality = InputModality.VOICE
    metadata: ObservationMetadata = field(default_factory=ObservationMetadata)

    # Pipeline enrichment (populated after classification)
    intent: str | None = None
    intent_confidence: float = 0.0
    emotion: EmotionSnapshot = field(default_factory=EmotionSnapshot)
    entities: tuple[EntityRef, ...] = field(default_factory=tuple)
    behavioral_signal: BehavioralSignal = BehavioralSignal.NONE
    surfaced_task_id: str | None = None
    embedding: tuple[float, ...] | None = None
    source_working_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_episode(self) -> Episode:
        """Convert this observation to an Episode for episodic storage.

        This is the canonical path from observation -> persistent memory.
        The conversion is lossless for all fields that Episode supports.
        """
        return Episode(
            id=self.id,
            user_id=self.user_id,
            timestamp=self.timestamp,
            raw_text=self.raw_text,
            modality=self.modality,
            intent=self.intent or "task",
            intent_confidence=self.intent_confidence,
            emotion=self.emotion,
            entities=list(self.entities),
            behavioral_signal=self.behavioral_signal,
            surfaced_task_id=self.surfaced_task_id,
            context=self.metadata.to_episode_context(),
            embedding=list(self.embedding) if self.embedding else None,
            source_working_id=self.source_working_id,
        )

    def with_classification(
        self,
        intent: str,
        confidence: float,
    ) -> Observation:
        """Return a new Observation with classification results applied."""
        return Observation(
            id=self.id,
            user_id=self.user_id,
            timestamp=self.timestamp,
            raw_text=self.raw_text,
            modality=self.modality,
            metadata=self.metadata,
            intent=intent,
            intent_confidence=confidence,
            emotion=self.emotion,
            entities=self.entities,
            behavioral_signal=self.behavioral_signal,
            surfaced_task_id=self.surfaced_task_id,
            embedding=self.embedding,
            source_working_id=self.source_working_id,
            extra=self.extra,
        )

    def with_emotion(self, emotion: EmotionSnapshot) -> Observation:
        """Return a new Observation with emotion detection applied."""
        return Observation(
            id=self.id,
            user_id=self.user_id,
            timestamp=self.timestamp,
            raw_text=self.raw_text,
            modality=self.modality,
            metadata=self.metadata,
            intent=self.intent,
            intent_confidence=self.intent_confidence,
            emotion=emotion,
            entities=self.entities,
            behavioral_signal=self.behavioral_signal,
            surfaced_task_id=self.surfaced_task_id,
            embedding=self.embedding,
            source_working_id=self.source_working_id,
            extra=self.extra,
        )

    def with_entities(self, entities: list[EntityRef]) -> Observation:
        """Return a new Observation with extracted entities."""
        return Observation(
            id=self.id,
            user_id=self.user_id,
            timestamp=self.timestamp,
            raw_text=self.raw_text,
            modality=self.modality,
            metadata=self.metadata,
            intent=self.intent,
            intent_confidence=self.intent_confidence,
            emotion=self.emotion,
            entities=tuple(entities),
            behavioral_signal=self.behavioral_signal,
            surfaced_task_id=self.surfaced_task_id,
            embedding=self.embedding,
            source_working_id=self.source_working_id,
            extra=self.extra,
        )

    def with_embedding(self, embedding: list[float]) -> Observation:
        """Return a new Observation with semantic embedding."""
        return Observation(
            id=self.id,
            user_id=self.user_id,
            timestamp=self.timestamp,
            raw_text=self.raw_text,
            modality=self.modality,
            metadata=self.metadata,
            intent=self.intent,
            intent_confidence=self.intent_confidence,
            emotion=self.emotion,
            entities=self.entities,
            behavioral_signal=self.behavioral_signal,
            surfaced_task_id=self.surfaced_task_id,
            embedding=tuple(embedding),
            source_working_id=self.source_working_id,
            extra=self.extra,
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def observe_voice(
    user_id: str,
    raw_text: str,
    *,
    session_id: str = "",
    audio_duration_ms: int | None = None,
    transcription_confidence: float | None = None,
    time_of_day: str = "morning",
    day_of_week: str = "monday",
    preceding_episode_id: str | None = None,
    active_task_id: str | None = None,
) -> Observation:
    """Create an Observation from voice input.

    This is the primary entry point for voice blurts. The observation captures
    the raw transcription and voice-specific metadata (duration, STT confidence).
    """
    return Observation(
        user_id=user_id,
        raw_text=raw_text,
        modality=InputModality.VOICE,
        metadata=ObservationMetadata(
            session_id=session_id,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            preceding_episode_id=preceding_episode_id,
            active_task_id=active_task_id,
            audio_duration_ms=audio_duration_ms,
            transcription_confidence=transcription_confidence,
        ),
    )


def observe_text(
    user_id: str,
    raw_text: str,
    *,
    session_id: str = "",
    time_of_day: str = "morning",
    day_of_week: str = "monday",
    preceding_episode_id: str | None = None,
    active_task_id: str | None = None,
) -> Observation:
    """Create an Observation from text input (edits/corrections)."""
    return Observation(
        user_id=user_id,
        raw_text=raw_text,
        modality=InputModality.TEXT,
        metadata=ObservationMetadata(
            session_id=session_id,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            preceding_episode_id=preceding_episode_id,
            active_task_id=active_task_id,
        ),
    )


# ---------------------------------------------------------------------------
# Repository: append observation to episodic store
# ---------------------------------------------------------------------------


class ObservationRepository:
    """Repository that records observations into the episodic memory store.

    Provides the canonical append operation: takes a fully-enriched Observation,
    converts it to an Episode, and appends it to the store. Returns the stored
    Episode for downstream use (e.g., linking to knowledge graph).

    Thread-safe: delegates to the underlying async store.
    """

    def __init__(self, store: EpisodicMemoryStore) -> None:
        self._store = store
        self._observation_count = 0

    @property
    def store(self) -> EpisodicMemoryStore:
        """Access the underlying episodic store."""
        return self._store

    @property
    def observation_count(self) -> int:
        """Total observations recorded through this repository."""
        return self._observation_count

    async def record(self, observation: Observation) -> Episode:
        """Record an observation as an episode in the append-only store.

        This is the single entry point for persisting observations. It:
        1. Converts the Observation to an Episode
        2. Appends to the episodic store (append-only, raises on duplicate)
        3. Returns the stored Episode

        Args:
            observation: A fully-enriched Observation (with intent, emotion, entities).

        Returns:
            The stored Episode.

        Raises:
            ValueError: If an episode with this ID already exists (append-only).
        """
        episode = observation.to_episode()
        stored = await self._store.append(episode)
        self._observation_count += 1
        return stored

    async def record_many(self, observations: list[Observation]) -> list[Episode]:
        """Record multiple observations in order. Stops on first failure.

        Args:
            observations: Ordered list of observations to record.

        Returns:
            List of stored Episodes in the same order.
        """
        episodes = []
        for obs in observations:
            ep = await self.record(obs)
            episodes.append(ep)
        return episodes

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Retrieve a stored episode by ID."""
        return await self._store.get(episode_id)

    async def count_user_episodes(self, user_id: str) -> int:
        """Count total episodes for a user."""
        return await self._store.count(user_id)
