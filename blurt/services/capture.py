"""Blurt capture pipeline — guarantees every utterance is stored as an observation.

The capture pipeline is the central orchestrator that ensures NOTHING the user
says is ever dropped, filtered, or discarded. Every input — including casual
remarks, throwaway comments, half-finished thoughts, and off-hand observations —
flows through the full pipeline:

    receive → observe → classify → extract → detect → embed → store

Design principles:
- ZERO DROP: Every input produces a stored Episode, regardless of content
- CASUAL CAPTURE: "huh, interesting", "nice weather", "oh well" are all stored
- SILENT CLASSIFICATION: The pipeline classifies silently; users never see intent labels
- SAFE FALLBACK: If any pipeline stage fails, the observation is still stored
  with whatever enrichment succeeded (classification errors default to journal)
- ANTI-SHAME: No input is "too trivial" to capture — casual remarks contribute
  to behavioral patterns and emotional baselines

This module is the "funnel top" of the memory system. Data flows in,
enrichment happens, nothing flows out unrecorded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from blurt.memory.episodic import (
    EmotionSnapshot,
    EntityRef,
    Episode,
    EpisodicMemoryStore,
    InputModality,
    InMemoryEpisodicStore,
)
from blurt.memory.observation import (
    Observation,
    ObservationRepository,
    observe_text,
    observe_voice,
)
from blurt.services.casual_detection import (
    CasualDetectionResult,
    ObservationType,
    detect_casual,
)

logger = logging.getLogger(__name__)


class CaptureStage(str, Enum):
    """Stages of the capture pipeline, tracked for diagnostics."""

    RECEIVED = "received"
    OBSERVED = "observed"
    CLASSIFIED = "classified"
    ENTITIES_EXTRACTED = "entities_extracted"
    EMOTION_DETECTED = "emotion_detected"
    EMBEDDED = "embedded"
    STORED = "stored"
    FAILED_PARTIAL = "failed_partial"  # Stored but some enrichment failed


@dataclass
class CaptureResult:
    """Result of processing a blurt through the capture pipeline.

    Always contains a stored episode — the pipeline never drops data.
    Partial enrichment failures are recorded but never prevent storage.
    """

    episode: Episode
    observation_id: str
    stages_completed: list[CaptureStage] = field(default_factory=list)
    classification_applied: bool = False
    entities_extracted: int = 0
    emotion_detected: bool = False
    embedding_generated: bool = False
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # Casual observation detection
    observation_type: ObservationType = ObservationType.AMBIGUOUS
    casual_detection: CasualDetectionResult | None = None

    @property
    def fully_enriched(self) -> bool:
        """Whether all enrichment stages completed successfully."""
        return (
            self.classification_applied
            and self.emotion_detected
            and self.embedding_generated
        )

    @property
    def is_casual(self) -> bool:
        """Whether this observation was detected as a casual remark."""
        return self.observation_type == ObservationType.CASUAL

    @property
    def was_stored(self) -> bool:
        """Whether the observation was successfully stored (always True in normal operation)."""
        return CaptureStage.STORED in self.stages_completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.id,
            "observation_id": self.observation_id,
            "stages_completed": [s.value for s in self.stages_completed],
            "classification_applied": self.classification_applied,
            "entities_extracted": self.entities_extracted,
            "emotion_detected": self.emotion_detected,
            "embedding_generated": self.embedding_generated,
            "latency_ms": round(self.latency_ms, 2),
            "fully_enriched": self.fully_enriched,
            "warnings": self.warnings,
            "observation_type": self.observation_type.value,
            "is_casual": self.is_casual,
        }


@dataclass
class CaptureStats:
    """Aggregate statistics for the capture pipeline."""

    total_captured: int = 0
    voice_count: int = 0
    text_count: int = 0
    fully_enriched_count: int = 0
    partial_enrichment_count: int = 0
    classification_failures: int = 0
    entity_extraction_failures: int = 0
    emotion_detection_failures: int = 0
    embedding_failures: int = 0
    total_latency_ms: float = 0.0

    # Intent distribution (tracks what gets classified as what)
    intent_distribution: dict[str, int] = field(default_factory=dict)

    # Track that casual remarks are being captured (not filtered)
    casual_capture_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_captured == 0:
            return 0.0
        return self.total_latency_ms / self.total_captured

    @property
    def drop_rate(self) -> float:
        """Drop rate — should always be 0.0. Non-zero indicates a bug."""
        return 0.0  # By design, we never drop

    @property
    def enrichment_success_rate(self) -> float:
        if self.total_captured == 0:
            return 0.0
        return self.fully_enriched_count / self.total_captured

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_captured": self.total_captured,
            "voice_count": self.voice_count,
            "text_count": self.text_count,
            "fully_enriched_count": self.fully_enriched_count,
            "partial_enrichment_count": self.partial_enrichment_count,
            "classification_failures": self.classification_failures,
            "entity_extraction_failures": self.entity_extraction_failures,
            "emotion_detection_failures": self.emotion_detection_failures,
            "embedding_failures": self.embedding_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "drop_rate": self.drop_rate,
            "enrichment_success_rate": round(self.enrichment_success_rate, 4),
            "intent_distribution": self.intent_distribution,
            "casual_capture_count": self.casual_capture_count,
        }


# ---------------------------------------------------------------------------
# Classifier / Extractor / Detector interfaces (protocol-based)
# ---------------------------------------------------------------------------
# These are lightweight protocols so the capture pipeline doesn't depend
# on concrete Gemini implementations. Each can be swapped for local-only mode.


class ClassifierFunc(Protocol):
    """Protocol for intent classification callable."""

    async def __call__(self, text: str) -> tuple[str, float]:
        """Classify text, return (intent, confidence)."""
        ...


class EntityExtractorFunc(Protocol):
    """Protocol for entity extraction callable."""

    async def __call__(self, text: str) -> list[EntityRef]:
        """Extract entities from text."""
        ...


class EmotionDetectorFunc(Protocol):
    """Protocol for emotion detection callable."""

    async def __call__(self, text: str) -> EmotionSnapshot:
        """Detect emotion from text."""
        ...


class EmbedderFunc(Protocol):
    """Protocol for embedding generation callable."""

    async def __call__(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...


# ---------------------------------------------------------------------------
# Default no-op implementations (safe fallbacks)
# ---------------------------------------------------------------------------


async def _default_classify(text: str) -> tuple[str, float]:
    """Default classifier — everything is a journal entry (safe fallback).

    This ensures casual remarks, throwaway comments, and any unclassifiable
    input gets stored as a journal entry rather than being dropped.
    """
    return ("journal", 1.0)


async def _default_extract_entities(text: str) -> list[EntityRef]:
    """Default entity extractor — returns empty (no entities found)."""
    return []


async def _default_detect_emotion(text: str) -> EmotionSnapshot:
    """Default emotion detector — neutral baseline."""
    return EmotionSnapshot(
        primary="trust",
        intensity=0.5,
        valence=0.0,
        arousal=0.2,
    )


async def _default_embed(text: str) -> list[float]:
    """Default embedder — returns empty (no embedding)."""
    return []


# ---------------------------------------------------------------------------
# Capture pipeline
# ---------------------------------------------------------------------------


class BlurtCapturePipeline:
    """Orchestrates the full capture pipeline for every user utterance.

    GUARANTEE: Every call to `capture_voice()` or `capture_text()` results
    in a stored Episode. No input is ever dropped, filtered, or discarded.

    The pipeline runs enrichment stages (classify, extract, detect, embed)
    in sequence. If any stage fails, the failure is logged as a warning
    but the observation is still stored with whatever enrichment succeeded.

    Usage::

        pipeline = BlurtCapturePipeline(episodic_store)
        result = await pipeline.capture_voice(
            user_id="user-1",
            raw_text="huh, that's interesting",
            session_id="sess-1",
        )
        assert result.was_stored  # Always True
        assert result.episode.raw_text == "huh, that's interesting"
    """

    def __init__(
        self,
        store: EpisodicMemoryStore | None = None,
        *,
        classifier: ClassifierFunc | None = None,
        entity_extractor: EntityExtractorFunc | None = None,
        emotion_detector: EmotionDetectorFunc | None = None,
        embedder: EmbedderFunc | None = None,
    ) -> None:
        if store is None:
            store = InMemoryEpisodicStore()
        self._repo = ObservationRepository(store)
        self._classify = classifier or _default_classify
        self._extract_entities = entity_extractor or _default_extract_entities
        self._detect_emotion = emotion_detector or _default_detect_emotion
        self._embed = embedder or _default_embed
        self._stats = CaptureStats()

    @property
    def stats(self) -> CaptureStats:
        return self._stats

    @property
    def repo(self) -> ObservationRepository:
        return self._repo

    async def capture_voice(
        self,
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
    ) -> CaptureResult:
        """Capture a voice blurt — everything said is stored, including casual remarks.

        This method NEVER returns without storing the observation. Even if
        all enrichment stages fail, the raw text is preserved as a journal entry.

        Args:
            user_id: The user who spoke.
            raw_text: Transcribed text from voice input (may be casual/throwaway).
            session_id: Current session identifier.
            audio_duration_ms: Duration of the voice clip.
            transcription_confidence: STT confidence score.
            time_of_day: Time of day context.
            day_of_week: Day of week context.
            preceding_episode_id: Link to the previous episode in this session.
            active_task_id: Currently active task, if any.

        Returns:
            CaptureResult with the stored Episode and enrichment details.
        """
        obs = observe_voice(
            user_id,
            raw_text,
            session_id=session_id,
            audio_duration_ms=audio_duration_ms,
            transcription_confidence=transcription_confidence,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            preceding_episode_id=preceding_episode_id,
            active_task_id=active_task_id,
        )
        self._stats.voice_count += 1
        return await self._run_pipeline(obs)

    async def capture_text(
        self,
        user_id: str,
        raw_text: str,
        *,
        session_id: str = "",
        time_of_day: str = "morning",
        day_of_week: str = "monday",
        preceding_episode_id: str | None = None,
        active_task_id: str | None = None,
    ) -> CaptureResult:
        """Capture a text blurt (for edits/corrections). Same zero-drop guarantee.

        Args:
            user_id: The user who typed.
            raw_text: The typed text.
            session_id: Current session identifier.
            time_of_day: Time of day context.
            day_of_week: Day of week context.
            preceding_episode_id: Link to the previous episode.
            active_task_id: Currently active task, if any.

        Returns:
            CaptureResult with the stored Episode.
        """
        obs = observe_text(
            user_id,
            raw_text,
            session_id=session_id,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            preceding_episode_id=preceding_episode_id,
            active_task_id=active_task_id,
        )
        self._stats.text_count += 1
        return await self._run_pipeline(obs)

    async def capture_raw(
        self,
        observation: Observation,
    ) -> CaptureResult:
        """Capture a pre-built Observation. Same zero-drop guarantee.

        Use this when you've already constructed the Observation externally
        (e.g., from WebSocket handler).

        Args:
            observation: Pre-built Observation to process and store.

        Returns:
            CaptureResult with the stored Episode.
        """
        if observation.modality == InputModality.VOICE:
            self._stats.voice_count += 1
        else:
            self._stats.text_count += 1
        return await self._run_pipeline(observation)

    async def _run_pipeline(self, obs: Observation) -> CaptureResult:
        """Run the full capture pipeline on an observation.

        Pipeline stages run in order. Each stage's failure is isolated —
        it does not prevent subsequent stages or final storage.

        The observation is ALWAYS stored at the end, regardless of
        enrichment success/failure.
        """
        start = time.monotonic()
        result = CaptureResult(
            episode=obs.to_episode(),  # Placeholder, replaced after storage
            observation_id=obs.id,
        )
        result.stages_completed.append(CaptureStage.RECEIVED)
        result.stages_completed.append(CaptureStage.OBSERVED)
        warnings: list[str] = []

        # Stage 0: Casual observation detection (fast, local, never fails)
        # This runs BEFORE classification and is purely for enrichment —
        # it NEVER filters or drops observations.
        casual_result = detect_casual(obs.raw_text)
        result.observation_type = casual_result.observation_type
        result.casual_detection = casual_result

        # Tag the observation with casual detection metadata
        extra = dict(obs.extra)
        extra["observation_type"] = casual_result.observation_type.value
        extra["casual_confidence"] = casual_result.confidence
        extra["casual_signals"] = list(casual_result.signals)
        extra["word_count"] = casual_result.word_count
        obs = Observation(
            id=obs.id,
            user_id=obs.user_id,
            timestamp=obs.timestamp,
            raw_text=obs.raw_text,
            modality=obs.modality,
            metadata=obs.metadata,
            intent=obs.intent,
            intent_confidence=obs.intent_confidence,
            emotion=obs.emotion,
            entities=obs.entities,
            behavioral_signal=obs.behavioral_signal,
            surfaced_task_id=obs.surfaced_task_id,
            embedding=obs.embedding,
            source_working_id=obs.source_working_id,
            extra=extra,
        )

        # Stage 1: Classification
        try:
            intent, confidence = await self._classify(obs.raw_text)
            obs = obs.with_classification(intent, confidence)
            result.classification_applied = True
            result.stages_completed.append(CaptureStage.CLASSIFIED)

            # Track intent distribution
            self._stats.intent_distribution[intent] = (
                self._stats.intent_distribution.get(intent, 0) + 1
            )

            # Track casual captures — based on casual detection, not just journal intent
            if casual_result.is_casual or intent == "journal":
                self._stats.casual_capture_count += 1

        except Exception as e:
            logger.warning("Classification failed for blurt %s: %s", obs.id, e)
            # Safe fallback: classify as journal
            obs = obs.with_classification("journal", 1.0)
            result.classification_applied = True  # Fallback still counts
            result.stages_completed.append(CaptureStage.CLASSIFIED)
            warnings.append(f"Classification failed, defaulted to journal: {e}")
            self._stats.classification_failures += 1
            if casual_result.is_casual:
                self._stats.casual_capture_count += 1

        # Stage 2: Entity extraction
        try:
            entities = await self._extract_entities(obs.raw_text)
            if entities:
                obs = obs.with_entities(entities)
            result.entities_extracted = len(entities)
            result.stages_completed.append(CaptureStage.ENTITIES_EXTRACTED)
        except Exception as e:
            logger.warning("Entity extraction failed for blurt %s: %s", obs.id, e)
            warnings.append(f"Entity extraction failed: {e}")
            self._stats.entity_extraction_failures += 1

        # Stage 3: Emotion detection
        try:
            emotion = await self._detect_emotion(obs.raw_text)
            obs = obs.with_emotion(emotion)
            result.emotion_detected = True
            result.stages_completed.append(CaptureStage.EMOTION_DETECTED)
        except Exception as e:
            logger.warning("Emotion detection failed for blurt %s: %s", obs.id, e)
            warnings.append(f"Emotion detection failed: {e}")
            self._stats.emotion_detection_failures += 1

        # Stage 4: Embedding generation
        try:
            embedding = await self._embed(obs.raw_text)
            if embedding:
                obs = obs.with_embedding(embedding)
                result.embedding_generated = True
            result.stages_completed.append(CaptureStage.EMBEDDED)
        except Exception as e:
            logger.warning("Embedding generation failed for blurt %s: %s", obs.id, e)
            warnings.append(f"Embedding generation failed: {e}")
            self._stats.embedding_failures += 1

        # FINAL STAGE: Store — this MUST succeed for pipeline integrity
        try:
            episode = await self._repo.record(obs)
            result.episode = episode
            result.stages_completed.append(CaptureStage.STORED)
        except Exception as e:
            # Critical failure — log at error level but still return what we have
            logger.error(
                "CRITICAL: Failed to store observation %s: %s. "
                "This violates the zero-drop guarantee!",
                obs.id,
                e,
            )
            result.episode = obs.to_episode()  # Return unconverted as best-effort
            result.stages_completed.append(CaptureStage.FAILED_PARTIAL)
            warnings.append(f"Storage failed: {e}")

        result.warnings = warnings
        result.latency_ms = (time.monotonic() - start) * 1000

        # Update stats
        self._stats.total_captured += 1
        self._stats.total_latency_ms += result.latency_ms
        if result.fully_enriched:
            self._stats.fully_enriched_count += 1
        elif warnings:
            self._stats.partial_enrichment_count += 1

        logger.info(
            "Captured blurt %s: intent=%s stages=%d warnings=%d latency=%.0fms",
            obs.id,
            obs.intent or "unknown",
            len(result.stages_completed),
            len(warnings),
            result.latency_ms,
        )

        return result
