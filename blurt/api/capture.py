"""API routes for blurt capture — the intake endpoint that stores everything.

Every voice utterance and text input flows through these endpoints.
Nothing is filtered, dropped, or discarded. Casual remarks, half-thoughts,
and throwaway comments are all captured as observations.

Endpoints:
- POST /api/v1/blurt          — Submit a raw blurt (voice or text)
- POST /api/v1/blurt/voice    — Submit a voice blurt with audio metadata
- POST /api/v1/blurt/text     — Submit a text blurt (edits/corrections)
- GET  /api/v1/blurt/stats    — Pipeline capture statistics
"""

from __future__ import annotations


from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from blurt.api.episodes import (
    EpisodeResponse,
    _episode_to_response,
    get_store,
)
from blurt.memory.episodic import EpisodicMemoryStore
from blurt.services.acknowledgment import AcknowledgmentService
from blurt.services.capture import BlurtCapturePipeline, CaptureResult

router = APIRouter(prefix="/api/v1/blurt", tags=["capture"])

# ---------------------------------------------------------------------------
# Pipeline singleton (DI-managed)
# ---------------------------------------------------------------------------

_pipeline: BlurtCapturePipeline | None = None
_ack_service: AcknowledgmentService | None = None


def get_pipeline(store: EpisodicMemoryStore = Depends(get_store)) -> BlurtCapturePipeline:
    """DI for the capture pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = BlurtCapturePipeline(store)
    return _pipeline


def get_ack_service() -> AcknowledgmentService:
    """DI for the acknowledgment service."""
    global _ack_service
    if _ack_service is None:
        _ack_service = AcknowledgmentService(history_size=5)
    return _ack_service


def set_pipeline(pipeline: BlurtCapturePipeline) -> None:
    """Override the pipeline (for testing)."""
    global _pipeline
    _pipeline = pipeline


def set_ack_service(service: AcknowledgmentService) -> None:
    """Override the acknowledgment service (for testing)."""
    global _ack_service
    _ack_service = service


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class BlurtRequest(BaseModel):
    """A raw blurt submission — captures everything the user says.

    No filtering, no minimum length, no content requirements.
    Even "hmm" or "nice weather" gets stored.
    """

    user_id: str
    raw_text: str = Field(
        description="The raw text content. Can be anything — casual remarks included."
    )
    modality: str = Field(
        default="voice",
        description="Input modality: 'voice' or 'text'",
    )
    session_id: str = Field(default="", description="Current session ID")
    time_of_day: str = Field(default="morning", description="morning/afternoon/evening/night")
    day_of_week: str = Field(default="monday", description="Day of week")
    preceding_episode_id: str | None = Field(default=None, description="Previous episode in session")
    active_task_id: str | None = Field(default=None, description="Currently active task")


class VoiceBlurtRequest(BaseModel):
    """Voice-specific blurt submission with audio metadata."""

    user_id: str
    raw_text: str = Field(description="Transcribed text from voice input")
    session_id: str = Field(default="", description="Current session ID")
    audio_duration_ms: int | None = Field(default=None, description="Audio clip duration in ms")
    transcription_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="STT confidence score",
    )
    time_of_day: str = Field(default="morning")
    day_of_week: str = Field(default="monday")
    preceding_episode_id: str | None = None
    active_task_id: str | None = None


class TextBlurtRequest(BaseModel):
    """Text blurt submission (for edits/corrections)."""

    user_id: str
    raw_text: str = Field(description="Typed text content")
    session_id: str = Field(default="", description="Current session ID")
    time_of_day: str = Field(default="morning")
    day_of_week: str = Field(default="monday")
    preceding_episode_id: str | None = None
    active_task_id: str | None = None


class AcknowledgmentResponse(BaseModel):
    """Brief verbal acknowledgment returned to the user.

    Designed to be spoken via TTS or shown as a short text confirmation.
    Always brief (1-8 words), never chatty.
    """

    text: str = Field(description="Short acknowledgment text (1-8 words)")
    tone: str = Field(default="calm", description="Emotional tone: warm, calm, energetic, gentle, matter_of_fact")
    is_silent: bool = Field(default=False, description="If true, no verbal ack (e.g. questions get answers)")
    answer: str | None = Field(default=None, description="For questions, the actual answer text")


class BlurtResponse(BaseModel):
    """Response from capturing a blurt. Always indicates successful capture."""

    captured: bool = True  # Always True — we never drop
    episode: EpisodeResponse
    observation_id: str
    intent: str
    intent_confidence: float
    acknowledgment: AcknowledgmentResponse
    entities_extracted: int = 0
    emotion_detected: bool = False
    fully_enriched: bool = False
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    # Casual observation detection — enrichment metadata, never used for filtering
    observation_type: str = "ambiguous"  # casual, substantive, ambiguous
    is_casual: bool = False


class CaptureStatsResponse(BaseModel):
    """Pipeline capture statistics."""

    total_captured: int
    voice_count: int
    text_count: int
    fully_enriched_count: int
    partial_enrichment_count: int
    classification_failures: int
    drop_rate: float  # Should always be 0.0
    enrichment_success_rate: float
    avg_latency_ms: float
    intent_distribution: dict[str, int]
    casual_capture_count: int


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _capture_result_to_response(
    result: CaptureResult,
    ack_service: AcknowledgmentService | None = None,
) -> BlurtResponse:
    from blurt.classification.models import ClassificationResult, ClassificationStatus, IntentScore
    from blurt.models.intents import BlurtIntent
    from blurt.services.acknowledgment import generate_acknowledgment

    # Build a ClassificationResult from the episode data for the ack service
    try:
        intent = BlurtIntent(result.episode.intent)
    except ValueError:
        intent = BlurtIntent.JOURNAL

    status = (
        ClassificationStatus.CONFIDENT
        if result.classification_applied
        else ClassificationStatus.ERROR
    )
    classification = ClassificationResult(
        input_text=result.episode.raw_text,
        primary_intent=intent,
        confidence=result.episode.intent_confidence,
        status=status,
        all_scores=[IntentScore(intent=intent, confidence=result.episode.intent_confidence)],
        model_used="pipeline",
    )

    # Generate brief, natural acknowledgment
    if ack_service is not None:
        ack = ack_service.acknowledge(classification)
    else:
        ack = generate_acknowledgment(classification)

    ack_response = AcknowledgmentResponse(
        text=ack.text,
        tone=ack.tone.value,
        is_silent=ack.is_silent,
        answer=ack.answer,
    )

    return BlurtResponse(
        captured=True,
        episode=_episode_to_response(result.episode),
        observation_id=result.observation_id,
        intent=result.episode.intent,
        intent_confidence=result.episode.intent_confidence,
        acknowledgment=ack_response,
        entities_extracted=result.entities_extracted,
        emotion_detected=result.emotion_detected,
        fully_enriched=result.fully_enriched,
        warnings=result.warnings,
        latency_ms=round(result.latency_ms, 2),
        observation_type=result.observation_type.value,
        is_casual=result.is_casual,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=BlurtResponse, status_code=201)
async def capture_blurt(
    request: BlurtRequest,
    pipeline: BlurtCapturePipeline = Depends(get_pipeline),
    ack_service: AcknowledgmentService = Depends(get_ack_service),
) -> BlurtResponse:
    """Capture a raw blurt — everything said is stored, including casual remarks.

    This is the primary intake endpoint. No input is too trivial:
    - "I need to call the dentist" → stored as task
    - "huh, interesting" → stored as journal
    - "nice weather today" → stored as journal
    - "oh right, the meeting" → stored as event/reminder
    - "" (empty) → still stored as empty observation

    The pipeline classifies, extracts, detects, and stores silently.
    The user never sees or interacts with the classification.
    """
    if request.modality == "voice":
        result = await pipeline.capture_voice(
            user_id=request.user_id,
            raw_text=request.raw_text,
            session_id=request.session_id,
            time_of_day=request.time_of_day,
            day_of_week=request.day_of_week,
            preceding_episode_id=request.preceding_episode_id,
            active_task_id=request.active_task_id,
        )
    else:
        result = await pipeline.capture_text(
            user_id=request.user_id,
            raw_text=request.raw_text,
            session_id=request.session_id,
            time_of_day=request.time_of_day,
            day_of_week=request.day_of_week,
            preceding_episode_id=request.preceding_episode_id,
            active_task_id=request.active_task_id,
        )

    return _capture_result_to_response(result, ack_service)


@router.post("/voice", response_model=BlurtResponse, status_code=201)
async def capture_voice_blurt(
    request: VoiceBlurtRequest,
    pipeline: BlurtCapturePipeline = Depends(get_pipeline),
    ack_service: AcknowledgmentService = Depends(get_ack_service),
) -> BlurtResponse:
    """Capture a voice blurt with audio-specific metadata.

    Every utterance is captured — casual remarks, filler words, and
    half-thoughts are all valid observations.
    """
    result = await pipeline.capture_voice(
        user_id=request.user_id,
        raw_text=request.raw_text,
        session_id=request.session_id,
        audio_duration_ms=request.audio_duration_ms,
        transcription_confidence=request.transcription_confidence,
        time_of_day=request.time_of_day,
        day_of_week=request.day_of_week,
        preceding_episode_id=request.preceding_episode_id,
        active_task_id=request.active_task_id,
    )
    return _capture_result_to_response(result, ack_service)


@router.post("/text", response_model=BlurtResponse, status_code=201)
async def capture_text_blurt(
    request: TextBlurtRequest,
    pipeline: BlurtCapturePipeline = Depends(get_pipeline),
    ack_service: AcknowledgmentService = Depends(get_ack_service),
) -> BlurtResponse:
    """Capture a text blurt (for edits/corrections).

    Text input follows the same zero-drop pipeline as voice.
    """
    result = await pipeline.capture_text(
        user_id=request.user_id,
        raw_text=request.raw_text,
        session_id=request.session_id,
        time_of_day=request.time_of_day,
        day_of_week=request.day_of_week,
        preceding_episode_id=request.preceding_episode_id,
        active_task_id=request.active_task_id,
    )
    return _capture_result_to_response(result, ack_service)


@router.get("/stats", response_model=CaptureStatsResponse)
async def get_capture_stats(
    pipeline: BlurtCapturePipeline = Depends(get_pipeline),
) -> CaptureStatsResponse:
    """Get capture pipeline statistics.

    The drop_rate should always be 0.0. If it's not, something is broken.
    """
    stats = pipeline.stats
    return CaptureStatsResponse(
        total_captured=stats.total_captured,
        voice_count=stats.voice_count,
        text_count=stats.text_count,
        fully_enriched_count=stats.fully_enriched_count,
        partial_enrichment_count=stats.partial_enrichment_count,
        classification_failures=stats.classification_failures,
        drop_rate=stats.drop_rate,
        enrichment_success_rate=round(stats.enrichment_success_rate, 4),
        avg_latency_ms=round(stats.avg_latency_ms, 2),
        intent_distribution=stats.intent_distribution,
        casual_capture_count=stats.casual_capture_count,
    )
