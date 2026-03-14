"""Full-duplex audio processing pipeline.

Orchestrates the complete flow:
  raw audio → chunking → Gemini processing → classification → structured response

This is the core pipeline that every voice blurt flows through.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from blurt.audio.models import (
    AudioChunk,
    AudioFormat,
    AudioProcessingResult,
    AudioStreamSession,
    StreamState,
)
from blurt.gemini.audio_client import GeminiAudioClient


@dataclass
class PipelineResult:
    """Complete result of processing a blurt through the audio pipeline.

    Contains everything downstream systems need: transcript, classification,
    entities, emotion, audio analysis, and processing metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""

    # Audio processing
    transcript: str = ""
    audio_duration_ms: int = 0
    audio_result: AudioProcessingResult | None = None

    # Classification
    intent: str = ""
    confidence: float = 0.0
    entities: list[dict[str, Any]] = field(default_factory=list)
    emotion: dict[str, Any] = field(default_factory=dict)
    acknowledgment: str = ""

    # Pipeline metadata
    pipeline_complete: bool = False
    processing_time_ms: int = 0
    model_used: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AudioPipeline:
    """Orchestrates full-duplex audio processing.

    The pipeline handles:
    1. Audio chunking and accumulation
    2. Gemini multimodal processing (audio → transcript + understanding)
    3. Classification (intent, entities, emotion)
    4. Response generation

    Designed for full-duplex: audio can stream in while previous
    utterances are still being processed.
    """

    def __init__(self, gemini_client: GeminiAudioClient):
        self._gemini = gemini_client
        self._active_sessions: dict[str, AudioStreamSession] = {}

    async def process_audio(
        self,
        audio_data: bytes,
        audio_format: AudioFormat,
        *,
        user_id: str = "",
    ) -> PipelineResult:
        """Process a complete audio clip through the full pipeline.

        This is the single-shot path: complete audio in, complete result out.
        Used when the client sends a committed utterance.
        """
        start_time = time.monotonic()
        result = PipelineResult(user_id=user_id)

        try:
            # Full pipeline through Gemini
            pipeline_output = await self._gemini.process_audio_full_pipeline(
                audio_data, audio_format
            )

            result.transcript = pipeline_output.get("transcript", "")
            result.pipeline_complete = pipeline_output.get("pipeline_complete", False)

            # Extract audio result
            audio_result_data = pipeline_output.get("audio_result")
            if audio_result_data:
                result.audio_result = AudioProcessingResult(**audio_result_data)
                result.audio_duration_ms = result.audio_result.audio_duration_ms
                result.model_used = result.audio_result.model_used

            # Extract classification
            classification = pipeline_output.get("classification")
            if classification:
                result.intent = classification.get("intent", "")
                result.confidence = classification.get("confidence", 0.0)
                result.entities = classification.get("entities", [])
                result.emotion = classification.get("emotion", {})
                result.acknowledgment = classification.get("acknowledgment", "")

        except Exception as e:
            result.error = str(e)
            result.pipeline_complete = False

        result.processing_time_ms = int((time.monotonic() - start_time) * 1000)
        return result

    async def process_audio_stream(
        self,
        chunks: AsyncIterator[AudioChunk],
        *,
        user_id: str = "",
        session_id: str | None = None,
    ) -> PipelineResult:
        """Process a stream of audio chunks through the full pipeline.

        This is the streaming path for full-duplex operation.
        Audio chunks arrive incrementally and are processed together.
        """
        start_time = time.monotonic()
        sid = session_id or str(uuid.uuid4())

        session = AudioStreamSession(
            session_id=sid,
            user_id=user_id,
            state=StreamState.RECEIVING,
            started_at=datetime.now(timezone.utc),
        )
        self._active_sessions[sid] = session

        result = PipelineResult(user_id=user_id)

        try:
            # Process through Gemini streaming
            audio_result = await self._gemini.process_audio_stream(chunks)

            result.transcript = audio_result.transcript
            result.audio_result = audio_result
            result.audio_duration_ms = audio_result.audio_duration_ms
            result.model_used = audio_result.model_used

            if not audio_result.is_empty:
                # Classify the transcript
                classification = await self._gemini.classify_transcript(
                    audio_result.transcript
                )
                result.intent = classification.get("intent", "")
                result.confidence = classification.get("confidence", 0.0)
                result.entities = classification.get("entities", [])
                result.emotion = classification.get("emotion", {})
                result.acknowledgment = classification.get("acknowledgment", "")

            result.pipeline_complete = True
            session.state = StreamState.COMPLETE

        except Exception as e:
            result.error = str(e)
            result.pipeline_complete = False
            session.state = StreamState.ERROR

        finally:
            session.completed_at = datetime.now(timezone.utc)
            result.processing_time_ms = int((time.monotonic() - start_time) * 1000)
            self._active_sessions.pop(sid, None)

        return result

    def create_session(self, user_id: str, audio_format: AudioFormat | None = None) -> str:
        """Create a new audio streaming session. Returns session_id."""
        session = AudioStreamSession(
            user_id=user_id,
            format=audio_format or AudioFormat(),
        )
        self._active_sessions[session.session_id] = session
        return session.session_id

    def get_session(self, session_id: str) -> AudioStreamSession | None:
        """Get an active session by ID."""
        return self._active_sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        """End and clean up a streaming session."""
        self._active_sessions.pop(session_id, None)

    @property
    def active_session_count(self) -> int:
        return len(self._active_sessions)
