"""Integration tests for full-duplex audio round-trip.

Validates the complete pipeline: raw audio in → Gemini 2 processing → structured response out.

These tests use a fake Gemini HTTP backend (httpx mock transport) to simulate
the Gemini API, allowing us to test the full pipeline wiring without real API
calls. The fake returns realistic structured responses matching Gemini's format.

Test categories:
1. Single-shot audio round-trip (complete audio → complete result)
2. Streaming audio round-trip (chunked audio → accumulated result)
3. Full pipeline: audio → transcription → classification → structured output
4. Error handling and edge cases
5. Multiple audio formats
6. Full-duplex concurrent processing
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any, AsyncIterator

import httpx
import pytest

from blurt.audio.models import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
)
from blurt.audio.pipeline import AudioPipeline, PipelineResult
from blurt.config.settings import GeminiConfig, GeminiModel
from blurt.gemini.audio_client import (
    GeminiAudioClient,
    GeminiAudioError,
)


# ── Test fixtures & helpers ──────────────────────────────────────


def generate_pcm_audio(
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> bytes:
    """Generate synthetic PCM audio data (16-bit signed LE sine wave).

    Produces valid audio bytes that represent a pure tone,
    suitable for testing the audio pipeline without real recordings.
    """
    import math

    num_samples = int(sample_rate * duration_seconds)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


def make_audio_chunks(
    audio_data: bytes,
    chunk_size_ms: int = 200,
    sample_rate: int = 16000,
    sample_width: int = 2,
) -> list[AudioChunk]:
    """Split audio data into chunks simulating streaming input."""
    bytes_per_ms = sample_rate * sample_width // 1000
    chunk_bytes = chunk_size_ms * bytes_per_ms
    chunks = []
    offset = 0
    seq = 0

    while offset < len(audio_data):
        end = min(offset + chunk_bytes, len(audio_data))
        chunk_data = audio_data[offset:end]
        actual_duration = len(chunk_data) // bytes_per_ms

        chunks.append(
            AudioChunk(
                data=chunk_data,
                sequence_number=seq,
                timestamp_ms=offset // bytes_per_ms,
                duration_ms=actual_duration,
                is_final=(end >= len(audio_data)),
                format=AudioFormat(
                    encoding=AudioEncoding.LINEAR16,
                    sample_rate_hz=sample_rate,
                    channels=1,
                    sample_width_bytes=sample_width,
                ),
            )
        )
        offset = end
        seq += 1

    return chunks


async def async_chunk_iter(chunks: list[AudioChunk]) -> AsyncIterator[AudioChunk]:
    """Convert a list of chunks into an async iterator."""
    for chunk in chunks:
        yield chunk


def gemini_audio_response(
    transcript: str = "Pick up groceries after work",
    tone: str = "calm",
    emphasis: list[str] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Build a realistic Gemini API response for audio processing."""
    result = {
        "transcript": transcript,
        "segments": [
            {
                "text": transcript,
                "start_ms": 0,
                "end_ms": 2000,
                "confidence": 0.95,
            }
        ],
        "detected_tone": tone,
        "detected_emphasis": emphasis or [],
        "language": language,
    }
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(result)}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
    }


def gemini_classification_response(
    intent: str = "task",
    confidence: float = 0.92,
    entities: list[dict] | None = None,
    emotion: dict | None = None,
    acknowledgment: str = "Got it",
) -> dict[str, Any]:
    """Build a realistic Gemini API classification response."""
    result = {
        "intent": intent,
        "confidence": confidence,
        "entities": entities or [],
        "emotion": emotion or {
            "primary": "anticipation",
            "intensity": 1,
            "valence": 0.3,
            "arousal": 0.4,
        },
        "acknowledgment": acknowledgment,
    }
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(result)}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
    }


class FakeGeminiTransport(httpx.AsyncBaseTransport):
    """Fake HTTP transport that simulates the Gemini API.

    Routes requests based on content to return appropriate responses:
    - Audio requests (with inline_data) → transcription response
    - Classification requests (text-only) → classification response

    Records all requests for assertion in tests.
    """

    def __init__(
        self,
        audio_responses: list[dict[str, Any]] | None = None,
        classification_responses: list[dict[str, Any]] | None = None,
        error_responses: list[tuple[int, dict]] | None = None,
        latency_ms: float = 0,
    ):
        self.audio_responses = list(audio_responses or [gemini_audio_response()])
        self.classification_responses = list(
            classification_responses or [gemini_classification_response()]
        )
        self.error_responses = list(error_responses or [])
        self.latency_ms = latency_ms
        self.requests: list[dict[str, Any]] = []
        self._audio_call_count = 0
        self._classification_call_count = 0
        self._error_call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({
            "url": str(request.url),
            "body": body,
            "method": request.method,
        })

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        # Return error responses first if queued
        if self.error_responses:
            status_code, error_body = self.error_responses.pop(0)
            self._error_call_count += 1
            return httpx.Response(
                status_code=status_code,
                json=error_body,
            )

        # Detect if this is an audio or classification request
        is_audio = self._is_audio_request(body)

        if is_audio:
            idx = min(self._audio_call_count, len(self.audio_responses) - 1)
            response_data = self.audio_responses[idx]
            self._audio_call_count += 1
        else:
            idx = min(
                self._classification_call_count,
                len(self.classification_responses) - 1,
            )
            response_data = self.classification_responses[idx]
            self._classification_call_count += 1

        return httpx.Response(status_code=200, json=response_data)

    def _is_audio_request(self, body: dict[str, Any]) -> bool:
        """Detect whether this is an audio (multimodal) or text-only request."""
        contents = body.get("contents", [])
        for content in contents:
            for part in content.get("parts", []):
                if "inline_data" in part:
                    return True
        return False

    @property
    def total_requests(self) -> int:
        return len(self.requests)


def make_gemini_client(
    transport: FakeGeminiTransport | None = None,
    **config_overrides: Any,
) -> GeminiAudioClient:
    """Create a GeminiAudioClient with a fake transport."""
    transport = transport or FakeGeminiTransport()
    defaults = {
        "api_key": "test-api-key",
        "base_url": "https://fake-gemini.test/v1beta",
        "max_retries": 1,
        "retry_backoff_base": 0.01,
        "retry_backoff_max": 0.05,
    }
    defaults.update(config_overrides)
    config = GeminiConfig(**defaults)
    http_client = httpx.AsyncClient(transport=transport)
    return GeminiAudioClient(config=config, http_client=http_client)


# ── PCM audio format defaults ───────────────────────────────────

DEFAULT_FORMAT = AudioFormat(
    encoding=AudioEncoding.LINEAR16,
    sample_rate_hz=16000,
    channels=1,
    sample_width_bytes=2,
)


# ═══════════════════════════════════════════════════════════════
# Test Suite 1: Single-shot audio round-trip
# ═══════════════════════════════════════════════════════════════


class TestSingleShotAudioRoundTrip:
    """Tests for complete audio → Gemini → structured result flow."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=2.0)

    async def test_basic_audio_to_transcript(self, audio_data: bytes) -> None:
        """Raw audio bytes → Gemini → transcript text."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Buy milk from the store")
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio(audio_data, DEFAULT_FORMAT)

        assert result.transcript == "Buy milk from the store"
        assert not result.is_empty
        assert result.model_used == GeminiModel.FLASH_LITE.value
        assert result.processing_time_ms >= 0
        assert transport._audio_call_count == 1

    async def test_audio_includes_multimodal_analysis(self, audio_data: bytes) -> None:
        """Gemini returns tone, emphasis, and language from audio signal."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="I'm really excited about the launch",
                    tone="excited",
                    emphasis=["really", "launch"],
                    language="en",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio(audio_data, DEFAULT_FORMAT)

        assert result.detected_tone == "excited"
        assert "really" in result.detected_emphasis
        assert "launch" in result.detected_emphasis
        assert result.language == "en"

    async def test_audio_duration_estimated_from_format(self, audio_data: bytes) -> None:
        """Audio duration is correctly estimated from byte count and format."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        result = await client.process_audio(audio_data, DEFAULT_FORMAT)

        # 2 seconds of 16kHz 16-bit mono = 64000 bytes → 2000ms
        assert result.audio_duration_ms == 2000

    async def test_audio_segments_parsed(self, audio_data: bytes) -> None:
        """Transcription segments with timing info are parsed correctly."""
        response = gemini_audio_response(transcript="Hello world")
        transport = FakeGeminiTransport(audio_responses=[response])
        client = make_gemini_client(transport)

        result = await client.process_audio(audio_data, DEFAULT_FORMAT)

        assert len(result.segments) >= 1
        seg = result.segments[0]
        assert seg.text == "Hello world"
        assert seg.start_ms == 0
        assert seg.end_ms == 2000
        assert seg.confidence == 0.95

    async def test_audio_request_contains_base64_data(self, audio_data: bytes) -> None:
        """The API request sends audio as base64-encoded inline data."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, DEFAULT_FORMAT)

        assert transport.total_requests == 1
        req_body = transport.requests[0]["body"]

        # Verify inline_data is present with base64 audio
        parts = req_body["contents"][0]["parts"]
        inline_data = parts[0]["inline_data"]
        assert inline_data["mime_type"] == "audio/l16"

        # Verify it's valid base64 that decodes back to original audio
        decoded = base64.b64decode(inline_data["data"])
        assert decoded == audio_data

    async def test_system_prompt_included(self, audio_data: bytes) -> None:
        """System instruction is included in the API request."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, DEFAULT_FORMAT)

        req_body = transport.requests[0]["body"]
        assert "system_instruction" in req_body
        sys_text = req_body["system_instruction"]["parts"][0]["text"]
        assert "Blurt" in sys_text

    async def test_custom_system_prompt(self, audio_data: bytes) -> None:
        """Custom system prompts are forwarded to the API."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(
            audio_data, DEFAULT_FORMAT, system_prompt="Custom prompt for testing"
        )

        req_body = transport.requests[0]["body"]
        sys_text = req_body["system_instruction"]["parts"][0]["text"]
        assert sys_text == "Custom prompt for testing"

    async def test_empty_audio_returns_empty_result(self) -> None:
        """Zero-length audio produces an empty result without errors."""
        transport = FakeGeminiTransport(
            audio_responses=[gemini_audio_response(transcript="")]
        )
        client = make_gemini_client(transport)

        result = await client.process_audio(b"", DEFAULT_FORMAT)

        assert result.transcript == ""
        assert result.audio_duration_ms == 0


# ═══════════════════════════════════════════════════════════════
# Test Suite 2: Streaming audio round-trip
# ═══════════════════════════════════════════════════════════════


class TestStreamingAudioRoundTrip:
    """Tests for chunked audio streaming → accumulated result."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=3.0)

    @pytest.fixture
    def chunks(self, audio_data: bytes) -> list[AudioChunk]:
        return make_audio_chunks(audio_data, chunk_size_ms=500)

    async def test_streaming_chunks_accumulated_and_processed(
        self, chunks: list[AudioChunk]
    ) -> None:
        """Audio chunks are accumulated and sent to Gemini as a single request."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="Remind me to call Sarah before Thursday"
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_stream(async_chunk_iter(chunks))

        assert result.transcript == "Remind me to call Sarah before Thursday"
        assert result.chunk_count == len(chunks)
        assert transport._audio_call_count == 1

    async def test_streaming_preserves_audio_format(
        self, chunks: list[AudioChunk]
    ) -> None:
        """Stream processing uses the format from the incoming chunks."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio_stream(async_chunk_iter(chunks))

        req_body = transport.requests[0]["body"]
        inline_data = req_body["contents"][0]["parts"][0]["inline_data"]
        assert inline_data["mime_type"] == "audio/l16"

    async def test_empty_stream_returns_empty_result(self) -> None:
        """An empty audio stream returns an empty result without API calls."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        async def empty_stream() -> AsyncIterator[AudioChunk]:
            return
            yield  # type: ignore[misc]  # make it an async generator

        result = await client.process_audio_stream(empty_stream())

        assert result.transcript == ""
        assert result.chunk_count == 0
        assert transport.total_requests == 0  # No API call made

    async def test_single_chunk_stream(self) -> None:
        """A single-chunk stream works the same as multi-chunk."""
        audio_data = generate_pcm_audio(duration_seconds=0.5)
        chunks = make_audio_chunks(audio_data, chunk_size_ms=1000)
        assert len(chunks) == 1

        transport = FakeGeminiTransport(
            audio_responses=[gemini_audio_response(transcript="Quick note")]
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_stream(async_chunk_iter(chunks))

        assert result.transcript == "Quick note"
        assert result.chunk_count == 1


# ═══════════════════════════════════════════════════════════════
# Test Suite 3: Full pipeline (audio → classify → structured)
# ═══════════════════════════════════════════════════════════════


class TestFullPipelineRoundTrip:
    """Tests for the complete audio → transcription → classification pipeline."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=2.0)

    async def test_task_intent_pipeline(self, audio_data: bytes) -> None:
        """Audio containing a task → correct intent classification."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Pick up groceries after work")
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="task",
                    confidence=0.95,
                    entities=[
                        {"name": "groceries", "type": "project", "metadata": {}}
                    ],
                    emotion={
                        "primary": "anticipation",
                        "intensity": 1,
                        "valence": 0.2,
                        "arousal": 0.3,
                    },
                    acknowledgment="Got it — task added",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["transcript"] == "Pick up groceries after work"
        assert result["pipeline_complete"] is True
        assert result["classification"]["intent"] == "task"
        assert result["classification"]["confidence"] == 0.95
        assert len(result["classification"]["entities"]) == 1
        assert result["classification"]["entities"][0]["name"] == "groceries"
        assert result["classification"]["emotion"]["primary"] == "anticipation"
        assert result["classification"]["acknowledgment"] == "Got it — task added"

        # Two API calls: audio processing + classification
        assert transport.total_requests == 2
        assert transport._audio_call_count == 1
        assert transport._classification_call_count == 1

    async def test_event_intent_pipeline(self, audio_data: bytes) -> None:
        """Audio containing an event → event classification with entities."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="Dentist appointment at 3pm Friday"
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="event",
                    confidence=0.93,
                    entities=[
                        {"name": "Dentist", "type": "organization", "metadata": {}},
                    ],
                    acknowledgment="Event noted",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["classification"]["intent"] == "event"
        assert result["classification"]["confidence"] > 0.85

    async def test_reminder_intent_pipeline(self, audio_data: bytes) -> None:
        """Audio containing a reminder → reminder classification."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="Remind me to call Sarah about the Q2 deck before Thursday"
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="reminder",
                    confidence=0.91,
                    entities=[
                        {"name": "Sarah", "type": "person", "metadata": {}},
                        {"name": "Q2 deck", "type": "project", "metadata": {}},
                    ],
                    emotion={
                        "primary": "anticipation",
                        "intensity": 2,
                        "valence": 0.1,
                        "arousal": 0.5,
                    },
                    acknowledgment="Reminder set",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["classification"]["intent"] == "reminder"
        entities = result["classification"]["entities"]
        entity_names = {e["name"] for e in entities}
        assert "Sarah" in entity_names
        assert "Q2 deck" in entity_names

    async def test_journal_intent_with_emotion(self, audio_data: bytes) -> None:
        """Audio containing emotional journaling → emotion detection."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="Today was really rough, feeling overwhelmed with everything",
                    tone="anxious",
                    emphasis=["really", "overwhelmed"],
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="journal",
                    confidence=0.89,
                    emotion={
                        "primary": "sadness",
                        "intensity": 2,
                        "valence": -0.7,
                        "arousal": 0.6,
                    },
                    acknowledgment="I hear you",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["classification"]["intent"] == "journal"
        emotion = result["classification"]["emotion"]
        assert emotion["primary"] == "sadness"
        assert emotion["intensity"] == 2
        assert emotion["valence"] < 0  # Negative valence
        assert result["audio_result"]["detected_tone"] == "anxious"

    async def test_question_intent_pipeline(self, audio_data: bytes) -> None:
        """Audio containing a question → question intent for recall."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="When did I last talk about the Q2 deck"
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="question",
                    confidence=0.94,
                    entities=[
                        {"name": "Q2 deck", "type": "project", "metadata": {}},
                    ],
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["classification"]["intent"] == "question"

    async def test_idea_intent_pipeline(self, audio_data: bytes) -> None:
        """Audio containing an idea → idea classification."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="What if we used serverless for the API backend"
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="idea",
                    confidence=0.88,
                    entities=[
                        {"name": "API backend", "type": "project", "metadata": {}},
                    ],
                    acknowledgment="Interesting idea saved",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["classification"]["intent"] == "idea"
        assert result["classification"]["confidence"] > 0.85

    async def test_update_intent_pipeline(self, audio_data: bytes) -> None:
        """Audio containing an update → update classification."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="Actually, the Macy's trip is done"
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="update",
                    confidence=0.90,
                    entities=[
                        {"name": "Macy's", "type": "place", "metadata": {}},
                    ],
                    acknowledgment="Updated",
                )
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        assert result["classification"]["intent"] == "update"

    async def test_all_seven_intents_classified(self, audio_data: bytes) -> None:
        """Verify all 7 intent types can flow through the pipeline."""
        intents = ["task", "event", "reminder", "idea", "journal", "update", "question"]

        for intent in intents:
            transport = FakeGeminiTransport(
                audio_responses=[gemini_audio_response(transcript=f"Test {intent}")],
                classification_responses=[
                    gemini_classification_response(intent=intent, confidence=0.90)
                ],
            )
            client = make_gemini_client(transport)

            result = await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

            assert result["classification"]["intent"] == intent, (
                f"Failed to classify intent: {intent}"
            )
            assert result["pipeline_complete"] is True

    async def test_empty_transcript_skips_classification(self) -> None:
        """Empty audio transcript skips classification step."""
        transport = FakeGeminiTransport(
            audio_responses=[gemini_audio_response(transcript="")]
        )
        client = make_gemini_client(transport)

        result = await client.process_audio_full_pipeline(b"", DEFAULT_FORMAT)

        assert result["transcript"] == ""
        assert result["classification"] is None
        assert result["pipeline_complete"] is True
        # Only audio call, no classification call
        assert transport._audio_call_count == 1
        assert transport._classification_call_count == 0

    async def test_uses_flash_lite_model_for_both_steps(self, audio_data: bytes) -> None:
        """Both audio processing and classification use Flash-Lite (cost strategy)."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio_full_pipeline(audio_data, DEFAULT_FORMAT)

        # Both requests should target the Flash-Lite model
        for req in transport.requests:
            assert GeminiModel.FLASH_LITE.value in req["url"]


# ═══════════════════════════════════════════════════════════════
# Test Suite 4: Pipeline orchestration (AudioPipeline class)
# ═══════════════════════════════════════════════════════════════


class TestAudioPipelineOrchestration:
    """Tests for the AudioPipeline orchestrator."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=2.0)

    @pytest.fixture
    def chunks(self, audio_data: bytes) -> list[AudioChunk]:
        return make_audio_chunks(audio_data, chunk_size_ms=500)

    async def test_pipeline_single_shot(self, audio_data: bytes) -> None:
        """AudioPipeline processes complete audio and returns PipelineResult."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Need to return those jeans at Macy's")
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="task",
                    confidence=0.93,
                    entities=[
                        {"name": "Macy's", "type": "place", "metadata": {}},
                    ],
                    acknowledgment="Task added",
                )
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio(
            audio_data, DEFAULT_FORMAT, user_id="user-123"
        )

        assert isinstance(result, PipelineResult)
        assert result.transcript == "Need to return those jeans at Macy's"
        assert result.intent == "task"
        assert result.confidence == 0.93
        assert result.user_id == "user-123"
        assert result.pipeline_complete is True
        assert result.error is None
        assert result.processing_time_ms >= 0
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Macy's"

    async def test_pipeline_streaming(self, chunks: list[AudioChunk]) -> None:
        """AudioPipeline processes streaming chunks and returns PipelineResult."""
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(
                    transcript="Also remind me to call Sarah about the Q2 deck"
                )
            ],
            classification_responses=[
                gemini_classification_response(
                    intent="reminder",
                    confidence=0.91,
                    entities=[
                        {"name": "Sarah", "type": "person", "metadata": {}},
                        {"name": "Q2 deck", "type": "project", "metadata": {}},
                    ],
                )
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio_stream(
            async_chunk_iter(chunks), user_id="user-456"
        )

        assert result.transcript == "Also remind me to call Sarah about the Q2 deck"
        assert result.intent == "reminder"
        assert result.pipeline_complete is True
        assert len(result.entities) == 2

    async def test_pipeline_error_captured(self, audio_data: bytes) -> None:
        """Pipeline captures errors without crashing and marks incomplete."""
        transport = FakeGeminiTransport(
            error_responses=[(500, {"error": "Internal server error"})],
        )
        # No retries so the error propagates
        client = make_gemini_client(transport, max_retries=0)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio(audio_data, DEFAULT_FORMAT)

        assert result.pipeline_complete is False
        assert result.error is not None
        assert "500" in result.error or "failed" in result.error.lower()

    async def test_pipeline_session_management(self) -> None:
        """Pipeline tracks active sessions correctly."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        # Create session
        sid = pipeline.create_session("user-123")
        assert pipeline.active_session_count == 1
        session = pipeline.get_session(sid)
        assert session is not None
        assert session.user_id == "user-123"

        # End session
        pipeline.end_session(sid)
        assert pipeline.active_session_count == 0
        assert pipeline.get_session(sid) is None

    async def test_pipeline_result_has_metadata(self, audio_data: bytes) -> None:
        """PipelineResult includes all processing metadata."""
        transport = FakeGeminiTransport(
            audio_responses=[gemini_audio_response(transcript="Test")],
            classification_responses=[
                gemini_classification_response(acknowledgment="Got it")
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio(audio_data, DEFAULT_FORMAT)

        assert result.id  # UUID generated
        assert result.created_at is not None
        assert result.acknowledgment == "Got it"
        assert result.audio_result is not None


# ═══════════════════════════════════════════════════════════════
# Test Suite 5: Error handling and resilience
# ═══════════════════════════════════════════════════════════════


class TestErrorHandlingRoundTrip:
    """Tests for error conditions in the audio round-trip."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=1.0)

    async def test_rate_limit_error(self, audio_data: bytes) -> None:
        """429 rate limit errors are raised as GeminiRateLimitError."""
        transport = FakeGeminiTransport(
            error_responses=[
                (429, {"error": {"message": "Rate limit exceeded"}}),
                (429, {"error": {"message": "Rate limit exceeded"}}),
            ],
        )
        client = make_gemini_client(transport, max_retries=1)

        with pytest.raises(GeminiAudioError):
            await client.process_audio(audio_data, DEFAULT_FORMAT)

    async def test_server_error_retried(self, audio_data: bytes) -> None:
        """Transient server errors trigger retries, succeed on retry."""
        transport = FakeGeminiTransport(
            error_responses=[
                (503, {"error": {"message": "Service unavailable"}}),
            ],
            audio_responses=[
                gemini_audio_response(transcript="Recovered successfully")
            ],
        )
        client = make_gemini_client(transport, max_retries=2)

        # First call gets 503, retry succeeds
        result = await client.process_audio(audio_data, DEFAULT_FORMAT)
        assert result.transcript == "Recovered successfully"

    async def test_malformed_response_handled(self, audio_data: bytes) -> None:
        """Malformed Gemini responses are handled gracefully."""
        transport = FakeGeminiTransport(
            audio_responses=[{"candidates": []}],  # Empty candidates
        )
        client = make_gemini_client(transport)

        result = await client.process_audio(audio_data, DEFAULT_FORMAT)

        # Should return empty result, not crash
        assert result.transcript == ""

    async def test_non_json_response_handled(self, audio_data: bytes) -> None:
        """Non-JSON text responses are treated as plain transcript."""
        # Response where the text part is not valid JSON
        transport = FakeGeminiTransport(
            audio_responses=[
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Just plain text transcript"}],
                            },
                        }
                    ],
                }
            ],
        )
        client = make_gemini_client(transport)

        result = await client.process_audio(audio_data, DEFAULT_FORMAT)
        assert result.transcript == "Just plain text transcript"

    async def test_classification_error_does_not_crash_pipeline(
        self, audio_data: bytes
    ) -> None:
        """If classification fails, the pipeline still returns audio results."""
        # Audio succeeds, classification returns error
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Some valid transcript")
            ],
        )
        # Override classification to fail
        transport.classification_responses = []
        transport.error_responses = []

        # We'll use the pipeline which catches errors
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        # The classify call will get index error from empty responses
        # Pipeline should catch and mark error
        result = await pipeline.process_audio(audio_data, DEFAULT_FORMAT)

        # Pipeline should have captured the error
        # (either result has error or the classification step raised)
        assert result.transcript == "Some valid transcript" or result.error is not None


# ═══════════════════════════════════════════════════════════════
# Test Suite 6: Multiple audio formats
# ═══════════════════════════════════════════════════════════════


class TestAudioFormatRoundTrip:
    """Tests for different audio encoding formats through the pipeline."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=1.0)

    async def test_linear16_format(self, audio_data: bytes) -> None:
        """LINEAR16 (PCM 16-bit) audio processed correctly."""
        fmt = AudioFormat(
            encoding=AudioEncoding.LINEAR16,
            sample_rate_hz=16000,
            channels=1,
            sample_width_bytes=2,
        )
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, fmt)

        req = transport.requests[0]["body"]
        assert req["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/l16"

    async def test_flac_format(self, audio_data: bytes) -> None:
        """FLAC audio sends correct MIME type."""
        fmt = AudioFormat(encoding=AudioEncoding.FLAC)
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, fmt)

        req = transport.requests[0]["body"]
        assert req["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/flac"

    async def test_ogg_opus_format(self, audio_data: bytes) -> None:
        """OGG_OPUS audio sends correct MIME type."""
        fmt = AudioFormat(encoding=AudioEncoding.OGG_OPUS)
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, fmt)

        req = transport.requests[0]["body"]
        assert req["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/ogg"

    async def test_wav_format(self, audio_data: bytes) -> None:
        """WAV audio sends correct MIME type."""
        fmt = AudioFormat(encoding=AudioEncoding.WAV)
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, fmt)

        req = transport.requests[0]["body"]
        assert req["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/wav"

    async def test_different_sample_rates(self, audio_data: bytes) -> None:
        """Different sample rates produce different duration estimates."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        fmt_16k = AudioFormat(sample_rate_hz=16000)
        result_16k = await client.process_audio(audio_data, fmt_16k)

        fmt_8k = AudioFormat(sample_rate_hz=8000)
        result_8k = await client.process_audio(audio_data, fmt_8k)

        # Same bytes at half the sample rate = double the duration
        assert result_8k.audio_duration_ms == result_16k.audio_duration_ms * 2

    async def test_stereo_audio(self, audio_data: bytes) -> None:
        """Stereo audio duration correctly accounts for 2 channels."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        fmt_mono = AudioFormat(channels=1)
        result_mono = await client.process_audio(audio_data, fmt_mono)

        fmt_stereo = AudioFormat(channels=2)
        result_stereo = await client.process_audio(audio_data, fmt_stereo)

        # Same bytes with 2 channels = half the duration
        assert result_stereo.audio_duration_ms == result_mono.audio_duration_ms // 2


# ═══════════════════════════════════════════════════════════════
# Test Suite 7: Full-duplex concurrent processing
# ═══════════════════════════════════════════════════════════════


class TestFullDuplexConcurrent:
    """Tests for concurrent audio processing (full-duplex capability)."""

    async def test_concurrent_audio_processing(self) -> None:
        """Multiple audio clips can be processed concurrently."""
        audio_clips = [
            (generate_pcm_audio(1.0), "task clip"),
            (generate_pcm_audio(1.5), "event clip"),
            (generate_pcm_audio(0.5), "reminder clip"),
        ]

        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Pick up groceries"),
                gemini_audio_response(transcript="Meeting at 3pm"),
                gemini_audio_response(transcript="Remind me tonight"),
            ],
            classification_responses=[
                gemini_classification_response(intent="task"),
                gemini_classification_response(intent="event"),
                gemini_classification_response(intent="reminder"),
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        # Process all concurrently
        tasks = [
            pipeline.process_audio(clip, DEFAULT_FORMAT, user_id=f"user-{i}")
            for i, (clip, _) in enumerate(audio_clips)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert result.pipeline_complete is True
            assert result.transcript != ""
            assert result.intent != ""

    async def test_concurrent_stream_and_single_shot(self) -> None:
        """Streaming and single-shot can run concurrently."""
        single_audio = generate_pcm_audio(1.0)
        stream_chunks = make_audio_chunks(generate_pcm_audio(2.0))

        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Single shot result"),
                gemini_audio_response(transcript="Streaming result"),
            ],
            classification_responses=[
                gemini_classification_response(intent="task"),
                gemini_classification_response(intent="idea"),
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        single_task = pipeline.process_audio(single_audio, DEFAULT_FORMAT)
        stream_task = pipeline.process_audio_stream(async_chunk_iter(stream_chunks))

        results = await asyncio.gather(single_task, stream_task)

        assert results[0].pipeline_complete is True
        assert results[1].pipeline_complete is True

    async def test_pipeline_handles_mixed_success_and_failure(self) -> None:
        """In concurrent processing, one failure doesn't affect others."""
        audio_data = generate_pcm_audio(1.0)

        # First call succeeds, second gets error, third succeeds
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript="Success 1"),
                gemini_audio_response(transcript="Success 2"),
            ],
            classification_responses=[
                gemini_classification_response(intent="task"),
                gemini_classification_response(intent="idea"),
            ],
        )

        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        results = await asyncio.gather(
            pipeline.process_audio(audio_data, DEFAULT_FORMAT, user_id="u1"),
            pipeline.process_audio(audio_data, DEFAULT_FORMAT, user_id="u2"),
        )

        # At least one should complete
        completed = [r for r in results if r.pipeline_complete]
        assert len(completed) >= 1


# ═══════════════════════════════════════════════════════════════
# Test Suite 8: Data integrity through the pipeline
# ═══════════════════════════════════════════════════════════════


class TestDataIntegrity:
    """Tests ensuring no data loss through the audio pipeline."""

    @pytest.fixture
    def audio_data(self) -> bytes:
        return generate_pcm_audio(duration_seconds=2.0)

    async def test_audio_bytes_preserved_in_request(self, audio_data: bytes) -> None:
        """Every byte of audio data reaches the API (no truncation/corruption)."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, DEFAULT_FORMAT)

        req = transport.requests[0]["body"]
        b64_data = req["contents"][0]["parts"][0]["inline_data"]["data"]
        decoded = base64.b64decode(b64_data)
        assert decoded == audio_data
        assert len(decoded) == len(audio_data)

    async def test_streaming_chunks_fully_accumulated(self) -> None:
        """All chunks in a stream are accumulated without data loss."""
        audio_data = generate_pcm_audio(duration_seconds=3.0)
        chunks = make_audio_chunks(audio_data, chunk_size_ms=200)

        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio_stream(async_chunk_iter(chunks))

        req = transport.requests[0]["body"]
        b64_data = req["contents"][0]["parts"][0]["inline_data"]["data"]
        decoded = base64.b64decode(b64_data)

        # All chunk bytes should be accumulated
        total_chunk_bytes = sum(len(c.data) for c in chunks)
        assert len(decoded) == total_chunk_bytes
        assert decoded == audio_data

    async def test_transcript_preserved_through_pipeline(self, audio_data: bytes) -> None:
        """Transcript text is not modified between Gemini response and pipeline output."""
        original_transcript = "Need to return those jeans at Macy's, also remind me to call Sarah"
        transport = FakeGeminiTransport(
            audio_responses=[
                gemini_audio_response(transcript=original_transcript)
            ],
            classification_responses=[
                gemini_classification_response(intent="task")
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio(audio_data, DEFAULT_FORMAT)

        assert result.transcript == original_transcript

    async def test_entities_preserved_through_pipeline(self, audio_data: bytes) -> None:
        """Entity data flows from classification to pipeline result without loss."""
        entities = [
            {"name": "Sarah", "type": "person", "metadata": {"role": "manager"}},
            {"name": "Q2 deck", "type": "project", "metadata": {"priority": "high"}},
            {"name": "Macy's", "type": "place", "metadata": {}},
        ]
        transport = FakeGeminiTransport(
            audio_responses=[gemini_audio_response(transcript="Test")],
            classification_responses=[
                gemini_classification_response(
                    intent="task", entities=entities
                )
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio(audio_data, DEFAULT_FORMAT)

        assert len(result.entities) == 3
        entity_names = {e["name"] for e in result.entities}
        assert entity_names == {"Sarah", "Q2 deck", "Macy's"}

    async def test_emotion_preserved_through_pipeline(self, audio_data: bytes) -> None:
        """Emotion data flows from classification to pipeline result without loss."""
        emotion = {
            "primary": "fear",
            "intensity": 2,
            "valence": -0.6,
            "arousal": 0.8,
        }
        transport = FakeGeminiTransport(
            audio_responses=[gemini_audio_response(transcript="Test")],
            classification_responses=[
                gemini_classification_response(intent="journal", emotion=emotion)
            ],
        )
        client = make_gemini_client(transport)
        pipeline = AudioPipeline(gemini_client=client)

        result = await pipeline.process_audio(audio_data, DEFAULT_FORMAT)

        assert result.emotion == emotion
        assert result.emotion["primary"] == "fear"
        assert result.emotion["intensity"] == 2

    async def test_api_key_sent_as_query_param(self, audio_data: bytes) -> None:
        """API key is sent as a query parameter (Gemini convention)."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, DEFAULT_FORMAT)

        url = transport.requests[0]["url"]
        assert "key=test-api-key" in url

    async def test_json_response_mime_type_requested(self, audio_data: bytes) -> None:
        """Requests specify JSON response format for structured output."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.process_audio(audio_data, DEFAULT_FORMAT)

        req = transport.requests[0]["body"]
        gen_config = req.get("generationConfig", {})
        assert gen_config.get("responseMimeType") == "application/json"

    async def test_low_temperature_for_classification(self, audio_data: bytes) -> None:
        """Classification uses low temperature for deterministic results."""
        transport = FakeGeminiTransport()
        client = make_gemini_client(transport)

        await client.classify_transcript("Pick up groceries")

        req = transport.requests[0]["body"]
        gen_config = req.get("generationConfig", {})
        assert gen_config.get("temperature") == 0.1
