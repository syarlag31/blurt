"""Tests for the end-to-end audio processing pipeline.

Validates:
- Batch processing: raw audio bytes → chunks → Gemini → merged result
- Streaming processing: async audio stream → chunks → Gemini → result
- Session management
- Error handling and resilience
- Full pipeline: audio → transcription → classification
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from blurt.audio.models import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioProcessingResult,
    AudioStreamSession,
    StreamState,
)
from blurt.audio.pipeline import AudioPipeline, PipelineResult
from blurt.config.settings import GeminiConfig
from blurt.gemini.audio_client import GeminiAudioClient, GeminiAudioError


# ── Test helpers ─────────────────────────────────────────────────


def make_pcm_tone(duration_ms: int, amplitude: int = 10000, sample_rate: int = 16000) -> bytes:
    """Generate a sine wave tone as PCM LINEAR16 audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * 440 * t))
        value = max(-32768, min(32767, value))
        samples.append(value)
    return struct.pack(f"<{num_samples}h", *samples)


def make_pcm_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


DEFAULT_FORMAT = AudioFormat(
    encoding=AudioEncoding.LINEAR16,
    sample_rate_hz=16000,
    channels=1,
    sample_width_bytes=2,
)


def make_gemini_response(
    transcript: str = "Hello world",
    tone: str = "neutral",
    language: str = "en",
) -> dict[str, Any]:
    """Build a mock Gemini API JSON response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "transcript": transcript,
                                "segments": [
                                    {
                                        "text": transcript,
                                        "start_ms": 0,
                                        "end_ms": 1000,
                                        "confidence": 0.95,
                                    }
                                ],
                                "detected_tone": tone,
                                "detected_emphasis": [],
                                "language": language,
                            })
                        }
                    ]
                }
            }
        ]
    }


def make_classification_response(
    intent: str = "task",
    confidence: float = 0.92,
) -> dict[str, Any]:
    """Build a mock classification JSON response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "intent": intent,
                                "confidence": confidence,
                                "entities": [
                                    {"name": "grocery store", "type": "place", "metadata": {}}
                                ],
                                "emotion": {
                                    "primary": "anticipation",
                                    "intensity": 1,
                                    "valence": 0.3,
                                    "arousal": 0.4,
                                },
                                "acknowledgment": "Got it — task added.",
                            })
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def gemini_config() -> GeminiConfig:
    """Create a test Gemini config."""
    return GeminiConfig(
        api_key="test-api-key",
        flash_lite_model="gemini-2.5-flash-lite",
        flash_model="gemini-2.5-flash",
        connect_timeout=5.0,
        read_timeout=10.0,
        max_retries=1,
    )


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


def make_mock_response(
    json_data: dict, status_code: int = 200
) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = json.dumps(json_data).encode()
    return resp


# ── Pipeline batch processing tests ──────────────────────────────


class TestPipelineBatchProcessing:
    """Tests for AudioPipeline.process_audio (batch/single-shot)."""

    @pytest.mark.asyncio
    async def test_process_audio_returns_pipeline_result(
        self, gemini_config, mock_http_client
    ):
        """Full single-shot pipeline should return a PipelineResult with transcript."""
        # Setup mock responses: first for audio, second for classification
        audio_resp = make_mock_response(make_gemini_response("Pick up groceries"))
        class_resp = make_mock_response(make_classification_response("task", 0.95))
        mock_http_client.post = AsyncMock(side_effect=[audio_resp, class_resp])

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        audio = make_pcm_tone(2000)
        result = await pipeline.process_audio(
            audio_data=audio,
            audio_format=DEFAULT_FORMAT,
            user_id="user-1",
        )

        assert isinstance(result, PipelineResult)
        assert result.transcript == "Pick up groceries"
        assert result.intent == "task"
        assert result.confidence == 0.95
        assert result.pipeline_complete is True
        assert result.user_id == "user-1"
        assert result.processing_time_ms >= 0  # May be 0 with mocked fast responses

    @pytest.mark.asyncio
    async def test_process_empty_audio(self, gemini_config, mock_http_client):
        """Empty audio should return empty result without API call."""
        # Return empty transcript from Gemini
        resp = make_mock_response(make_gemini_response(""))
        mock_http_client.post = AsyncMock(return_value=resp)

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        # Very short audio that produces empty transcript
        audio = make_pcm_silence(100)
        result = await pipeline.process_audio(
            audio_data=audio,
            audio_format=DEFAULT_FORMAT,
        )

        assert isinstance(result, PipelineResult)
        assert result.pipeline_complete is True

    @pytest.mark.asyncio
    async def test_process_audio_handles_api_error(
        self, gemini_config, mock_http_client
    ):
        """API errors should be captured in result, not crash the pipeline."""
        mock_http_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        audio = make_pcm_tone(1000)
        result = await pipeline.process_audio(
            audio_data=audio,
            audio_format=DEFAULT_FORMAT,
        )

        assert result.pipeline_complete is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_classification_extracts_entities(
        self, gemini_config, mock_http_client
    ):
        """Pipeline should extract entities from classification response."""
        audio_resp = make_mock_response(
            make_gemini_response("Call Sarah about the Q2 deck")
        )
        class_resp = make_mock_response({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "intent": "task",
                            "confidence": 0.9,
                            "entities": [
                                {"name": "Sarah", "type": "person", "metadata": {"role": "coworker"}},
                                {"name": "Q2 deck", "type": "project", "metadata": {}},
                            ],
                            "emotion": {
                                "primary": "anticipation",
                                "intensity": 1,
                                "valence": 0.2,
                                "arousal": 0.5,
                            },
                            "acknowledgment": "Got it.",
                        })
                    }]
                }
            }]
        })
        mock_http_client.post = AsyncMock(side_effect=[audio_resp, class_resp])

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        result = await pipeline.process_audio(
            audio_data=make_pcm_tone(2000),
            audio_format=DEFAULT_FORMAT,
        )

        assert len(result.entities) == 2
        assert result.entities[0]["name"] == "Sarah"
        assert result.entities[1]["name"] == "Q2 deck"


# ── Pipeline streaming tests ────────────────────────────────────


class TestPipelineStreamProcessing:
    """Tests for AudioPipeline.process_audio_stream (streaming)."""

    @pytest.mark.asyncio
    async def test_stream_processing(self, gemini_config, mock_http_client):
        """Stream of chunks should produce a unified PipelineResult."""
        # Mock: first call is audio processing, second is classification
        audio_resp = make_mock_response(
            make_gemini_response("Need to return those jeans at Macys")
        )
        class_resp = make_mock_response(
            make_classification_response("task", 0.91)
        )
        mock_http_client.post = AsyncMock(
            side_effect=[audio_resp, class_resp]
        )

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        async def chunk_source() -> AsyncIterator[AudioChunk]:
            for i in range(3):
                yield AudioChunk(
                    data=make_pcm_tone(1000),
                    sequence_number=i,
                    timestamp_ms=i * 1000,
                    duration_ms=1000,
                    is_final=(i == 2),
                    format=DEFAULT_FORMAT,
                )

        result = await pipeline.process_audio_stream(
            chunks=chunk_source(),
            user_id="user-2",
        )

        assert isinstance(result, PipelineResult)
        assert result.transcript != ""
        assert result.pipeline_complete is True
        assert result.user_id == "user-2"


# ── Session management tests ────────────────────────────────────


class TestSessionManagement:
    """Tests for pipeline session lifecycle."""

    def test_create_session(self, gemini_config, mock_http_client):
        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        session_id = pipeline.create_session(user_id="user-1")
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_get_session(self, gemini_config, mock_http_client):
        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        session_id = pipeline.create_session(user_id="user-1")
        session = pipeline.get_session(session_id)
        assert session is not None
        assert session.user_id == "user-1"

    def test_get_nonexistent_session(self, gemini_config, mock_http_client):
        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        assert pipeline.get_session("nonexistent") is None

    def test_end_session(self, gemini_config, mock_http_client):
        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        session_id = pipeline.create_session(user_id="user-1")
        pipeline.end_session(session_id)
        assert pipeline.get_session(session_id) is None


# ── Gemini audio client tests ───────────────────────────────────


class TestGeminiAudioClient:
    """Tests for the Gemini audio client used by the pipeline."""

    @pytest.mark.asyncio
    async def test_process_audio_single_shot(
        self, gemini_config, mock_http_client
    ):
        """process_audio should send audio and return parsed result."""
        resp = make_mock_response(make_gemini_response("Hello world", "calm"))
        mock_http_client.post = AsyncMock(return_value=resp)

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        result = await client.process_audio(
            audio_data=make_pcm_tone(1000),
            audio_format=DEFAULT_FORMAT,
        )

        assert isinstance(result, AudioProcessingResult)
        assert result.transcript == "Hello world"
        assert result.detected_tone == "calm"
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_classify_transcript(
        self, gemini_config, mock_http_client
    ):
        """classify_transcript should return structured classification."""
        resp = make_mock_response(make_classification_response("reminder", 0.88))
        mock_http_client.post = AsyncMock(return_value=resp)

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        result = await client.classify_transcript(
            "Remind me to call Mom tonight"
        )

        assert result["intent"] == "reminder"
        assert result["confidence"] == 0.88

    @pytest.mark.asyncio
    async def test_full_pipeline(
        self, gemini_config, mock_http_client
    ):
        """process_audio_full_pipeline should chain audio → classification."""
        audio_resp = make_mock_response(
            make_gemini_response("Dentist at 3pm Friday")
        )
        class_resp = make_mock_response(
            make_classification_response("event", 0.93)
        )
        mock_http_client.post = AsyncMock(
            side_effect=[audio_resp, class_resp]
        )

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        result = await client.process_audio_full_pipeline(
            audio_data=make_pcm_tone(2000),
            audio_format=DEFAULT_FORMAT,
        )

        assert result["transcript"] == "Dentist at 3pm Friday"
        assert result["classification"]["intent"] == "event"
        assert result["pipeline_complete"] is True

    @pytest.mark.asyncio
    async def test_retry_on_timeout(
        self, gemini_config, mock_http_client
    ):
        """Client should retry on timeout and succeed on retry."""
        success_resp = make_mock_response(make_gemini_response("Hello"))
        mock_http_client.post = AsyncMock(
            side_effect=[
                httpx.TimeoutException("timeout"),
                success_resp,
            ]
        )

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        result = await client.process_audio(
            audio_data=make_pcm_tone(1000),
            audio_format=DEFAULT_FORMAT,
        )

        assert result.transcript == "Hello"
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_error(
        self, gemini_config, mock_http_client
    ):
        """Client should raise on rate limit after retries."""
        rate_limit_resp = make_mock_response({}, status_code=429)
        mock_http_client.post = AsyncMock(return_value=rate_limit_resp)

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        with pytest.raises(GeminiAudioError):
            await client.process_audio(
                audio_data=make_pcm_tone(1000),
                audio_format=DEFAULT_FORMAT,
            )

    @pytest.mark.asyncio
    async def test_non_json_response_handled(
        self, gemini_config, mock_http_client
    ):
        """Non-JSON response from Gemini should be treated as plain transcript."""
        resp = make_mock_response({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Just a plain transcript without JSON"}]
                }
            }]
        })
        mock_http_client.post = AsyncMock(return_value=resp)

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        result = await client.process_audio(
            audio_data=make_pcm_tone(1000),
            audio_format=DEFAULT_FORMAT,
        )

        assert result.transcript == "Just a plain transcript without JSON"

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(
        self, gemini_config, mock_http_client
    ):
        """Empty candidates should return empty transcript."""
        resp = make_mock_response({"candidates": []})
        mock_http_client.post = AsyncMock(return_value=resp)

        client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )

        result = await client.process_audio(
            audio_data=make_pcm_tone(1000),
            audio_format=DEFAULT_FORMAT,
        )

        assert result.transcript == ""
        assert result.is_empty


# ── End-to-end pipeline integration tests ────────────────────────


class TestEndToEndPipeline:
    """Integration tests validating the full audio → understanding flow."""

    @pytest.mark.asyncio
    async def test_complete_blurt_flow(
        self, gemini_config, mock_http_client
    ):
        """Simulate a complete blurt: voice input → classification → entities + emotion."""
        audio_resp = make_mock_response(make_gemini_response(
            "Need to return those jeans at Macys also remind me to call Sarah about the Q2 deck before Thursday",
            tone="neutral",
        ))
        class_resp = make_mock_response({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "intent": "task",
                            "confidence": 0.87,
                            "entities": [
                                {"name": "Macys", "type": "place", "metadata": {}},
                                {"name": "Sarah", "type": "person", "metadata": {}},
                                {"name": "Q2 deck", "type": "project", "metadata": {}},
                            ],
                            "emotion": {
                                "primary": "anticipation",
                                "intensity": 1,
                                "valence": 0.3,
                                "arousal": 0.4,
                            },
                            "acknowledgment": "Got it — two tasks noted.",
                        })
                    }]
                }
            }]
        })
        mock_http_client.post = AsyncMock(side_effect=[audio_resp, class_resp])

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        result = await pipeline.process_audio(
            audio_data=make_pcm_tone(5000),
            audio_format=DEFAULT_FORMAT,
            user_id="dogfood-user",
        )

        # Verify pipeline integrity: every step completed
        assert result.pipeline_complete is True
        assert result.transcript != ""
        assert result.intent == "task"
        assert result.confidence > 0.85
        assert len(result.entities) == 3
        assert result.emotion["primary"] == "anticipation"
        assert result.acknowledgment != ""
        assert result.error is None

    @pytest.mark.asyncio
    async def test_emotion_detection_in_pipeline(
        self, gemini_config, mock_http_client
    ):
        """Pipeline should detect and pass through emotional state."""
        audio_resp = make_mock_response(make_gemini_response(
            "Today was rough, really overwhelmed with everything",
            tone="sad",
        ))
        class_resp = make_mock_response({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "intent": "journal",
                            "confidence": 0.94,
                            "entities": [],
                            "emotion": {
                                "primary": "sadness",
                                "intensity": 2,
                                "valence": -0.6,
                                "arousal": 0.3,
                            },
                            "acknowledgment": "Heard you. That sounds tough.",
                        })
                    }]
                }
            }]
        })
        mock_http_client.post = AsyncMock(side_effect=[audio_resp, class_resp])

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        result = await pipeline.process_audio(
            audio_data=make_pcm_tone(3000),
            audio_format=DEFAULT_FORMAT,
        )

        assert result.intent == "journal"
        assert result.emotion["primary"] == "sadness"
        assert result.emotion["intensity"] == 2
        assert result.emotion["valence"] < 0
        assert "tough" in result.acknowledgment.lower() or "heard" in result.acknowledgment.lower()

    @pytest.mark.asyncio
    async def test_pipeline_result_has_timing(
        self, gemini_config, mock_http_client
    ):
        """Pipeline result should include processing time metadata."""
        resp = make_mock_response(make_gemini_response("Test"))
        class_resp = make_mock_response(make_classification_response())
        mock_http_client.post = AsyncMock(side_effect=[resp, class_resp])

        gemini_client = GeminiAudioClient(
            config=gemini_config, http_client=mock_http_client
        )
        pipeline = AudioPipeline(gemini_client=gemini_client)

        result = await pipeline.process_audio(
            audio_data=make_pcm_tone(1000),
            audio_format=DEFAULT_FORMAT,
        )

        assert result.processing_time_ms >= 0
        assert result.created_at is not None


# ── Audio format and model tests ─────────────────────────────────


class TestAudioModels:
    """Tests for audio data models."""

    def test_audio_format_bytes_per_second(self):
        fmt = AudioFormat(
            encoding=AudioEncoding.LINEAR16,
            sample_rate_hz=16000,
            channels=1,
            sample_width_bytes=2,
        )
        assert fmt.bytes_per_second == 32000

    def test_audio_format_stereo(self):
        fmt = AudioFormat(
            encoding=AudioEncoding.LINEAR16,
            sample_rate_hz=44100,
            channels=2,
            sample_width_bytes=2,
        )
        assert fmt.bytes_per_second == 176400

    def test_audio_format_mime_types(self):
        assert AudioFormat(encoding=AudioEncoding.LINEAR16).mime_type == "audio/l16"
        assert AudioFormat(encoding=AudioEncoding.WAV).mime_type == "audio/wav"
        assert AudioFormat(encoding=AudioEncoding.FLAC).mime_type == "audio/flac"
        assert AudioFormat(encoding=AudioEncoding.OGG_OPUS).mime_type == "audio/ogg"

    def test_audio_chunk_size_bytes(self):
        data = b"\x00" * 1000
        chunk = AudioChunk(
            data=data,
            sequence_number=0,
            timestamp_ms=0,
            duration_ms=31,
        )
        assert chunk.size_bytes == 1000

    def test_audio_processing_result_is_empty(self):
        assert AudioProcessingResult(transcript="").is_empty
        assert AudioProcessingResult(transcript="   ").is_empty
        assert not AudioProcessingResult(transcript="hello").is_empty

    def test_stream_session_states(self):
        session = AudioStreamSession(user_id="test")
        assert session.state == StreamState.IDLE
        assert session.chunks_received == 0
        assert session.partial_transcript == ""
