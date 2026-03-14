"""Gemini API client for audio processing.

Handles raw audio → Gemini 2 multimodal API → structured understanding.
Uses the two-model strategy:
  - Flash-Lite for fast classification/extraction
  - Flash for reasoning/insights

Designed for full-duplex operation: audio streams in while
responses stream back concurrently.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, AsyncIterator

import httpx

from blurt.audio.models import (
    AudioChunk,
    AudioFormat,
    AudioProcessingResult,
    TranscriptionSegment,
)
from blurt.config.settings import GeminiConfig


class GeminiAudioError(Exception):
    """Raised when Gemini API returns an error for audio processing."""

    def __init__(self, message: str, status_code: int | None = None, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class GeminiRateLimitError(GeminiAudioError):
    """Raised when Gemini API rate limit is hit."""


class GeminiAudioClient:
    """Client for processing audio through the Gemini 2 multimodal API.

    Supports:
    - Single-shot audio processing (complete audio → complete result)
    - Streaming audio processing (chunks in → partial results out)
    - Two-model strategy for cost optimization

    The client handles retry logic, error mapping, and response parsing.
    """

    def __init__(self, config: GeminiConfig, http_client: httpx.AsyncClient | None = None):
        self._config = config
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=config.connect_timeout,
                read=config.read_timeout,
                write=config.connect_timeout,
                pool=config.connect_timeout,
            ),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> GeminiAudioClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ── Single-shot audio processing ─────────────────────────────

    async def process_audio(
        self,
        audio_data: bytes,
        audio_format: AudioFormat,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> AudioProcessingResult:
        """Process a complete audio clip through Gemini.

        Sends raw audio bytes with multimodal content parts.
        Returns structured transcription + audio understanding.
        """
        model = model or self._config.flash_lite_model
        start_time = time.monotonic()

        request_body = self._build_audio_request(
            audio_data=audio_data,
            mime_type=audio_format.mime_type,
            system_prompt=system_prompt or self._default_audio_system_prompt(),
        )

        response_data = await self._api_call(
            model=model,
            body=request_body,
        )

        processing_time_ms = int((time.monotonic() - start_time) * 1000)
        return self._parse_audio_response(
            response_data,
            model_used=model,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=self._estimate_duration_ms(audio_data, audio_format),
        )

    # ── Streaming audio processing ───────────────────────────────

    async def process_audio_stream(
        self,
        chunks: AsyncIterator[AudioChunk],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> AudioProcessingResult:
        """Process a stream of audio chunks through Gemini.

        Accumulates chunks, then processes the complete audio.
        For true streaming transcription in production, this would
        use Gemini's streaming API; here we accumulate for reliability.
        """
        model = model or self._config.flash_lite_model
        start_time = time.monotonic()

        accumulated_data = bytearray()
        chunk_count = 0
        audio_format = AudioFormat()

        async for chunk in chunks:
            accumulated_data.extend(chunk.data)
            chunk_count += 1
            audio_format = chunk.format

        if not accumulated_data:
            return AudioProcessingResult(
                transcript="",
                model_used=model,
                processing_time_ms=0,
                chunk_count=0,
            )

        request_body = self._build_audio_request(
            audio_data=bytes(accumulated_data),
            mime_type=audio_format.mime_type,
            system_prompt=system_prompt or self._default_audio_system_prompt(),
        )

        response_data = await self._api_call(model=model, body=request_body)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)
        result = self._parse_audio_response(
            response_data,
            model_used=model,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=self._estimate_duration_ms(
                bytes(accumulated_data), audio_format
            ),
        )
        result.chunk_count = chunk_count
        return result

    # ── Classification (Flash-Lite) ──────────────────────────────

    async def classify_transcript(
        self, transcript: str
    ) -> dict[str, Any]:
        """Classify a transcript into intent, entities, emotion using Flash-Lite.

        Returns a structured dict with:
        - intent: one of task, event, reminder, idea, journal, update, question
        - confidence: float 0.0-1.0
        - entities: list of {name, type, metadata}
        - emotion: {primary, intensity, valence, arousal}
        """
        prompt = self._classification_prompt(transcript)

        response_data = await self._api_call(
            model=self._config.flash_lite_model,
            body={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.1,
                },
            },
        )

        return self._parse_json_response(response_data)

    # ── Full pipeline: audio → structured result ─────────────────

    async def process_audio_full_pipeline(
        self,
        audio_data: bytes,
        audio_format: AudioFormat,
    ) -> dict[str, Any]:
        """Complete audio round-trip: raw audio → understanding → structured output.

        This is the core full-duplex pipeline:
        1. Send raw audio to Gemini for transcription + audio analysis
        2. Classify the transcript (intent, entities, emotion)
        3. Return unified structured result

        Uses Flash-Lite for both steps (cost-optimized).
        """
        # Step 1: Audio → transcript + audio understanding
        audio_result = await self.process_audio(audio_data, audio_format)

        if audio_result.is_empty:
            return {
                "transcript": "",
                "audio_result": audio_result.model_dump(),
                "classification": None,
                "pipeline_complete": True,
            }

        # Step 2: Transcript → classification
        classification = await self.classify_transcript(audio_result.transcript)

        return {
            "transcript": audio_result.transcript,
            "audio_result": audio_result.model_dump(),
            "classification": classification,
            "pipeline_complete": True,
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _build_audio_request(
        self,
        audio_data: bytes,
        mime_type: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Build the multimodal request body with inline audio data."""
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_b64,
                            }
                        },
                        {
                            "text": (
                                "Transcribe this audio and analyze the speaker's tone, "
                                "emphasis, and emotional cues. Return a JSON object with:\n"
                                '- "transcript": the full text\n'
                                '- "segments": [{text, start_ms, end_ms, confidence}]\n'
                                '- "detected_tone": overall tone (e.g., calm, anxious, excited)\n'
                                '- "detected_emphasis": list of emphasized words/phrases\n'
                                '- "language": detected language code'
                            ),
                        },
                    ],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
            },
        }

    async def _api_call(
        self,
        model: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a Gemini API call with retry logic."""
        url = f"{self._config.base_url}/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self._config.api_key}

        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._http.post(
                    url,
                    json=body,
                    headers=headers,
                    params=params,
                )

                if response.status_code == 429:
                    raise GeminiRateLimitError(
                        "Rate limit exceeded",
                        status_code=429,
                    )

                if response.status_code >= 500:
                    # Server errors are retryable
                    raise GeminiAudioError(
                        f"Gemini API error: {response.status_code}",
                        status_code=response.status_code,
                        details=response.json() if response.content else {},
                    )

                if response.status_code >= 400:
                    # Client errors are not retryable (except 429)
                    raise GeminiAudioError(
                        f"Gemini API error: {response.status_code}",
                        status_code=response.status_code,
                        details=response.json() if response.content else {},
                    )

                return response.json()

            except GeminiAudioError as e:
                # Retry server errors (5xx)
                if e.status_code is not None and e.status_code >= 500:
                    last_error = e
                    if attempt < self._config.max_retries:
                        delay = min(
                            self._config.retry_backoff_base * (2 ** attempt),
                            self._config.retry_backoff_max,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                raise

            except (httpx.TimeoutException, httpx.ConnectError, GeminiRateLimitError) as e:
                last_error = e
                if attempt < self._config.max_retries:
                    delay = min(
                        self._config.retry_backoff_base * (2 ** attempt),
                        self._config.retry_backoff_max,
                    )
                    await asyncio.sleep(delay)
                    continue
                break

        raise GeminiAudioError(
            f"Gemini API call failed after {self._config.max_retries + 1} attempts",
            details={"last_error": str(last_error)},
        )

    def _parse_audio_response(
        self,
        response_data: dict[str, Any],
        *,
        model_used: str,
        processing_time_ms: int,
        audio_duration_ms: int,
    ) -> AudioProcessingResult:
        """Parse Gemini response into AudioProcessingResult."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return AudioProcessingResult(
                    transcript="",
                    model_used=model_used,
                    processing_time_ms=processing_time_ms,
                    audio_duration_ms=audio_duration_ms,
                )

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            raw_text = parts[0].get("text", "") if parts else ""

            # Try to parse as JSON
            try:
                parsed = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, treat as plain transcript
                return AudioProcessingResult(
                    transcript=raw_text.strip(),
                    model_used=model_used,
                    processing_time_ms=processing_time_ms,
                    audio_duration_ms=audio_duration_ms,
                )

            # Build segments from parsed data
            segments = []
            for seg in parsed.get("segments", []):
                segments.append(
                    TranscriptionSegment(
                        text=seg.get("text", ""),
                        start_ms=seg.get("start_ms", 0),
                        end_ms=seg.get("end_ms", 0),
                        confidence=seg.get("confidence", 0.0),
                    )
                )

            return AudioProcessingResult(
                transcript=parsed.get("transcript", raw_text.strip()),
                segments=segments,
                language=parsed.get("language", "en"),
                audio_duration_ms=audio_duration_ms,
                detected_tone=parsed.get("detected_tone"),
                detected_emphasis=parsed.get("detected_emphasis", []),
                model_used=model_used,
                processing_time_ms=processing_time_ms,
            )

        except (KeyError, IndexError, TypeError) as e:
            raise GeminiAudioError(
                f"Failed to parse Gemini response: {e}",
                details={"response": response_data},
            ) from e

    def _parse_json_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Parse a JSON response from Gemini."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return {}

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            raw_text = parts[0].get("text", "") if parts else ""

            return json.loads(raw_text)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise GeminiAudioError(
                f"Failed to parse JSON response: {e}",
                details={"response": response_data},
            ) from e

    def _classification_prompt(self, transcript: str) -> str:
        """Build the classification prompt for Flash-Lite."""
        return f"""Analyze this transcript and classify it. Return a JSON object with exactly these fields:

{{
  "intent": "<one of: task, event, reminder, idea, journal, update, question>",
  "confidence": <float 0.0-1.0>,
  "entities": [
    {{"name": "<entity name>", "type": "<person|place|project|organization>", "metadata": {{}}}}
  ],
  "emotion": {{
    "primary": "<one of: joy, trust, fear, surprise, sadness, disgust, anger, anticipation>",
    "intensity": <int 0-3>,
    "valence": <float -1.0 to 1.0>,
    "arousal": <float 0.0 to 1.0>
  }},
  "acknowledgment": "<brief natural acknowledgment, 2-8 words>"
}}

Transcript: "{transcript}"

Rules:
- Classify intent with >85% confidence when clear
- Extract ALL people, places, projects, organizations mentioned
- Detect emotional tone from language and context
- Acknowledgment should be brief and natural, not chatty
- If uncertain about intent, default to "task" with lower confidence"""

    @staticmethod
    def _default_audio_system_prompt() -> str:
        """Default system prompt for audio processing."""
        return (
            "You are Blurt, an AI second brain that captures and understands "
            "natural speech. Transcribe the audio accurately, preserving the "
            "speaker's exact words. Also analyze tone, emphasis, and emotional "
            "cues from the audio signal. Return structured JSON."
        )

    @staticmethod
    def _estimate_duration_ms(audio_data: bytes, audio_format: AudioFormat) -> int:
        """Estimate audio duration from raw bytes and format."""
        if not audio_data or audio_format.bytes_per_second == 0:
            return 0
        return int(len(audio_data) / audio_format.bytes_per_second * 1000)
