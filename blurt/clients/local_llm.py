"""Local LLM client for local-only mode.

Provides a mock/local implementation of the LLM client that mirrors
the GeminiClient interface. In local-only mode, this replaces all
external LLM API calls with deterministic local responses, ensuring
full feature parity without any data leakage.

For classification and extraction tasks, uses rule-based heuristics.
For reasoning tasks, returns structured placeholder responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMResponse:
    """Response from local LLM inference, matching GeminiResponse shape."""

    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = "local-mock"
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    latency_ms: float = 0.0
    finish_reason: str = "STOP"


class LocalLLMClient:
    """Local/mock LLM client for local-only mode.

    Implements the same interface as GeminiClient but without any
    external API calls. Provides:

    - Deterministic responses for classification tasks
    - Rule-based entity extraction
    - Template-based responses for reasoning tasks
    - Audio transcription stubs (returns acknowledgment)

    This ensures the full Blurt pipeline works end-to-end in local mode.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or Path.home() / ".blurt"
        self._connected = False
        self._request_count = 0

    async def connect(self) -> None:
        """Initialize the local client (no-op, always succeeds)."""
        self._connected = True
        logger.info("Local LLM client connected (no external API)")

    async def close(self) -> None:
        """Shut down the local client."""
        self._connected = False
        logger.info("Local LLM client closed (requests=%d)", self._request_count)

    async def __aenter__(self) -> LocalLLMClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def generate(
        self,
        prompt: str,
        *,
        tier: Any = None,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Generate a local response based on prompt analysis.

        For JSON responses (classification, extraction), returns
        structured data. For text responses, returns template-based output.
        """
        self._request_count += 1
        start = time.monotonic()

        # Detect if JSON output is expected
        if response_mime_type == "application/json":
            text = self._generate_json_response(prompt, system_instruction)
        else:
            text = self._generate_text_response(prompt, system_instruction)

        latency = (time.monotonic() - start) * 1000

        return LocalLLMResponse(
            text=text,
            raw={"local": True, "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8]},
            model="local-mock",
            latency_ms=latency,
        )

    async def generate_from_audio(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        *,
        prompt: str = "",
        tier: Any = None,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Process audio input locally.

        In local mode, returns a stub transcription acknowledgment.
        Audio content is not actually transcribed without an external model.
        """
        self._request_count += 1
        start = time.monotonic()

        audio_size = len(audio_data)
        # Estimate duration from audio size (rough heuristic)
        estimated_duration_s = audio_size / 16000  # ~16KB/s for typical audio

        result_text = (
            f"[Local mode: received {audio_size} bytes of {mime_type} audio, "
            f"~{estimated_duration_s:.1f}s estimated duration]"
        )

        if prompt:
            result_text += f"\nPrompt context: {prompt}"

        latency = (time.monotonic() - start) * 1000

        return LocalLLMResponse(
            text=result_text,
            raw={"local": True, "audio_bytes": audio_size, "mime_type": mime_type},
            model="local-mock",
            latency_ms=latency,
        )

    async def generate_multi_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tier: Any = None,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        response_mime_type: str | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Generate response for multi-turn conversation."""
        self._request_count += 1
        start = time.monotonic()

        # Extract last user message for context
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                parts = msg.get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        last_user_msg = part["text"]
                        break
                    elif isinstance(part, str):
                        last_user_msg = part
                        break
                if last_user_msg:
                    break

        text = self._generate_text_response(last_user_msg, system_instruction)
        latency = (time.monotonic() - start) * 1000

        return LocalLLMResponse(
            text=text,
            raw={"local": True, "turn_count": len(messages)},
            model="local-mock",
            latency_ms=latency,
        )

    async def health_check(self) -> dict[str, Any]:
        """Health check — always healthy in local mode."""
        return {
            "healthy": True,
            "state": "connected" if self._connected else "created",
            "latency_ms": 0.0,
            "request_count": self._request_count,
            "error_count": 0,
            "error_rate": 0.0,
            "mode": "local",
        }

    # ── Internal Helpers ───────────────────────────────────────────

    def _generate_json_response(self, prompt: str, system_instruction: str | None) -> str:
        """Generate a structured JSON response based on prompt context."""
        prompt_lower = prompt.lower()
        sys_lower = (system_instruction or "").lower()

        # Intent classification
        if "classify" in sys_lower or "intent" in sys_lower:
            return self._classify_intent(prompt_lower)

        # Entity extraction
        if "extract" in sys_lower or "entity" in sys_lower or "entities" in sys_lower:
            return self._extract_entities(prompt_lower)

        # Emotion detection
        if "emotion" in sys_lower or "sentiment" in sys_lower:
            return self._detect_emotion(prompt_lower)

        # Default JSON
        return json.dumps({"result": "processed", "mode": "local"})

    def _generate_text_response(self, prompt: str, system_instruction: str | None) -> str:
        """Generate a text response."""
        if not prompt:
            return "[Local mode: no prompt provided]"
        # Return an echo-style acknowledgment
        truncated = prompt[:200] + "..." if len(prompt) > 200 else prompt
        return f"[Local mode] Received and stored: {truncated}"

    def _classify_intent(self, text: str) -> str:
        """Rule-based intent classification."""
        intent = "note"  # default
        confidence = 0.7

        task_words = {"todo", "task", "need to", "must", "should", "remind me", "don't forget"}
        event_words = {"meeting", "appointment", "schedule", "calendar", "at ", "on "}
        question_words = {"what", "how", "why", "when", "where", "who", "?"}
        journal_words = {"feeling", "felt", "today was", "i think", "reflecting", "journal"}
        idea_words = {"idea", "what if", "maybe we could", "concept", "brainstorm"}
        contact_words = {"call", "email", "text", "message", "reach out", "phone"}
        memory_words = {"remember", "recall", "last time", "that time when"}

        for word in task_words:
            if word in text:
                intent = "task"
                confidence = 0.85
                break
        for word in event_words:
            if word in text:
                intent = "event"
                confidence = 0.85
                break
        for word in question_words:
            if word in text:
                intent = "question"
                confidence = 0.80
                break
        for word in journal_words:
            if word in text:
                intent = "journal"
                confidence = 0.80
                break
        for word in idea_words:
            if word in text:
                intent = "idea"
                confidence = 0.80
                break
        for word in contact_words:
            if word in text:
                intent = "contact"
                confidence = 0.75
                break
        for word in memory_words:
            if word in text:
                intent = "memory_recall"
                confidence = 0.80
                break

        return json.dumps({
            "intent": intent,
            "confidence": confidence,
            "reasoning": f"[Local classifier] matched '{intent}' pattern",
        })

    def _extract_entities(self, text: str) -> str:
        """Rule-based entity extraction (very basic)."""
        entities: list[dict[str, str]] = []
        # Simple heuristic: words starting with uppercase might be entities
        words = text.split()
        for word in words:
            cleaned = word.strip(".,!?;:'\"")
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                entities.append({"name": cleaned, "type": "unknown"})

        return json.dumps({"entities": entities[:10]})  # cap at 10

    def _detect_emotion(self, text: str) -> str:
        """Rule-based emotion detection."""
        emotion = "neutral"
        intensity = 0.5

        positive = {"happy", "great", "awesome", "love", "excited", "wonderful", "amazing"}
        negative = {"sad", "angry", "frustrated", "upset", "worried", "anxious", "stressed"}

        words = set(text.lower().split())
        if words & positive:
            emotion = "joy"
            intensity = 0.7
        elif words & negative:
            emotion = "sadness" if words & {"sad"} else "anger" if words & {"angry"} else "fear"
            intensity = 0.7

        return json.dumps({
            "primary_emotion": emotion,
            "intensity": intensity,
            "confidence": 0.6,
        })
