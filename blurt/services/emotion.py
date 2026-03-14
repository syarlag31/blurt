"""Emotion detection service using Gemini Flash-Lite.

Analyzes text input and returns scored Plutchik emotions with intensity
values. Uses the FAST tier (Flash-Lite) for low-latency classification.
Results feed into the knowledge graph, task surfacing, and behavioral
pattern detection.

Design principles:
- Zero friction: runs silently on every blurt, no user action needed.
- Anti-shame: emotions are data, never judged or used to guilt.
- Compounding: emotion history builds behavioral patterns over time.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from blurt.clients.gemini import GeminiClient, GeminiResponse, ModelTier
from blurt.models.emotions import (
    NEUTRAL_EMOTION,
    PLUTCHIK_DYADS,
    EmotionResult,
    EmotionScore,
    PrimaryEmotion,
)

logger = logging.getLogger(__name__)

# Minimum score threshold — emotions below this are filtered out
_MIN_EMOTION_THRESHOLD = 0.05

# System instruction for Gemini Flash-Lite emotion detection
_EMOTION_SYSTEM_INSTRUCTION = """\
You are an emotion detection system. Analyze the given text and detect \
emotions using Plutchik's wheel of emotions.

Return a JSON object with these fields:
- "emotions": an object mapping emotion names to intensity scores (0.0-1.0). \
The 8 primary emotions are: joy, trust, fear, surprise, sadness, disgust, anger, anticipation. \
Only include emotions with score > 0.05. At least one emotion must be included.
- "valence": overall sentiment from -1.0 (very negative) to 1.0 (very positive).
- "arousal": energy/activation level from 0.0 (very calm) to 1.0 (very activated).
- "confidence": your confidence in this analysis from 0.0 to 1.0.

Rules:
- Be nuanced: most text has multiple emotions at different intensities.
- Neutral/factual text should get low-intensity trust or anticipation.
- Do NOT over-detect. Mundane statements should have low intensities.
- Scores should sum-normalize: the highest-scoring emotion rarely exceeds 0.9.

Respond with ONLY valid JSON. No markdown, no explanation.\
"""

# Valence weights for computing valence from emotion scores
_VALENCE_WEIGHTS: dict[PrimaryEmotion, float] = {
    PrimaryEmotion.JOY: 1.0,
    PrimaryEmotion.TRUST: 0.6,
    PrimaryEmotion.ANTICIPATION: 0.3,
    PrimaryEmotion.SURPRISE: 0.0,
    PrimaryEmotion.FEAR: -0.6,
    PrimaryEmotion.SADNESS: -0.8,
    PrimaryEmotion.DISGUST: -0.7,
    PrimaryEmotion.ANGER: -0.8,
}

# Arousal weights for computing arousal from emotion scores
_AROUSAL_WEIGHTS: dict[PrimaryEmotion, float] = {
    PrimaryEmotion.JOY: 0.6,
    PrimaryEmotion.TRUST: 0.3,
    PrimaryEmotion.FEAR: 0.8,
    PrimaryEmotion.SURPRISE: 0.9,
    PrimaryEmotion.SADNESS: 0.2,
    PrimaryEmotion.DISGUST: 0.4,
    PrimaryEmotion.ANGER: 0.9,
    PrimaryEmotion.ANTICIPATION: 0.7,
}


def _compute_valence(scores: list[EmotionScore]) -> float:
    """Compute overall valence from emotion scores. Returns [-1.0, 1.0]."""
    if not scores:
        return 0.0
    weighted_sum = sum(
        s.intensity * _VALENCE_WEIGHTS.get(s.emotion, 0.0) for s in scores
    )
    total_weight = sum(s.intensity for s in scores)
    if total_weight == 0:
        return 0.0
    return max(-1.0, min(1.0, weighted_sum / total_weight))


def _compute_arousal(scores: list[EmotionScore]) -> float:
    """Compute overall arousal from emotion scores. Returns [0.0, 1.0]."""
    if not scores:
        return 0.0
    weighted_sum = sum(
        s.intensity * _AROUSAL_WEIGHTS.get(s.emotion, 0.5) for s in scores
    )
    total_weight = sum(s.intensity for s in scores)
    if total_weight == 0:
        return 0.0
    return max(0.0, min(1.0, weighted_sum / total_weight))


def _detect_dyad(
    primary: EmotionScore, secondary: EmotionScore | None
) -> str | None:
    """Check if primary + secondary form a Plutchik dyad."""
    if secondary is None:
        return None
    pair = (primary.emotion, secondary.emotion)
    reverse = (secondary.emotion, primary.emotion)
    return PLUTCHIK_DYADS.get(pair) or PLUTCHIK_DYADS.get(reverse)


def parse_emotion_response(raw_json: dict[str, Any]) -> EmotionResult:
    """Parse a Gemini emotion detection JSON response into an EmotionResult.

    This function is public so it can be tested independently of the API.

    Args:
        raw_json: Parsed JSON dict from Gemini with "emotions", "valence",
            "arousal", and "confidence" fields.

    Returns:
        EmotionResult with scored emotions, valence, arousal, and dyad detection.

    Raises:
        ValueError: If the JSON structure is invalid or missing required fields.
    """
    emotions_dict = raw_json.get("emotions")
    if not isinstance(emotions_dict, dict) or not emotions_dict:
        raise ValueError("Response must contain a non-empty 'emotions' object")

    scores: list[EmotionScore] = []
    for name, intensity in emotions_dict.items():
        try:
            emotion = PrimaryEmotion(name.lower())
        except ValueError:
            logger.warning("Unknown emotion '%s' in response, skipping", name)
            continue

        try:
            intensity_val = float(intensity)
        except (TypeError, ValueError):
            logger.warning("Invalid intensity for '%s': %s", name, intensity)
            continue

        # Clamp to valid range
        intensity_val = max(0.0, min(1.0, intensity_val))
        if intensity_val >= _MIN_EMOTION_THRESHOLD:
            scores.append(EmotionScore(emotion=emotion, intensity=intensity_val))

    if not scores:
        return NEUTRAL_EMOTION

    # Sort by intensity descending
    scores.sort(key=lambda s: s.intensity, reverse=True)

    primary = scores[0]
    secondary = scores[1] if len(scores) > 1 else None

    # Use model-provided valence/arousal if available, otherwise compute
    valence = raw_json.get("valence")
    if valence is not None:
        try:
            valence = max(-1.0, min(1.0, float(valence)))
        except (TypeError, ValueError):
            valence = _compute_valence(scores)
    else:
        valence = _compute_valence(scores)

    arousal = raw_json.get("arousal")
    if arousal is not None:
        try:
            arousal = max(0.0, min(1.0, float(arousal)))
        except (TypeError, ValueError):
            arousal = _compute_arousal(scores)
    else:
        arousal = _compute_arousal(scores)

    confidence = raw_json.get("confidence", 0.8)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.8

    composite_label = _detect_dyad(primary, secondary)

    return EmotionResult(
        primary=primary,
        secondary=secondary,
        valence=valence,
        arousal=arousal,
        confidence=confidence,
        composite_label=composite_label,
    )


class EmotionDetectionService:
    """Detects emotions from text using Gemini Flash-Lite.

    Uses the two-model strategy: Flash-Lite (FAST tier) handles emotion
    detection since it's a classification task — fast and cheap.

    Usage::

        async with GeminiClient(config) as client:
            service = EmotionDetectionService(client)
            result = await service.detect("I'm so excited about this!")
            print(result.primary.emotion)  # PrimaryEmotion.JOY
            print(result.primary.intensity_label)  # "ecstasy"
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        *,
        temperature: float = 0.1,
        min_threshold: float = _MIN_EMOTION_THRESHOLD,
    ) -> None:
        """Initialize the emotion detection service.

        Args:
            gemini_client: Connected GeminiClient instance.
            temperature: Model temperature for emotion detection (low = deterministic).
            min_threshold: Minimum intensity to include an emotion in results.
        """
        self._client = gemini_client
        self._temperature = temperature
        self._min_threshold = min_threshold
        self._call_count = 0
        self._total_latency_ms = 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Service usage statistics."""
        return {
            "call_count": self._call_count,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._call_count
                if self._call_count > 0
                else 0.0
            ),
        }

    async def detect(self, text: str) -> EmotionResult:
        """Detect emotions in the given text.

        Sends the text to Gemini Flash-Lite for emotion analysis and returns
        a structured EmotionResult with scored Plutchik emotions.

        Args:
            text: The text to analyze for emotions.

        Returns:
            EmotionResult with primary/secondary emotions, valence, arousal.
            Returns NEUTRAL_EMOTION for empty or whitespace-only input.
        """
        if not text or not text.strip():
            logger.debug("Empty input, returning neutral emotion")
            return NEUTRAL_EMOTION

        start = time.monotonic()

        try:
            response = await self._client.generate(
                prompt=text,
                tier=ModelTier.FAST,
                system_instruction=_EMOTION_SYSTEM_INSTRUCTION,
                temperature=self._temperature,
                max_output_tokens=512,
                response_mime_type="application/json",
            )

            raw_json = self._parse_json_response(response)
            result = parse_emotion_response(raw_json)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse emotion response JSON: %s", e)
            result = NEUTRAL_EMOTION
        except ValueError as e:
            logger.warning("Invalid emotion response structure: %s", e)
            result = NEUTRAL_EMOTION
        except Exception as e:
            logger.error("Emotion detection failed: %s", e, exc_info=True)
            result = NEUTRAL_EMOTION

        latency_ms = (time.monotonic() - start) * 1000
        self._call_count += 1
        self._total_latency_ms += latency_ms

        logger.debug(
            "Emotion detected: %s (%.2f) in %.1fms",
            result.primary.emotion.value,
            result.primary.intensity,
            latency_ms,
        )

        return result

    async def detect_batch(self, texts: list[str]) -> list[EmotionResult]:
        """Detect emotions for multiple texts concurrently.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of EmotionResult objects, one per input text.
        """
        import asyncio

        tasks = [self.detect(text) for text in texts]
        return list(await asyncio.gather(*tasks))

    def _parse_json_response(self, response: GeminiResponse) -> dict[str, Any]:
        """Extract and parse JSON from a Gemini response.

        Handles responses that may be wrapped in markdown code blocks.
        """
        text = response.text.strip()

        # Strip markdown code block if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        return json.loads(text)  # type: ignore[no-any-return]
