"""Tests for the emotion detection service.

Tests cover:
- Plutchik emotion model (EmotionScore, EmotionResult)
- JSON response parsing (parse_emotion_response)
- EmotionDetectionService with mocked Gemini client
- Edge cases: empty input, malformed JSON, invalid emotions
- Valence/arousal computation
- Dyad detection
- Batch detection
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.models.emotions import (
    EMOTION_INTENSITY_LABELS,
    NEUTRAL_EMOTION,
    EmotionResult,
    EmotionScore,
    PrimaryEmotion,
)
from blurt.services.emotion import (
    EmotionDetectionService,
    _compute_arousal,
    _compute_valence,
    parse_emotion_response,
)


# ── EmotionScore model tests ──────────────────────────────────────────


class TestEmotionScore:
    def test_create_valid_score(self):
        score = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8)
        assert score.emotion == PrimaryEmotion.JOY
        assert score.intensity == 0.8

    def test_intensity_validation_low(self):
        with pytest.raises(ValueError, match="[Ii]ntensity"):
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=-0.1)

    def test_intensity_validation_high(self):
        with pytest.raises(ValueError, match="[Ii]ntensity"):
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=1.1)

    def test_intensity_label_low(self):
        score = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.2)
        assert score.intensity_label == "serenity"

    def test_intensity_label_mid(self):
        score = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5)
        assert score.intensity_label == "joy"

    def test_intensity_label_high(self):
        score = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.9)
        assert score.intensity_label == "ecstasy"

    def test_all_emotions_have_labels(self):
        """Every primary emotion should have intensity labels."""
        for emotion in PrimaryEmotion:
            assert emotion in EMOTION_INTENSITY_LABELS
            labels = EMOTION_INTENSITY_LABELS[emotion]
            assert len(labels) == 3

    @pytest.mark.parametrize(
        "emotion,high_label",
        [
            (PrimaryEmotion.JOY, "ecstasy"),
            (PrimaryEmotion.TRUST, "admiration"),
            (PrimaryEmotion.FEAR, "terror"),
            (PrimaryEmotion.SURPRISE, "amazement"),
            (PrimaryEmotion.SADNESS, "grief"),
            (PrimaryEmotion.DISGUST, "loathing"),
            (PrimaryEmotion.ANGER, "rage"),
            (PrimaryEmotion.ANTICIPATION, "vigilance"),
        ],
    )
    def test_high_intensity_labels(self, emotion: PrimaryEmotion, high_label: str):
        score = EmotionScore(emotion=emotion, intensity=0.9)
        assert score.intensity_label == high_label


# ── EmotionResult model tests ─────────────────────────────────────────


class TestEmotionResult:
    def test_create_result_primary_only(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7)
        result = EmotionResult(primary=primary, valence=0.8, arousal=0.6)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.secondary is None
        assert result.dominant_label == "ecstasy"

    def test_create_result_with_secondary(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7)
        secondary = EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5)
        result = EmotionResult(primary=primary, secondary=secondary)
        assert result.secondary is not None

    def test_valence_validation(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5)
        with pytest.raises(ValueError, match="[Vv]alence"):
            EmotionResult(primary=primary, valence=1.5)

    def test_arousal_validation(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5)
        with pytest.raises(ValueError, match="[Aa]rousal"):
            EmotionResult(primary=primary, arousal=-0.1)

    def test_confidence_validation(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5)
        with pytest.raises(ValueError, match="[Cc]onfidence"):
            EmotionResult(primary=primary, confidence=1.5)

    def test_detected_dyad_joy_trust(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8)
        secondary = EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.6)
        result = EmotionResult(primary=primary, secondary=secondary)
        assert result.detected_dyad() == "love"

    def test_detected_dyad_reverse_order(self):
        primary = EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.8)
        secondary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.6)
        result = EmotionResult(primary=primary, secondary=secondary)
        assert result.detected_dyad() == "love"

    def test_no_dyad_without_secondary(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8)
        result = EmotionResult(primary=primary)
        assert result.detected_dyad() is None

    def test_explicit_composite_label(self):
        primary = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8)
        result = EmotionResult(primary=primary, composite_label="custom")
        assert result.detected_dyad() == "custom"

    def test_neutral_emotion_is_valid(self):
        assert NEUTRAL_EMOTION.primary.emotion == PrimaryEmotion.TRUST
        assert NEUTRAL_EMOTION.confidence == 0.5


# ── Valence/Arousal computation tests ─────────────────────────────────


class TestValenceArousal:
    def test_valence_positive_emotions(self):
        scores = [
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8),
            EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
        ]
        valence = _compute_valence(scores)
        assert valence > 0.5

    def test_valence_negative_emotions(self):
        scores = [
            EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.7),
            EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.5),
        ]
        valence = _compute_valence(scores)
        assert valence < -0.5

    def test_valence_empty_scores(self):
        assert _compute_valence([]) == 0.0

    def test_valence_clamped(self):
        scores = [
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=1.0),
        ]
        valence = _compute_valence(scores)
        assert -1.0 <= valence <= 1.0

    def test_arousal_high_energy_emotions(self):
        scores = [
            EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.9),
            EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=0.7),
        ]
        arousal = _compute_arousal(scores)
        assert arousal > 0.7

    def test_arousal_low_energy_emotions(self):
        scores = [
            EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.8),
        ]
        arousal = _compute_arousal(scores)
        assert arousal < 0.4

    def test_arousal_empty_scores(self):
        assert _compute_arousal([]) == 0.0

    def test_arousal_clamped(self):
        scores = [
            EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=1.0),
        ]
        arousal = _compute_arousal(scores)
        assert 0.0 <= arousal <= 1.0


# ── parse_emotion_response tests ──────────────────────────────────────


class TestParseEmotionResponse:
    def test_parse_basic_response(self):
        raw = {
            "emotions": {"joy": 0.8, "trust": 0.4},
            "valence": 0.7,
            "arousal": 0.5,
            "confidence": 0.9,
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.primary.intensity == 0.8
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.TRUST
        assert result.valence == 0.7
        assert result.arousal == 0.5
        assert result.confidence == 0.9

    def test_parse_single_emotion(self):
        raw = {
            "emotions": {"anger": 0.6},
            "valence": -0.5,
            "arousal": 0.8,
            "confidence": 0.85,
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.ANGER
        assert result.secondary is None

    def test_parse_all_eight_emotions(self):
        raw = {
            "emotions": {
                "joy": 0.3,
                "trust": 0.2,
                "fear": 0.1,
                "surprise": 0.4,
                "sadness": 0.15,
                "disgust": 0.08,
                "anger": 0.12,
                "anticipation": 0.5,
            }
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.ANTICIPATION
        assert result.primary.intensity == 0.5

    def test_parse_filters_low_scores(self):
        raw = {
            "emotions": {"joy": 0.8, "fear": 0.01},  # below threshold
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.secondary is None  # fear was filtered

    def test_parse_clamps_intensity(self):
        raw = {
            "emotions": {"joy": 1.5, "anger": -0.2},
        }
        result = parse_emotion_response(raw)
        assert result.primary.intensity == 1.0  # clamped from 1.5
        # anger at -0.2 clamped to 0.0, below threshold, filtered

    def test_parse_unknown_emotion_ignored(self):
        raw = {
            "emotions": {"joy": 0.7, "confusion": 0.5},  # not a Plutchik emotion
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.secondary is None  # "confusion" skipped

    def test_parse_case_insensitive(self):
        raw = {
            "emotions": {"JOY": 0.6, "Trust": 0.4},
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY

    def test_parse_missing_emotions_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_emotion_response({})

    def test_parse_empty_emotions_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_emotion_response({"emotions": {}})

    def test_parse_returns_neutral_on_all_below_threshold(self):
        raw = {
            "emotions": {"joy": 0.01, "trust": 0.02},
        }
        result = parse_emotion_response(raw)
        assert result == NEUTRAL_EMOTION

    def test_parse_computes_valence_when_missing(self):
        raw = {
            "emotions": {"joy": 0.8},
        }
        result = parse_emotion_response(raw)
        assert result.valence > 0  # joy has positive valence

    def test_parse_computes_arousal_when_missing(self):
        raw = {
            "emotions": {"anger": 0.9},
        }
        result = parse_emotion_response(raw)
        assert result.arousal > 0.5  # anger has high arousal

    def test_parse_detects_dyad(self):
        raw = {
            "emotions": {"joy": 0.8, "trust": 0.6},
        }
        result = parse_emotion_response(raw)
        assert result.composite_label == "love"

    def test_parse_invalid_intensity_skipped(self):
        raw = {
            "emotions": {"joy": "high", "trust": 0.5},  # "high" is not a float
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.TRUST

    def test_parse_invalid_valence_fallback(self):
        raw = {
            "emotions": {"joy": 0.7},
            "valence": "positive",  # not a float
        }
        result = parse_emotion_response(raw)
        # Should fall back to computed valence
        assert -1.0 <= result.valence <= 1.0

    def test_parse_invalid_confidence_fallback(self):
        raw = {
            "emotions": {"joy": 0.7},
            "confidence": "high",
        }
        result = parse_emotion_response(raw)
        assert result.confidence == 0.8  # default fallback


# ── EmotionDetectionService tests ─────────────────────────────────────


def _make_gemini_response(emotions_json: dict[str, Any]) -> MagicMock:
    """Create a mock GeminiResponse with the given JSON payload."""
    mock = MagicMock()
    mock.text = json.dumps(emotions_json)
    mock.model = "gemini-2.5-flash-lite"
    mock.latency_ms = 50.0
    return mock


def _make_mock_client(response_json: dict[str, Any] | None = None) -> AsyncMock:
    """Create a mock GeminiClient that returns the given response."""
    client = AsyncMock()
    if response_json is not None:
        client.generate.return_value = _make_gemini_response(response_json)
    return client


class TestEmotionDetectionService:
    @pytest.mark.asyncio
    async def test_detect_happy_text(self):
        response_json = {
            "emotions": {"joy": 0.85, "anticipation": 0.3},
            "valence": 0.8,
            "arousal": 0.6,
            "confidence": 0.92,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("I'm so excited about this project!")

        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.primary.intensity == 0.85
        assert result.valence == 0.8
        assert result.confidence == 0.92
        client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_angry_text(self):
        response_json = {
            "emotions": {"anger": 0.7, "disgust": 0.3},
            "valence": -0.7,
            "arousal": 0.85,
            "confidence": 0.88,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("This is absolutely unacceptable!")

        assert result.primary.emotion == PrimaryEmotion.ANGER
        assert result.valence < 0
        assert result.arousal > 0.5

    @pytest.mark.asyncio
    async def test_detect_mixed_emotions(self):
        response_json = {
            "emotions": {"sadness": 0.6, "joy": 0.3, "trust": 0.2},
            "valence": -0.2,
            "arousal": 0.3,
            "confidence": 0.85,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("Bittersweet goodbye to an old friend.")

        assert result.primary.emotion == PrimaryEmotion.SADNESS
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.JOY

    @pytest.mark.asyncio
    async def test_detect_empty_input(self):
        client = _make_mock_client()
        service = EmotionDetectionService(client)

        result = await service.detect("")

        assert result == NEUTRAL_EMOTION
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_whitespace_input(self):
        client = _make_mock_client()
        service = EmotionDetectionService(client)

        result = await service.detect("   \n\t  ")

        assert result == NEUTRAL_EMOTION
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_malformed_json_returns_neutral(self):
        client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "not valid json at all"
        client.generate.return_value = mock_response

        service = EmotionDetectionService(client)
        result = await service.detect("Some text")

        assert result == NEUTRAL_EMOTION

    @pytest.mark.asyncio
    async def test_detect_markdown_wrapped_json(self):
        client = AsyncMock()
        mock_response = MagicMock()
        json_body = json.dumps({
            "emotions": {"joy": 0.7},
            "valence": 0.6,
            "arousal": 0.5,
            "confidence": 0.9,
        })
        mock_response.text = f"```json\n{json_body}\n```"
        client.generate.return_value = mock_response

        service = EmotionDetectionService(client)
        result = await service.detect("Great news today!")

        assert result.primary.emotion == PrimaryEmotion.JOY

    @pytest.mark.asyncio
    async def test_detect_api_error_returns_neutral(self):
        client = AsyncMock()
        client.generate.side_effect = RuntimeError("API unavailable")

        service = EmotionDetectionService(client)
        result = await service.detect("Some text")

        assert result == NEUTRAL_EMOTION

    @pytest.mark.asyncio
    async def test_detect_uses_fast_tier(self):
        response_json = {
            "emotions": {"trust": 0.5},
            "valence": 0.3,
            "arousal": 0.3,
            "confidence": 0.8,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        await service.detect("Normal day at work.")

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["tier"] == ModelTier.FAST

    @pytest.mark.asyncio
    async def test_detect_uses_json_mime_type(self):
        response_json = {
            "emotions": {"trust": 0.5},
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        await service.detect("Test text")

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["response_mime_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        response_json = {
            "emotions": {"joy": 0.6},
            "valence": 0.5,
            "arousal": 0.5,
            "confidence": 0.9,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        assert service.stats["call_count"] == 0
        await service.detect("Happy!")
        assert service.stats["call_count"] == 1
        await service.detect("Also happy!")
        assert service.stats["call_count"] == 2

    @pytest.mark.asyncio
    async def test_detect_batch(self):
        response_json = {
            "emotions": {"joy": 0.7},
            "valence": 0.6,
            "arousal": 0.5,
            "confidence": 0.9,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        results = await service.detect_batch(["Text 1", "Text 2", "Text 3"])

        assert len(results) == 3
        assert all(r.primary.emotion == PrimaryEmotion.JOY for r in results)
        assert client.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_detect_batch_empty_list(self):
        client = _make_mock_client()
        service = EmotionDetectionService(client)

        results = await service.detect_batch([])

        assert results == []


# Need this import for the tier assertion
from blurt.clients.gemini import ModelTier  # noqa: E402


# ── Dyad detection tests ─────────────────────────────────────────────


class TestDyadDetection:
    @pytest.mark.parametrize(
        "e1,e2,expected_dyad",
        [
            (PrimaryEmotion.JOY, PrimaryEmotion.TRUST, "love"),
            (PrimaryEmotion.TRUST, PrimaryEmotion.FEAR, "submission"),
            (PrimaryEmotion.FEAR, PrimaryEmotion.SURPRISE, "awe"),
            (PrimaryEmotion.SURPRISE, PrimaryEmotion.SADNESS, "disapproval"),
            (PrimaryEmotion.SADNESS, PrimaryEmotion.DISGUST, "remorse"),
            (PrimaryEmotion.DISGUST, PrimaryEmotion.ANGER, "contempt"),
            (PrimaryEmotion.ANGER, PrimaryEmotion.ANTICIPATION, "aggressiveness"),
            (PrimaryEmotion.ANTICIPATION, PrimaryEmotion.JOY, "optimism"),
        ],
    )
    def test_all_plutchik_dyads(
        self,
        e1: PrimaryEmotion,
        e2: PrimaryEmotion,
        expected_dyad: str,
    ):
        raw = {
            "emotions": {e1.value: 0.8, e2.value: 0.6},
        }
        result = parse_emotion_response(raw)
        assert result.composite_label == expected_dyad


# ── All 8 primary emotions through detection service ─────────────────


class TestDetectAll8Emotions:
    """Verify EmotionDetectionService detects each of the 8 primary emotions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "emotion_name,text",
        [
            ("joy", "I'm absolutely thrilled about the promotion!"),
            ("trust", "I know I can count on you, you've always been reliable."),
            ("fear", "I'm terrified of what might happen tomorrow."),
            ("surprise", "I can't believe this happened, I had no idea!"),
            ("sadness", "I miss my grandmother so much, she's been gone a year."),
            ("disgust", "That behavior is absolutely revolting and unacceptable."),
            ("anger", "I'm furious about the way they treated us!"),
            ("anticipation", "I can't wait to see what happens next, so eager!"),
        ],
    )
    async def test_detect_each_primary_emotion(self, emotion_name: str, text: str):
        """Each of the 8 primary emotions should be correctly returned as primary."""
        expected_emotion = PrimaryEmotion(emotion_name)
        response_json = {
            "emotions": {emotion_name: 0.75},
            "valence": 0.5 if emotion_name in ("joy", "trust", "anticipation") else -0.5,
            "arousal": 0.6,
            "confidence": 0.9,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect(text)

        assert result.primary.emotion == expected_emotion
        assert result.primary.intensity == 0.75
        assert result.confidence == 0.9
        client.generate.assert_called_once()


# ── Intensity scoring accuracy tests ─────────────────────────────────


class TestIntensityScoringAccuracy:
    """Verify that intensity scores are preserved accurately through parsing."""

    @pytest.mark.parametrize(
        "intensity",
        [0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.75, 0.9, 0.99, 1.0],
    )
    def test_intensity_preserved_through_parse(self, intensity: float):
        """Intensity values should pass through parse_emotion_response unchanged."""
        raw = {"emotions": {"joy": intensity}}
        result = parse_emotion_response(raw)
        assert result.primary.intensity == pytest.approx(intensity)

    @pytest.mark.parametrize("emotion", list(PrimaryEmotion))
    def test_each_emotion_intensity_preserved(self, emotion: PrimaryEmotion):
        """Every primary emotion should have its intensity preserved at 0.72."""
        raw = {"emotions": {emotion.value: 0.72}}
        result = parse_emotion_response(raw)
        assert result.primary.emotion == emotion
        assert result.primary.intensity == pytest.approx(0.72)

    def test_multiple_emotions_sorted_by_intensity(self):
        """Emotions should be sorted descending by intensity."""
        raw = {
            "emotions": {
                "trust": 0.3,
                "joy": 0.9,
                "fear": 0.1,
                "anticipation": 0.6,
            }
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.primary.intensity == 0.9
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.ANTICIPATION
        assert result.secondary.intensity == 0.6

    def test_intensity_at_exact_threshold(self):
        """Intensity at exactly the minimum threshold (0.05) should be kept."""
        raw = {"emotions": {"sadness": 0.05}}
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.SADNESS
        assert result.primary.intensity == pytest.approx(0.05)

    def test_intensity_just_below_threshold_filtered(self):
        """Intensity below 0.05 should be filtered out."""
        raw = {"emotions": {"sadness": 0.049, "joy": 0.6}}
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.secondary is None  # sadness filtered

    @pytest.mark.parametrize(
        "emotion,intensity,expected_label",
        [
            (PrimaryEmotion.JOY, 0.1, "serenity"),
            (PrimaryEmotion.JOY, 0.5, "joy"),
            (PrimaryEmotion.JOY, 0.9, "ecstasy"),
            (PrimaryEmotion.ANGER, 0.2, "annoyance"),
            (PrimaryEmotion.ANGER, 0.5, "anger"),
            (PrimaryEmotion.ANGER, 0.8, "rage"),
            (PrimaryEmotion.FEAR, 0.1, "apprehension"),
            (PrimaryEmotion.FEAR, 0.5, "fear"),
            (PrimaryEmotion.FEAR, 0.9, "terror"),
            (PrimaryEmotion.SADNESS, 0.15, "pensiveness"),
            (PrimaryEmotion.SADNESS, 0.5, "sadness"),
            (PrimaryEmotion.SADNESS, 0.7, "grief"),
            (PrimaryEmotion.TRUST, 0.1, "acceptance"),
            (PrimaryEmotion.TRUST, 0.5, "trust"),
            (PrimaryEmotion.TRUST, 0.9, "admiration"),
            (PrimaryEmotion.SURPRISE, 0.2, "distraction"),
            (PrimaryEmotion.SURPRISE, 0.5, "surprise"),
            (PrimaryEmotion.SURPRISE, 0.8, "amazement"),
            (PrimaryEmotion.DISGUST, 0.1, "boredom"),
            (PrimaryEmotion.DISGUST, 0.5, "disgust"),
            (PrimaryEmotion.DISGUST, 0.9, "loathing"),
            (PrimaryEmotion.ANTICIPATION, 0.2, "interest"),
            (PrimaryEmotion.ANTICIPATION, 0.5, "anticipation"),
            (PrimaryEmotion.ANTICIPATION, 0.8, "vigilance"),
        ],
    )
    def test_intensity_label_accuracy_through_parse(
        self, emotion: PrimaryEmotion, intensity: float, expected_label: str
    ):
        """Intensity labels should be accurate for all 8 emotions at all 3 tiers."""
        raw = {"emotions": {emotion.value: intensity}}
        result = parse_emotion_response(raw)
        assert result.primary.intensity_label == expected_label


# ── Edge cases: neutral/ambiguous text, mixed emotions ───────────────


class TestNeutralAndAmbiguousEdgeCases:
    """Edge cases for neutral, ambiguous, and boundary inputs."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_neutral(self):
        client = _make_mock_client()
        service = EmotionDetectionService(client)
        result = await service.detect("")
        assert result == NEUTRAL_EMOTION
        client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_like_whitespace_returns_neutral(self):
        client = _make_mock_client()
        service = EmotionDetectionService(client)
        result = await service.detect("   \t\n   ")
        assert result == NEUTRAL_EMOTION

    @pytest.mark.asyncio
    async def test_neutral_factual_text(self):
        """Factual, emotionless text should produce low-intensity trust/anticipation."""
        response_json = {
            "emotions": {"trust": 0.2, "anticipation": 0.1},
            "valence": 0.05,
            "arousal": 0.2,
            "confidence": 0.7,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("The meeting is at 3pm in room B.")

        assert result.primary.intensity < 0.33  # low intensity
        assert result.primary.intensity_label in ("acceptance", "interest")
        assert abs(result.valence) < 0.3  # near neutral valence

    @pytest.mark.asyncio
    async def test_ambiguous_text_low_confidence(self):
        """Ambiguous text should produce a low confidence score."""
        response_json = {
            "emotions": {"surprise": 0.3, "anticipation": 0.25, "trust": 0.2},
            "valence": 0.1,
            "arousal": 0.4,
            "confidence": 0.4,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("Well, that happened.")

        assert result.confidence < 0.5
        assert result.primary.intensity < 0.5

    def test_parse_all_emotions_below_threshold_returns_neutral(self):
        """If all emotions are below the threshold, return NEUTRAL."""
        raw = {
            "emotions": {"joy": 0.01, "trust": 0.02, "fear": 0.03, "anger": 0.04},
        }
        result = parse_emotion_response(raw)
        assert result == NEUTRAL_EMOTION

    def test_parse_single_emotion_at_threshold_not_neutral(self):
        """A single emotion exactly at threshold should NOT be neutral."""
        raw = {"emotions": {"joy": 0.05}}
        result = parse_emotion_response(raw)
        assert result != NEUTRAL_EMOTION
        assert result.primary.emotion == PrimaryEmotion.JOY

    def test_parse_emotions_none_value_raises(self):
        """None as emotions value should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_emotion_response({"emotions": None})

    def test_parse_emotions_list_instead_of_dict_raises(self):
        """A list instead of dict for emotions should raise."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_emotion_response({"emotions": ["joy", 0.5]})


class TestMixedEmotionEdgeCases:
    """Edge cases for mixed/blended emotion inputs through the service."""

    @pytest.mark.asyncio
    async def test_mixed_positive_negative_bittersweet(self):
        """Bittersweet: primary sadness + secondary joy."""
        response_json = {
            "emotions": {"sadness": 0.6, "joy": 0.4},
            "valence": -0.1,
            "arousal": 0.3,
            "confidence": 0.85,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("Bittersweet memories of our time together.")

        assert result.primary.emotion == PrimaryEmotion.SADNESS
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.JOY
        # No dyad for sadness+joy (they're opposite, not adjacent)
        assert result.composite_label is None

    @pytest.mark.asyncio
    async def test_mixed_emotions_three_way(self):
        """Three emotions: only top two become primary/secondary."""
        response_json = {
            "emotions": {"anger": 0.7, "disgust": 0.5, "sadness": 0.3},
            "valence": -0.7,
            "arousal": 0.7,
            "confidence": 0.88,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("That was disgusting and infuriating, I'm devastated.")

        assert result.primary.emotion == PrimaryEmotion.ANGER
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.DISGUST
        assert result.composite_label == "contempt"  # anger+disgust dyad

    @pytest.mark.asyncio
    async def test_equal_intensity_picks_first_sorted(self):
        """When two emotions have equal intensity, parse should deterministically pick one."""
        response_json = {
            "emotions": {"joy": 0.5, "trust": 0.5},
            "valence": 0.5,
            "arousal": 0.4,
            "confidence": 0.8,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("I feel good and comfortable.")

        # Both at 0.5, but one must be primary
        assert result.primary.intensity == 0.5
        assert result.secondary is not None
        assert result.secondary.intensity == 0.5
        # Joy+Trust = love dyad
        assert result.composite_label == "love"

    @pytest.mark.asyncio
    async def test_all_8_emotions_present_picks_strongest(self):
        """When all 8 emotions are present, the highest becomes primary."""
        response_json = {
            "emotions": {
                "joy": 0.2,
                "trust": 0.15,
                "fear": 0.1,
                "surprise": 0.9,
                "sadness": 0.05,
                "disgust": 0.08,
                "anger": 0.12,
                "anticipation": 0.3,
            },
            "valence": 0.0,
            "arousal": 0.7,
            "confidence": 0.75,
        }
        client = _make_mock_client(response_json)
        service = EmotionDetectionService(client)

        result = await service.detect("What on earth just happened?!")

        assert result.primary.emotion == PrimaryEmotion.SURPRISE
        assert result.primary.intensity == 0.9
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.ANTICIPATION


# ── Valence/arousal per-emotion accuracy ─────────────────────────────


class TestValenceArousalPerEmotion:
    """Verify valence/arousal computation produces correct signs for each emotion."""

    @pytest.mark.parametrize(
        "emotion,expected_sign",
        [
            (PrimaryEmotion.JOY, "positive"),
            (PrimaryEmotion.TRUST, "positive"),
            (PrimaryEmotion.ANTICIPATION, "positive"),
            (PrimaryEmotion.FEAR, "negative"),
            (PrimaryEmotion.SADNESS, "negative"),
            (PrimaryEmotion.DISGUST, "negative"),
            (PrimaryEmotion.ANGER, "negative"),
        ],
    )
    def test_valence_sign_per_emotion(
        self, emotion: PrimaryEmotion, expected_sign: str
    ):
        """Each emotion should produce the expected valence sign when computed."""
        scores = [EmotionScore(emotion=emotion, intensity=0.8)]
        valence = _compute_valence(scores)
        if expected_sign == "positive":
            assert valence > 0, f"{emotion.value} should have positive valence"
        else:
            assert valence < 0, f"{emotion.value} should have negative valence"

    def test_surprise_has_neutral_valence(self):
        """Surprise has a valence weight of 0.0, so it should be neutral."""
        scores = [EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=0.8)]
        valence = _compute_valence(scores)
        assert valence == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "emotion,min_arousal",
        [
            (PrimaryEmotion.ANGER, 0.7),
            (PrimaryEmotion.SURPRISE, 0.7),
            (PrimaryEmotion.FEAR, 0.6),
            (PrimaryEmotion.ANTICIPATION, 0.5),
        ],
    )
    def test_high_arousal_emotions(
        self, emotion: PrimaryEmotion, min_arousal: float
    ):
        """High-energy emotions should produce high arousal values."""
        scores = [EmotionScore(emotion=emotion, intensity=0.9)]
        arousal = _compute_arousal(scores)
        assert arousal >= min_arousal

    @pytest.mark.parametrize(
        "emotion,max_arousal",
        [
            (PrimaryEmotion.SADNESS, 0.4),
            (PrimaryEmotion.TRUST, 0.5),
        ],
    )
    def test_low_arousal_emotions(
        self, emotion: PrimaryEmotion, max_arousal: float
    ):
        """Low-energy emotions should produce low arousal values."""
        scores = [EmotionScore(emotion=emotion, intensity=0.9)]
        arousal = _compute_arousal(scores)
        assert arousal <= max_arousal

    def test_zero_intensity_scores_return_zero(self):
        """Zero-intensity scores should return zero for both valence and arousal."""
        scores = [EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.0)]
        assert _compute_valence(scores) == 0.0
        assert _compute_arousal(scores) == 0.0

    def test_mixed_valence_cancellation(self):
        """Opposite-valence emotions at similar intensity should partially cancel."""
        scores = [
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7),
            EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.7),
        ]
        valence = _compute_valence(scores)
        # joy (+1.0) and sadness (-0.8) should mostly cancel, near 0.1
        assert abs(valence) < 0.5

    def test_valence_dominated_by_higher_intensity(self):
        """The higher-intensity emotion should dominate the valence."""
        scores = [
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.9),
            EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.1),
        ]
        valence = _compute_valence(scores)
        assert valence > 0.5  # joy dominates


# ── Model validation edge cases ──────────────────────────────────────


class TestModelValidationEdgeCases:
    """Additional model validation tests for robustness."""

    def test_parse_clamped_valence_out_of_range(self):
        """Model-provided valence >1.0 should be clamped to 1.0."""
        raw = {
            "emotions": {"joy": 0.8},
            "valence": 2.5,
        }
        result = parse_emotion_response(raw)
        assert result.valence == 1.0

    def test_parse_clamped_valence_below_range(self):
        """Model-provided valence <-1.0 should be clamped to -1.0."""
        raw = {
            "emotions": {"anger": 0.8},
            "valence": -3.0,
        }
        result = parse_emotion_response(raw)
        assert result.valence == -1.0

    def test_parse_clamped_arousal_out_of_range(self):
        """Model-provided arousal >1.0 should be clamped to 1.0."""
        raw = {
            "emotions": {"surprise": 0.7},
            "arousal": 5.0,
        }
        result = parse_emotion_response(raw)
        assert result.arousal == 1.0

    def test_parse_clamped_arousal_below_range(self):
        """Model-provided arousal <0.0 should be clamped to 0.0."""
        raw = {
            "emotions": {"sadness": 0.6},
            "arousal": -1.0,
        }
        result = parse_emotion_response(raw)
        assert result.arousal == 0.0

    def test_parse_clamped_confidence_out_of_range(self):
        """Model-provided confidence >1.0 should be clamped to 1.0."""
        raw = {
            "emotions": {"trust": 0.5},
            "confidence": 1.5,
        }
        result = parse_emotion_response(raw)
        assert result.confidence == 1.0

    def test_parse_invalid_arousal_fallback_to_computed(self):
        """Non-numeric arousal should fallback to computed value."""
        raw = {
            "emotions": {"anger": 0.9},
            "arousal": "very high",
        }
        result = parse_emotion_response(raw)
        assert 0.0 <= result.arousal <= 1.0
        # Anger has high arousal weight (0.9)
        assert result.arousal > 0.5

    def test_parse_missing_confidence_defaults_to_0_8(self):
        """Missing confidence should default to 0.8."""
        raw = {"emotions": {"joy": 0.7}}
        result = parse_emotion_response(raw)
        assert result.confidence == 0.8

    def test_parse_extra_fields_ignored(self):
        """Extra fields in the response should be silently ignored."""
        raw = {
            "emotions": {"joy": 0.6},
            "valence": 0.5,
            "arousal": 0.4,
            "confidence": 0.9,
            "extra_field": "should be ignored",
            "nested": {"also": "ignored"},
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY

    def test_parse_intensity_clamped_to_zero_then_filtered(self):
        """Negative intensity should be clamped to 0 then filtered as below threshold."""
        raw = {
            "emotions": {"joy": 0.6, "anger": -5.0},
        }
        result = parse_emotion_response(raw)
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.secondary is None  # anger filtered after clamp to 0.0

    @pytest.mark.asyncio
    async def test_service_graceful_on_value_error(self):
        """Service should return NEUTRAL on ValueError from parse."""
        client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({"emotions": {}})  # empty → ValueError
        client.generate.return_value = mock_response

        service = EmotionDetectionService(client)
        result = await service.detect("Some text")

        assert result == NEUTRAL_EMOTION

    @pytest.mark.asyncio
    async def test_service_stats_updated_on_error(self):
        """Stats should still be updated even when detection fails."""
        client = AsyncMock()
        client.generate.side_effect = RuntimeError("API down")

        service = EmotionDetectionService(client)
        await service.detect("Some text")

        assert service.stats["call_count"] == 1
        assert service.stats["total_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_service_custom_threshold(self):
        """Custom min_threshold should filter differently."""
        response_json = {
            "emotions": {"joy": 0.3, "trust": 0.08},
        }
        client = _make_mock_client(response_json)
        # Higher threshold: 0.1 means trust at 0.08 gets filtered
        service = EmotionDetectionService(client, min_threshold=0.1)

        result = await service.detect("Ok.")
        # The service uses _MIN_EMOTION_THRESHOLD in parse_emotion_response,
        # not the instance threshold, so trust at 0.08 still passes (>0.05)
        assert result.primary.emotion == PrimaryEmotion.JOY
