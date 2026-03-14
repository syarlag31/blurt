"""Comprehensive tests for emotion detection models.

Covers:
- All 8 Plutchik primary emotions
- Intensity scoring accuracy and label thresholds
- Mixed-emotion (secondary) inputs and Plutchik dyads
- Valence/arousal/confidence validation and boundary conditions
- EmotionScore and EmotionResult construction and constraints
- NEUTRAL_EMOTION default baseline
- Cross-module consistency checks
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from blurt.models.emotions import (
    EMOTION_INTENSITY_LABELS,
    NEUTRAL_EMOTION,
    PLUTCHIK_DYADS,
    EmotionResult,
    EmotionScore,
    PrimaryEmotion,
)


# ---------------------------------------------------------------------------
# PrimaryEmotion enum — all 8 Plutchik emotions
# ---------------------------------------------------------------------------


class TestPrimaryEmotion:
    """Ensure all 8 Plutchik primary emotions are correctly defined."""

    EXPECTED_EMOTIONS = [
        "joy", "trust", "fear", "surprise",
        "sadness", "disgust", "anger", "anticipation",
    ]

    def test_has_exactly_8_emotions(self):
        assert len(PrimaryEmotion) == 8

    @pytest.mark.parametrize("emotion_value", EXPECTED_EMOTIONS)
    def test_each_emotion_exists(self, emotion_value: str):
        emotion = PrimaryEmotion(emotion_value)
        assert emotion.value == emotion_value

    def test_all_values_match_expected(self):
        actual = sorted(e.value for e in PrimaryEmotion)
        expected = sorted(self.EXPECTED_EMOTIONS)
        assert actual == expected

    def test_is_str_enum(self):
        """PrimaryEmotion should be usable as a plain string."""
        assert isinstance(PrimaryEmotion.JOY, str)
        assert PrimaryEmotion.JOY == "joy"
        assert PrimaryEmotion.ANGER == "anger"

    def test_invalid_emotion_value_raises(self):
        with pytest.raises(ValueError):
            PrimaryEmotion("happiness")  # not a valid Plutchik emotion

    def test_case_sensitivity(self):
        """Enum values are lowercase."""
        with pytest.raises(ValueError):
            PrimaryEmotion("JOY")
        assert PrimaryEmotion("joy") == PrimaryEmotion.JOY


# ---------------------------------------------------------------------------
# EMOTION_INTENSITY_LABELS — Plutchik gradations
# ---------------------------------------------------------------------------


class TestEmotionIntensityLabels:
    """Verify Plutchik's intensity gradation labels for every emotion."""

    def test_all_8_emotions_have_labels(self):
        assert len(EMOTION_INTENSITY_LABELS) == 8
        for emotion in PrimaryEmotion:
            assert emotion in EMOTION_INTENSITY_LABELS

    def test_each_label_is_3_tuple_of_strings(self):
        for emotion, labels in EMOTION_INTENSITY_LABELS.items():
            assert len(labels) == 3, f"{emotion} should have (low, mid, high) labels"
            for label in labels:
                assert isinstance(label, str) and len(label) > 0

    @pytest.mark.parametrize(
        "emotion, low, mid, high",
        [
            (PrimaryEmotion.JOY, "serenity", "joy", "ecstasy"),
            (PrimaryEmotion.TRUST, "acceptance", "trust", "admiration"),
            (PrimaryEmotion.FEAR, "apprehension", "fear", "terror"),
            (PrimaryEmotion.SURPRISE, "distraction", "surprise", "amazement"),
            (PrimaryEmotion.SADNESS, "pensiveness", "sadness", "grief"),
            (PrimaryEmotion.DISGUST, "boredom", "disgust", "loathing"),
            (PrimaryEmotion.ANGER, "annoyance", "anger", "rage"),
            (PrimaryEmotion.ANTICIPATION, "interest", "anticipation", "vigilance"),
        ],
    )
    def test_correct_intensity_labels_for_each_emotion(self, emotion, low, mid, high):
        labels = EMOTION_INTENSITY_LABELS[emotion]
        assert labels == (low, mid, high)

    def test_all_labels_are_unique_across_emotions(self):
        """Each intensity label should be unique (no duplication)."""
        all_labels = []
        for labels in EMOTION_INTENSITY_LABELS.values():
            all_labels.extend(labels)
        assert len(all_labels) == len(set(all_labels))


# ---------------------------------------------------------------------------
# EmotionScore — individual emotion with intensity
# ---------------------------------------------------------------------------


class TestEmotionScore:
    """Tests for individual emotion scoring."""

    # -- Basic construction --

    def test_basic_construction(self):
        score = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8)
        assert score.emotion == PrimaryEmotion.JOY
        assert score.intensity == 0.8

    @pytest.mark.parametrize("emotion", list(PrimaryEmotion))
    def test_construct_each_of_8_emotions(self, emotion: PrimaryEmotion):
        """Every primary emotion should be constructible with valid intensity."""
        score = EmotionScore(emotion=emotion, intensity=0.5)
        assert score.emotion == emotion
        assert score.intensity == 0.5

    def test_frozen_model_prevents_mutation(self):
        score = EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.6)
        with pytest.raises((AttributeError, ValidationError)):
            score.intensity = 0.9  # type: ignore[misc]
        with pytest.raises((AttributeError, ValidationError)):
            score.emotion = PrimaryEmotion.JOY  # type: ignore[misc]

    # -- Intensity boundary validation --

    def test_intensity_exact_zero(self):
        score = EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=0.0)
        assert score.intensity == 0.0

    def test_intensity_exact_one(self):
        score = EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=1.0)
        assert score.intensity == 1.0

    def test_intensity_negative_raises(self):
        with pytest.raises(ValueError, match="[Ii]ntensity"):
            EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=-0.1)

    def test_intensity_above_one_raises(self):
        with pytest.raises(ValueError, match="[Ii]ntensity"):
            EmotionScore(emotion=PrimaryEmotion.JOY, intensity=1.01)

    def test_intensity_large_negative_raises(self):
        with pytest.raises(ValueError):
            EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=-100.0)

    def test_intensity_large_positive_raises(self):
        with pytest.raises(ValueError):
            EmotionScore(emotion=PrimaryEmotion.DISGUST, intensity=5.0)

    def test_intensity_barely_above_one_raises(self):
        with pytest.raises(ValueError):
            EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=1.0001)

    def test_intensity_barely_below_zero_raises(self):
        with pytest.raises(ValueError):
            EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=-0.0001)

    # -- Intensity label thresholds (JOY as canonical example) --

    @pytest.mark.parametrize(
        "intensity, expected_label",
        [
            (0.0, "serenity"),      # low: min
            (0.15, "serenity"),     # low: middle
            (0.32, "serenity"),     # low: just under threshold
            (0.329, "serenity"),    # low: edge
            (0.33, "joy"),          # mid: threshold
            (0.5, "joy"),           # mid: middle
            (0.65, "joy"),          # mid: just under threshold
            (0.659, "joy"),         # mid: edge
            (0.66, "ecstasy"),      # high: threshold
            (0.8, "ecstasy"),       # high: middle
            (1.0, "ecstasy"),       # high: max
        ],
    )
    def test_joy_intensity_label_thresholds(self, intensity: float, expected_label: str):
        score = EmotionScore(emotion=PrimaryEmotion.JOY, intensity=intensity)
        assert score.intensity_label == expected_label

    # -- Intensity labels for all 8 emotions at each tier --

    @pytest.mark.parametrize(
        "emotion, intensity, expected_label",
        [
            # FEAR gradations
            (PrimaryEmotion.FEAR, 0.1, "apprehension"),
            (PrimaryEmotion.FEAR, 0.5, "fear"),
            (PrimaryEmotion.FEAR, 0.9, "terror"),
            # ANGER gradations
            (PrimaryEmotion.ANGER, 0.2, "annoyance"),
            (PrimaryEmotion.ANGER, 0.5, "anger"),
            (PrimaryEmotion.ANGER, 0.8, "rage"),
            # SADNESS gradations
            (PrimaryEmotion.SADNESS, 0.1, "pensiveness"),
            (PrimaryEmotion.SADNESS, 0.5, "sadness"),
            (PrimaryEmotion.SADNESS, 0.9, "grief"),
            # TRUST gradations
            (PrimaryEmotion.TRUST, 0.1, "acceptance"),
            (PrimaryEmotion.TRUST, 0.5, "trust"),
            (PrimaryEmotion.TRUST, 0.9, "admiration"),
            # SURPRISE gradations
            (PrimaryEmotion.SURPRISE, 0.1, "distraction"),
            (PrimaryEmotion.SURPRISE, 0.5, "surprise"),
            (PrimaryEmotion.SURPRISE, 0.9, "amazement"),
            # DISGUST gradations
            (PrimaryEmotion.DISGUST, 0.1, "boredom"),
            (PrimaryEmotion.DISGUST, 0.5, "disgust"),
            (PrimaryEmotion.DISGUST, 0.9, "loathing"),
            # ANTICIPATION gradations
            (PrimaryEmotion.ANTICIPATION, 0.1, "interest"),
            (PrimaryEmotion.ANTICIPATION, 0.5, "anticipation"),
            (PrimaryEmotion.ANTICIPATION, 0.9, "vigilance"),
            # JOY gradations (already covered above but included for completeness)
            (PrimaryEmotion.JOY, 0.1, "serenity"),
            (PrimaryEmotion.JOY, 0.5, "joy"),
            (PrimaryEmotion.JOY, 0.9, "ecstasy"),
        ],
    )
    def test_intensity_labels_all_8_emotions(
        self, emotion: PrimaryEmotion, intensity: float, expected_label: str
    ):
        score = EmotionScore(emotion=emotion, intensity=intensity)
        assert score.intensity_label == expected_label

    # -- Float precision --

    def test_float_precision_intensity(self):
        score = EmotionScore(emotion=PrimaryEmotion.ANTICIPATION, intensity=0.123456789)
        assert 0.123 < score.intensity < 0.124

    def test_very_small_nonzero_intensity(self):
        score = EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.001)
        assert score.intensity == pytest.approx(0.001)
        assert score.intensity_label == "acceptance"  # low tier

    def test_near_one_intensity(self):
        score = EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.999)
        assert score.intensity_label == "rage"  # high tier


# ---------------------------------------------------------------------------
# EmotionResult — complete emotion detection result
# ---------------------------------------------------------------------------


class TestEmotionResult:
    """Tests for the complete emotion detection result."""

    # -- Construction --

    def test_basic_construction_defaults(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7),
        )
        assert result.primary.emotion == PrimaryEmotion.JOY
        assert result.secondary is None
        assert result.valence == 0.0
        assert result.arousal == 0.5
        assert result.confidence == 1.0
        assert result.composite_label is None

    def test_full_construction(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8),
            secondary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
            valence=0.7,
            arousal=0.6,
            confidence=0.92,
            composite_label="love",
        )
        assert result.secondary is not None
        assert result.secondary.emotion == PrimaryEmotion.TRUST
        assert result.composite_label == "love"

    def test_minimal_valid_result(self):
        """Minimum valid EmotionResult: just a primary with zero intensity."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.0),
        )
        assert result.valence == 0.0
        assert result.arousal == 0.5
        assert result.confidence == 1.0

    def test_frozen_prevents_mutation(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5),
        )
        with pytest.raises((AttributeError, ValidationError)):
            result.valence = 0.9  # type: ignore[misc]
        with pytest.raises((AttributeError, ValidationError)):
            result.arousal = 0.9  # type: ignore[misc]
        with pytest.raises((AttributeError, ValidationError)):
            result.confidence = 0.9  # type: ignore[misc]

    # -- Valence validation --

    @pytest.mark.parametrize("valence", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_valid_valence_values(self, valence: float):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
            valence=valence,
        )
        assert result.valence == valence

    def test_valence_below_range_raises(self):
        with pytest.raises(ValueError, match="[Vv]alence"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.5),
                valence=-1.01,
            )

    def test_valence_above_range_raises(self):
        with pytest.raises(ValueError, match="[Vv]alence"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5),
                valence=1.01,
            )

    def test_valence_far_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.5),
                valence=-5.0,
            )

    # -- Arousal validation --

    @pytest.mark.parametrize("arousal", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_arousal_values(self, arousal: float):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=0.5),
            arousal=arousal,
        )
        assert result.arousal == arousal

    def test_arousal_below_range_raises(self):
        with pytest.raises(ValueError, match="[Aa]rousal"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=0.5),
                arousal=-0.1,
            )

    def test_arousal_above_range_raises(self):
        with pytest.raises(ValueError, match="[Aa]rousal"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=0.5),
                arousal=1.1,
            )

    # -- Confidence validation --

    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_confidence_values(self, confidence: float):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=0.5),
            confidence=confidence,
        )
        assert result.confidence == confidence

    def test_confidence_below_range_raises(self):
        with pytest.raises(ValueError, match="[Cc]onfidence"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
                confidence=-0.1,
            )

    def test_confidence_above_range_raises(self):
        with pytest.raises(ValueError, match="[Cc]onfidence"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
                confidence=1.5,
            )

    # -- dominant_label property --

    def test_dominant_label_low(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.1),
        )
        assert result.dominant_label == "annoyance"

    def test_dominant_label_mid(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.5),
        )
        assert result.dominant_label == "anger"

    def test_dominant_label_high(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.9),
        )
        assert result.dominant_label == "rage"

    @pytest.mark.parametrize("emotion", list(PrimaryEmotion))
    def test_dominant_label_is_string_for_all_emotions(self, emotion: PrimaryEmotion):
        result = EmotionResult(
            primary=EmotionScore(emotion=emotion, intensity=0.5),
        )
        assert isinstance(result.dominant_label, str)
        assert len(result.dominant_label) > 0

    # -- All 8 emotions at all 3 intensity tiers produce correct dominant label --

    @pytest.mark.parametrize(
        "emotion, intensity, expected_label",
        [
            (PrimaryEmotion.JOY, 0.1, "serenity"),
            (PrimaryEmotion.JOY, 0.5, "joy"),
            (PrimaryEmotion.JOY, 0.9, "ecstasy"),
            (PrimaryEmotion.TRUST, 0.1, "acceptance"),
            (PrimaryEmotion.TRUST, 0.5, "trust"),
            (PrimaryEmotion.TRUST, 0.9, "admiration"),
            (PrimaryEmotion.FEAR, 0.1, "apprehension"),
            (PrimaryEmotion.FEAR, 0.5, "fear"),
            (PrimaryEmotion.FEAR, 0.9, "terror"),
            (PrimaryEmotion.SURPRISE, 0.1, "distraction"),
            (PrimaryEmotion.SURPRISE, 0.5, "surprise"),
            (PrimaryEmotion.SURPRISE, 0.9, "amazement"),
            (PrimaryEmotion.SADNESS, 0.1, "pensiveness"),
            (PrimaryEmotion.SADNESS, 0.5, "sadness"),
            (PrimaryEmotion.SADNESS, 0.9, "grief"),
            (PrimaryEmotion.DISGUST, 0.1, "boredom"),
            (PrimaryEmotion.DISGUST, 0.5, "disgust"),
            (PrimaryEmotion.DISGUST, 0.9, "loathing"),
            (PrimaryEmotion.ANGER, 0.1, "annoyance"),
            (PrimaryEmotion.ANGER, 0.5, "anger"),
            (PrimaryEmotion.ANGER, 0.9, "rage"),
            (PrimaryEmotion.ANTICIPATION, 0.1, "interest"),
            (PrimaryEmotion.ANTICIPATION, 0.5, "anticipation"),
            (PrimaryEmotion.ANTICIPATION, 0.9, "vigilance"),
        ],
    )
    def test_dominant_label_all_emotions_all_tiers(
        self, emotion: PrimaryEmotion, intensity: float, expected_label: str
    ):
        result = EmotionResult(primary=EmotionScore(emotion=emotion, intensity=intensity))
        assert result.dominant_label == expected_label


# ---------------------------------------------------------------------------
# Plutchik Dyads — mixed-emotion detection
# ---------------------------------------------------------------------------


class TestPlutchikDyads:
    """Tests for Plutchik's primary dyad (mixed emotion) definitions and detection."""

    ALL_EXPECTED_DYADS = [
        (PrimaryEmotion.JOY, PrimaryEmotion.TRUST, "love"),
        (PrimaryEmotion.TRUST, PrimaryEmotion.FEAR, "submission"),
        (PrimaryEmotion.FEAR, PrimaryEmotion.SURPRISE, "awe"),
        (PrimaryEmotion.SURPRISE, PrimaryEmotion.SADNESS, "disapproval"),
        (PrimaryEmotion.SADNESS, PrimaryEmotion.DISGUST, "remorse"),
        (PrimaryEmotion.DISGUST, PrimaryEmotion.ANGER, "contempt"),
        (PrimaryEmotion.ANGER, PrimaryEmotion.ANTICIPATION, "aggressiveness"),
        (PrimaryEmotion.ANTICIPATION, PrimaryEmotion.JOY, "optimism"),
    ]

    def test_exactly_8_dyads_defined(self):
        assert len(PLUTCHIK_DYADS) == 8

    @pytest.mark.parametrize("primary, secondary, expected", ALL_EXPECTED_DYADS)
    def test_dyad_lookup_forward(self, primary, secondary, expected):
        assert PLUTCHIK_DYADS[(primary, secondary)] == expected

    @pytest.mark.parametrize("primary, secondary, expected", ALL_EXPECTED_DYADS)
    def test_detected_dyad_forward_order(self, primary, secondary, expected):
        """detected_dyad() works for canonical pair ordering."""
        result = EmotionResult(
            primary=EmotionScore(emotion=primary, intensity=0.6),
            secondary=EmotionScore(emotion=secondary, intensity=0.4),
            valence=0.0,
            arousal=0.5,
        )
        assert result.detected_dyad() == expected

    @pytest.mark.parametrize("primary, secondary, expected", ALL_EXPECTED_DYADS)
    def test_detected_dyad_reverse_order(self, primary, secondary, expected):
        """detected_dyad() should work regardless of primary/secondary ordering."""
        result = EmotionResult(
            primary=EmotionScore(emotion=secondary, intensity=0.6),
            secondary=EmotionScore(emotion=primary, intensity=0.4),
            valence=0.0,
            arousal=0.5,
        )
        assert result.detected_dyad() == expected

    def test_detected_dyad_no_secondary_returns_none(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.7),
        )
        assert result.detected_dyad() is None

    def test_detected_dyad_non_adjacent_returns_none(self):
        """Non-adjacent emotions should not form a recognized dyad."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7),
            secondary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.5),
            valence=0.0,
            arousal=0.7,
        )
        assert result.detected_dyad() is None

    def test_detected_dyad_opposite_emotions_no_match(self):
        """Opposite emotions (e.g., joy and sadness) should not produce a dyad."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7),
            secondary=EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.3),
            valence=0.2,
            arousal=0.5,
        )
        assert result.detected_dyad() is None

    def test_composite_label_overrides_dyad_lookup(self):
        """Explicit composite_label takes precedence over dyad lookup."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.7),
            secondary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
            valence=0.8,
            arousal=0.5,
            composite_label="custom_love",
        )
        assert result.detected_dyad() == "custom_love"

    def test_composite_label_without_secondary(self):
        """composite_label returns even without secondary emotion."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.5),
            composite_label="custom_emotion",
        )
        assert result.detected_dyad() == "custom_emotion"


# ---------------------------------------------------------------------------
# Mixed-emotion inputs — realistic blended states
# ---------------------------------------------------------------------------


class TestMixedEmotionInputs:
    """Test emotion results with both primary and secondary emotions in realistic scenarios."""

    def test_primary_stronger_than_secondary(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.9),
            secondary=EmotionScore(emotion=PrimaryEmotion.ANTICIPATION, intensity=0.3),
            valence=0.7,
            arousal=0.6,
        )
        assert result.secondary is not None
        assert result.primary.intensity > result.secondary.intensity
        assert result.detected_dyad() == "optimism"

    def test_equal_intensity_primary_and_secondary(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=0.5),
            secondary=EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=0.5),
            valence=-0.3,
            arousal=0.8,
        )
        assert result.secondary is not None
        assert result.primary.intensity == result.secondary.intensity
        assert result.detected_dyad() == "awe"

    def test_low_intensity_both_emotions_near_neutral(self):
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.1),
            secondary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.05),
            valence=0.1,
            arousal=0.1,
            confidence=0.3,
        )
        assert result.dominant_label == "acceptance"
        assert result.confidence == 0.3

    def test_high_intensity_mixed_aggression(self):
        """Strong anger + anticipation = aggressiveness."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.9),
            secondary=EmotionScore(emotion=PrimaryEmotion.ANTICIPATION, intensity=0.8),
            valence=-0.5,
            arousal=0.9,
        )
        assert result.dominant_label == "rage"
        assert result.detected_dyad() == "aggressiveness"

    def test_same_emotion_primary_and_secondary(self):
        """Edge case: same emotion for both primary and secondary."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.8),
            secondary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.4),
            valence=0.9,
            arousal=0.6,
        )
        # Same-emotion pair won't be in the dyads dict
        assert result.detected_dyad() is None

    def test_contempt_disgust_anger_dyad(self):
        """Disgust + anger = contempt."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.DISGUST, intensity=0.7),
            secondary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=0.6),
            valence=-0.7,
            arousal=0.6,
        )
        assert result.detected_dyad() == "contempt"

    def test_submission_trust_fear_dyad(self):
        """Trust + fear = submission."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.6),
            secondary=EmotionScore(emotion=PrimaryEmotion.FEAR, intensity=0.4),
            valence=-0.1,
            arousal=0.5,
        )
        assert result.detected_dyad() == "submission"

    def test_remorse_sadness_disgust_dyad(self):
        """Sadness + disgust = remorse."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.8),
            secondary=EmotionScore(emotion=PrimaryEmotion.DISGUST, intensity=0.5),
            valence=-0.8,
            arousal=0.3,
        )
        assert result.detected_dyad() == "remorse"

    def test_disapproval_surprise_sadness_dyad(self):
        """Surprise + sadness = disapproval."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=0.6),
            secondary=EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.5),
            valence=-0.4,
            arousal=0.6,
        )
        assert result.detected_dyad() == "disapproval"


# ---------------------------------------------------------------------------
# NEUTRAL_EMOTION baseline
# ---------------------------------------------------------------------------


class TestNeutralEmotion:
    """Tests for the NEUTRAL_EMOTION default baseline."""

    def test_neutral_is_valid_emotion_result(self):
        assert isinstance(NEUTRAL_EMOTION, EmotionResult)

    def test_neutral_primary_is_trust(self):
        assert NEUTRAL_EMOTION.primary.emotion == PrimaryEmotion.TRUST

    def test_neutral_low_intensity(self):
        assert NEUTRAL_EMOTION.primary.intensity == 0.3

    def test_neutral_zero_valence(self):
        assert NEUTRAL_EMOTION.valence == 0.0

    def test_neutral_low_arousal(self):
        assert NEUTRAL_EMOTION.arousal == 0.3
        assert NEUTRAL_EMOTION.arousal < 0.5

    def test_neutral_moderate_confidence(self):
        assert NEUTRAL_EMOTION.confidence == 0.5

    def test_neutral_dominant_label_is_acceptance(self):
        """At 0.3 intensity, trust → 'acceptance' (low tier)."""
        assert NEUTRAL_EMOTION.dominant_label == "acceptance"

    def test_neutral_no_secondary(self):
        assert NEUTRAL_EMOTION.secondary is None

    def test_neutral_no_dyad(self):
        assert NEUTRAL_EMOTION.detected_dyad() is None

    def test_neutral_is_frozen(self):
        with pytest.raises((AttributeError, ValidationError)):
            NEUTRAL_EMOTION.valence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and edge-case tests for the emotion detection models."""

    # -- Intensity threshold boundaries --

    def test_intensity_exact_at_033_boundary(self):
        """At exactly 0.33, should switch to 'mid' label."""
        score = EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.33)
        assert score.intensity_label == "sadness"  # mid

    def test_intensity_just_below_033_boundary(self):
        score = EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.329)
        assert score.intensity_label == "pensiveness"  # low

    def test_intensity_exact_at_066_boundary(self):
        """At exactly 0.66, should switch to 'high' label."""
        score = EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.66)
        assert score.intensity_label == "grief"  # high

    def test_intensity_just_below_066_boundary(self):
        score = EmotionScore(emotion=PrimaryEmotion.SADNESS, intensity=0.659)
        assert score.intensity_label == "sadness"  # mid

    # -- Extreme valid emotion states --

    def test_extreme_negative_high_arousal(self):
        """Extreme negative valence + high arousal (rage scenario)."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.ANGER, intensity=1.0),
            valence=-1.0,
            arousal=1.0,
        )
        assert result.dominant_label == "rage"
        assert result.valence == -1.0
        assert result.arousal == 1.0

    def test_extreme_positive_low_arousal(self):
        """Extreme positive valence + low arousal (serenity scenario)."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=0.1),
            valence=1.0,
            arousal=0.0,
        )
        assert result.dominant_label == "serenity"
        assert result.valence == 1.0
        assert result.arousal == 0.0

    def test_zero_confidence_still_valid(self):
        """Zero confidence is valid — model is uncertain."""
        result = EmotionResult(
            primary=EmotionScore(emotion=PrimaryEmotion.SURPRISE, intensity=0.2),
            confidence=0.0,
        )
        assert result.confidence == 0.0

    # -- Combined validation failures --

    def test_invalid_intensity_with_invalid_valence(self):
        """Invalid intensity should raise before valence is checked."""
        with pytest.raises(ValueError, match="[Ii]ntensity"):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.JOY, intensity=-0.5),
                valence=-5.0,
            )

    # -- All 8 emotions constructible as complete results --

    @pytest.mark.parametrize("emotion", list(PrimaryEmotion))
    def test_all_emotions_produce_valid_results(self, emotion: PrimaryEmotion):
        result = EmotionResult(
            primary=EmotionScore(emotion=emotion, intensity=0.5),
            valence=0.0,
            arousal=0.5,
        )
        assert result.primary.emotion == emotion
        assert isinstance(result.dominant_label, str)
        assert result.detected_dyad() is None  # no secondary

    # -- Multiple validation errors --

    def test_arousal_negative_far_out_of_range(self):
        with pytest.raises(ValueError):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
                arousal=-10.0,
            )

    def test_confidence_far_out_of_range(self):
        with pytest.raises(ValueError):
            EmotionResult(
                primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.5),
                confidence=100.0,
            )


# ---------------------------------------------------------------------------
# Cross-module consistency checks
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    """Ensure emotion models are consistent across the codebase."""

    def test_primary_emotion_values_match_working_memory_emotion_labels(self):
        """PrimaryEmotion values should match EmotionLabel values from working memory."""
        from blurt.memory.working import EmotionLabel

        primary_values = {e.value for e in PrimaryEmotion}
        label_values = {e.value for e in EmotionLabel}
        assert primary_values == label_values, (
            f"PrimaryEmotion and EmotionLabel are out of sync: "
            f"only in PrimaryEmotion={primary_values - label_values}, "
            f"only in EmotionLabel={label_values - primary_values}"
        )

    def test_classification_prompt_references_all_8_emotions(self):
        """The Gemini classification prompt should mention all 8 emotion names."""
        from blurt.gemini.audio_client import GeminiAudioClient

        client = GeminiAudioClient.__new__(GeminiAudioClient)
        prompt = client._classification_prompt("test input")

        for emotion in PrimaryEmotion:
            assert emotion.value in prompt, (
                f"Emotion '{emotion.value}' missing from classification prompt"
            )

    def test_models_package_re_exports_emotion_types(self):
        """blurt.models should re-export all emotion types."""
        from blurt.models import (
            EMOTION_INTENSITY_LABELS,
            NEUTRAL_EMOTION,
            PLUTCHIK_DYADS,
            EmotionResult,
            PrimaryEmotion,
        )

        assert len(PrimaryEmotion) == 8
        assert isinstance(NEUTRAL_EMOTION, EmotionResult)
        assert len(EMOTION_INTENSITY_LABELS) == 8
        assert len(PLUTCHIK_DYADS) == 8
