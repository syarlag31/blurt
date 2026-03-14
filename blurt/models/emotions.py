"""Plutchik emotion model for Blurt emotion detection.

Implements Plutchik's wheel of emotions with 8 primary emotions,
intensity scaling (0.0–1.0), and composite emotion results as Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrimaryEmotion(str, Enum):
    """Plutchik's 8 primary emotions."""

    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


# Plutchik defines intensity gradations for each primary emotion.
# Low intensity → primary → high intensity
EMOTION_INTENSITY_LABELS: dict[PrimaryEmotion, tuple[str, str, str]] = {
    PrimaryEmotion.JOY: ("serenity", "joy", "ecstasy"),
    PrimaryEmotion.TRUST: ("acceptance", "trust", "admiration"),
    PrimaryEmotion.FEAR: ("apprehension", "fear", "terror"),
    PrimaryEmotion.SURPRISE: ("distraction", "surprise", "amazement"),
    PrimaryEmotion.SADNESS: ("pensiveness", "sadness", "grief"),
    PrimaryEmotion.DISGUST: ("boredom", "disgust", "loathing"),
    PrimaryEmotion.ANGER: ("annoyance", "anger", "rage"),
    PrimaryEmotion.ANTICIPATION: ("interest", "anticipation", "vigilance"),
}

# Plutchik's dyads — adjacent primary emotions combine into complex emotions
PLUTCHIK_DYADS: dict[tuple[PrimaryEmotion, PrimaryEmotion], str] = {
    (PrimaryEmotion.JOY, PrimaryEmotion.TRUST): "love",
    (PrimaryEmotion.TRUST, PrimaryEmotion.FEAR): "submission",
    (PrimaryEmotion.FEAR, PrimaryEmotion.SURPRISE): "awe",
    (PrimaryEmotion.SURPRISE, PrimaryEmotion.SADNESS): "disapproval",
    (PrimaryEmotion.SADNESS, PrimaryEmotion.DISGUST): "remorse",
    (PrimaryEmotion.DISGUST, PrimaryEmotion.ANGER): "contempt",
    (PrimaryEmotion.ANGER, PrimaryEmotion.ANTICIPATION): "aggressiveness",
    (PrimaryEmotion.ANTICIPATION, PrimaryEmotion.JOY): "optimism",
}


class EmotionScore(BaseModel):
    """A single emotion with its intensity.

    Attributes:
        emotion: One of Plutchik's 8 primary emotions.
        intensity: Strength of the emotion, 0.0 (absent) to 1.0 (maximum).
    """

    model_config = ConfigDict(frozen=True)

    emotion: PrimaryEmotion
    intensity: float = Field(ge=0.0, le=1.0, description="Emotion intensity from 0.0 to 1.0")

    @property
    def intensity_label(self) -> str:
        """Return the Plutchik intensity label (low/mid/high) for this score."""
        labels = EMOTION_INTENSITY_LABELS[self.emotion]
        if self.intensity < 0.33:
            return labels[0]  # low
        elif self.intensity < 0.66:
            return labels[1]  # mid (primary name)
        else:
            return labels[2]  # high


class EmotionResult(BaseModel):
    """Complete emotion detection result for a single blurt.

    Attributes:
        primary: The dominant detected emotion with intensity.
        secondary: Optional second-strongest emotion (for blended states).
        valence: Overall positive/negative sentiment, -1.0 to 1.0.
        arousal: Energy/activation level, 0.0 (calm) to 1.0 (activated).
        confidence: Model confidence in the detection, 0.0 to 1.0.
        composite_label: Optional Plutchik dyad label if primary+secondary
            form a recognized combination (e.g. "optimism", "love").
    """

    model_config = ConfigDict(frozen=True)

    primary: EmotionScore
    secondary: Optional[EmotionScore] = None
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Sentiment from -1.0 (negative) to 1.0 (positive)")
    arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="Energy/activation level from 0.0 (calm) to 1.0 (activated)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model confidence from 0.0 to 1.0")
    composite_label: Optional[str] = None

    @property
    def dominant_label(self) -> str:
        """Human-readable label for the dominant emotion at its intensity."""
        return self.primary.intensity_label

    def detected_dyad(self) -> Optional[str]:
        """Check if primary + secondary form a Plutchik dyad."""
        if self.composite_label:
            return self.composite_label
        if self.secondary is None:
            return None
        pair = (self.primary.emotion, self.secondary.emotion)
        reverse = (self.secondary.emotion, self.primary.emotion)
        return PLUTCHIK_DYADS.get(pair) or PLUTCHIK_DYADS.get(reverse)


# Neutral baseline — useful as a default when no emotion is detected
NEUTRAL_EMOTION = EmotionResult(
    primary=EmotionScore(emotion=PrimaryEmotion.TRUST, intensity=0.3),
    valence=0.0,
    arousal=0.3,
    confidence=0.5,
)
