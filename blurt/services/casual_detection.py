"""Casual observation detection — identifies when input is a casual remark.

Every blurt is captured regardless of whether it's "casual" or "substantive."
This module classifies the *nature* of the observation so downstream systems
(behavioral learning, pattern detection, emotional baselines) can leverage
casual remarks as signal rather than noise.

Casual remarks include:
- Throwaway comments ("huh", "interesting", "nice weather")
- Filler words ("hmm", "yeah", "ok", "sure")
- Reactions ("oh wow", "cool", "that's funny")
- Half-finished thoughts ("I was thinking maybe...")
- Brief observations ("traffic was bad today")
- Empty/near-empty utterances (silence, single words)

Design principles:
- NEVER used for filtering — casual detection is for enrichment only
- Casual observations are ALWAYS stored (zero-drop guarantee)
- Casual observations contribute to behavioral patterns and emotional baselines
- No shame: no input is "too trivial" — everything has signal value
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ObservationType(str, Enum):
    """Classification of observation nature — NOT intent.

    This is orthogonal to intent classification. A "journal" intent can be
    either casual ("nice weather") or substantive ("I've been reflecting on
    my career path"). ObservationType captures this distinction.
    """

    CASUAL = "casual"           # Throwaway remarks, filler, reactions
    SUBSTANTIVE = "substantive" # Meaningful content with clear semantic payload
    AMBIGUOUS = "ambiguous"     # Could go either way (short but potentially meaningful)


@dataclass(frozen=True)
class CasualDetectionResult:
    """Result of casual observation detection.

    Attributes:
        observation_type: Whether the input is casual, substantive, or ambiguous.
        confidence: How confident we are in the classification (0.0-1.0).
        signals: Which signals triggered the classification.
        word_count: Number of words in the input.
        is_casual: Convenience property — True if casual or ambiguous-leaning-casual.
    """

    observation_type: ObservationType
    confidence: float
    signals: tuple[str, ...]
    word_count: int

    @property
    def is_casual(self) -> bool:
        """Whether this observation is likely casual."""
        return self.observation_type == ObservationType.CASUAL

    @property
    def is_substantive(self) -> bool:
        """Whether this observation has clear substantive content."""
        return self.observation_type == ObservationType.SUBSTANTIVE


# ---------------------------------------------------------------------------
# Signal patterns for casual detection
# ---------------------------------------------------------------------------

# Single-word filler/reactions (exact match, case-insensitive)
_FILLER_WORDS: frozenset[str] = frozenset({
    "hmm", "hm", "huh", "oh", "ah", "uh", "um", "umm",
    "yeah", "yep", "yup", "nah", "nope",
    "ok", "okay", "sure", "right", "cool", "nice",
    "wow", "whoa", "yikes", "ugh", "meh", "eh",
    "whatever", "anyway", "anyways", "welp", "well",
    "thanks", "thankyou", "bye", "hi", "hey", "hello",
    "yes", "no", "maybe", "true", "false",
    "interesting", "funny", "weird", "crazy", "wild",
})

# Multi-word casual phrases (startswith match, case-insensitive)
_CASUAL_PREFIXES: tuple[str, ...] = (
    "huh,", "hmm,", "oh,", "ah,", "oh well", "oh right",
    "oh wow", "oh cool", "oh nice", "oh no", "oh man",
    "yeah,", "yeah ", "ok so", "okay so", "sure,",
    "whatever,", "anyway,", "anyways,",
    "nice weather", "nice day", "good weather",
    "that's funny", "that's cool", "that's nice",
    "that's interesting", "that's weird", "that's wild",
    "not bad", "not great", "could be worse",
    "i guess", "i suppose", "i dunno", "i don't know",
    "no idea", "who knows", "beats me",
)

# Patterns suggesting substantive content
_SUBSTANTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(need to|have to|should|must|gotta|got to)\b", re.I),
    re.compile(r"\b(remind me|don't forget|remember to)\b", re.I),
    re.compile(r"\b(meeting|appointment|call with|dinner with)\b", re.I),
    re.compile(r"\b(at \d{1,2}(:\d{2})?\s*(am|pm|AM|PM)?)\b", re.I),
    re.compile(r"\b(tomorrow|next week|on monday|on tuesday|on wednesday|on thursday|on friday|on saturday|on sunday)\b", re.I),
    re.compile(r"\b(what if|i think|idea|hypothesis|theory)\b", re.I),
    re.compile(r"\b(feeling|felt|emotion|stressed|anxious|excited|grateful)\b", re.I),
    re.compile(r"\b(finished|completed|done with|started|began)\b", re.I),
    re.compile(r"\b(cancel|reschedule|move to|postpone|extend)\b", re.I),
    re.compile(r"\b(what did|when did|how many|where is|who is)\b", re.I),
    re.compile(r"\b(buy|purchase|order|book|reserve)\b", re.I),
)


def detect_casual(text: str) -> CasualDetectionResult:
    """Detect whether a text input is a casual remark or substantive observation.

    This is a fast, local-only heuristic — no LLM call required. It runs
    synchronously and is designed for sub-millisecond performance.

    The detection is used for enrichment and behavioral learning, NEVER
    for filtering or dropping observations.

    Args:
        text: Raw input text (transcribed from voice or typed).

    Returns:
        CasualDetectionResult with type, confidence, and signals.
    """
    stripped = text.strip()
    words = stripped.split()
    word_count = len(words)
    signals: list[str] = []

    # Empty or near-empty input
    if word_count == 0:
        return CasualDetectionResult(
            observation_type=ObservationType.CASUAL,
            confidence=1.0,
            signals=("empty_input",),
            word_count=0,
        )

    text_lower = stripped.lower()

    # Single-word filler detection
    if word_count == 1 and text_lower.rstrip(".,!?") in _FILLER_WORDS:
        return CasualDetectionResult(
            observation_type=ObservationType.CASUAL,
            confidence=0.95,
            signals=("single_filler_word",),
            word_count=1,
        )

    # Check for casual prefixes
    casual_prefix_match = False
    for prefix in _CASUAL_PREFIXES:
        if text_lower.startswith(prefix):
            casual_prefix_match = True
            signals.append(f"casual_prefix:{prefix}")
            break

    # Check for substantive patterns
    substantive_matches = 0
    for pattern in _SUBSTANTIVE_PATTERNS:
        if pattern.search(text_lower):
            substantive_matches += 1
            signals.append(f"substantive_pattern:{pattern.pattern[:30]}")

    # Decision logic
    if substantive_matches >= 2:
        return CasualDetectionResult(
            observation_type=ObservationType.SUBSTANTIVE,
            confidence=min(0.7 + substantive_matches * 0.1, 0.95),
            signals=tuple(signals),
            word_count=word_count,
        )

    if substantive_matches == 1 and not casual_prefix_match:
        return CasualDetectionResult(
            observation_type=ObservationType.SUBSTANTIVE,
            confidence=0.7,
            signals=tuple(signals),
            word_count=word_count,
        )

    if casual_prefix_match and substantive_matches == 0:
        return CasualDetectionResult(
            observation_type=ObservationType.CASUAL,
            confidence=0.85,
            signals=tuple(signals),
            word_count=word_count,
        )

    # Short utterances without substantive markers lean casual
    if word_count <= 3 and substantive_matches == 0:
        signals.append("short_utterance")
        return CasualDetectionResult(
            observation_type=ObservationType.CASUAL,
            confidence=0.75,
            signals=tuple(signals),
            word_count=word_count,
        )

    # Medium length without clear markers — ambiguous
    if word_count <= 6 and substantive_matches == 0:
        signals.append("medium_no_markers")
        return CasualDetectionResult(
            observation_type=ObservationType.AMBIGUOUS,
            confidence=0.6,
            signals=tuple(signals),
            word_count=word_count,
        )

    # Longer text without explicit casual markers — likely substantive
    if word_count > 6 and not casual_prefix_match:
        signals.append("long_utterance")
        return CasualDetectionResult(
            observation_type=ObservationType.SUBSTANTIVE,
            confidence=0.65,
            signals=tuple(signals),
            word_count=word_count,
        )

    # Mixed signals — ambiguous
    if not signals:
        signals.append("no_clear_signals")
    return CasualDetectionResult(
        observation_type=ObservationType.AMBIGUOUS,
        confidence=0.5,
        signals=tuple(signals),
        word_count=word_count,
    )
