"""Classification result models.

Defines the data structures for intent classification results,
confidence scoring, ambiguity handling, and downstream routing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from blurt.models.intents import BlurtIntent


class ClassificationStatus(str, Enum):
    """Status of a classification attempt."""

    CONFIDENT = "confident"        # Primary intent above threshold
    AMBIGUOUS = "ambiguous"        # Multiple intents close in score
    LOW_CONFIDENCE = "low_confidence"  # Below confidence threshold
    RESOLVED = "resolved"          # Ambiguity resolved via secondary model
    MULTI_INTENT = "multi_intent"  # Input contains multiple intents
    ERROR = "error"                # Classification failed


class FallbackStrategy(str, Enum):
    """Strategy used when classification confidence is below threshold."""

    ESCALATE_TO_SMART = "escalate_to_smart"  # Re-classify with Flash (smarter model)
    MULTI_INTENT_SPLIT = "multi_intent_split"  # Split into multiple classifications
    DEFAULT_JOURNAL = "default_journal"  # Default to journal (safe fallback)
    ASK_CLARIFICATION = "ask_clarification"  # Ask user for clarification


# Confidence threshold for primary classification (85%)
CONFIDENCE_THRESHOLD = 0.85

# If the gap between top-2 intents is less than this, it's ambiguous
AMBIGUITY_GAP_THRESHOLD = 0.15

# If a secondary intent scores above this alongside the primary, check for multi-intent
MULTI_INTENT_THRESHOLD = 0.40


@dataclass
class IntentScore:
    """A single intent with its confidence score."""

    intent: BlurtIntent
    confidence: float

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class AmbiguityResolution:
    """Details about how an ambiguous classification was resolved."""

    original_status: ClassificationStatus
    strategy_used: FallbackStrategy
    original_scores: list[IntentScore]
    resolved_intent: BlurtIntent
    resolved_confidence: float
    resolution_model: str = ""  # Which model resolved it
    resolution_latency_ms: float = 0.0


@dataclass
class ClassificationResult:
    """Complete result of classifying a user input.

    Every blurt flows through this structure. It captures the primary
    classification, all intent scores, confidence level, and any
    ambiguity resolution that was performed.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    primary_intent: BlurtIntent = BlurtIntent.JOURNAL
    confidence: float = 0.0
    status: ClassificationStatus = ClassificationStatus.LOW_CONFIDENCE
    all_scores: list[IntentScore] = field(default_factory=list)
    resolution: AmbiguityResolution | None = None
    model_used: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """Whether the classification met the confidence threshold."""
        return self.status in (
            ClassificationStatus.CONFIDENT,
            ClassificationStatus.RESOLVED,
        )

    @property
    def was_ambiguous(self) -> bool:
        """Whether the classification required ambiguity resolution."""
        return self.resolution is not None

    @property
    def secondary_intent(self) -> BlurtIntent | None:
        """Second-highest scoring intent, if any."""
        if len(self.all_scores) >= 2:
            sorted_scores = sorted(self.all_scores, key=lambda s: s.confidence, reverse=True)
            return sorted_scores[1].intent
        return None

    @property
    def is_multi_intent(self) -> bool:
        """Whether the input was classified as containing multiple intents."""
        return self.status == ClassificationStatus.MULTI_INTENT
