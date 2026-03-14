"""Importance scoring engine for memory promotion decisions.

Scores memory items based on multiple signals:
- Relevance: intent type weight, entity richness, emotion intensity
- Repetition: access count, mention frequency, co-occurrence
- Recency: time decay with configurable half-life
- Importance: composite score used for promotion thresholds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from blurt.core.memory.models import (
    EpisodicMemoryItem,
    IntentType,
    WorkingMemoryItem,
)


@dataclass
class ScoringWeights:
    """Configurable weights for the importance scoring formula."""

    # Relevance signals
    intent_weight: float = 0.25
    entity_richness_weight: float = 0.15
    emotion_intensity_weight: float = 0.10

    # Repetition signals
    access_count_weight: float = 0.20
    mention_count_weight: float = 0.15

    # Recency signal
    recency_weight: float = 0.15

    # Time decay half-life in seconds (default 1 hour)
    recency_half_life_seconds: float = 3600.0

    # Intent-type base scores (some intents are inherently more important)
    intent_scores: dict[str, float] = field(default_factory=lambda: {
        IntentType.TASK.value: 0.9,
        IntentType.EVENT.value: 0.85,
        IntentType.REMINDER.value: 0.8,
        IntentType.IDEA.value: 0.7,
        IntentType.QUESTION.value: 0.6,
        IntentType.UPDATE.value: 0.5,
        IntentType.JOURNAL.value: 0.4,
    })


class ImportanceScorer:
    """Computes importance scores for memory items.

    The score is a weighted combination of relevance, repetition, and recency
    signals, producing a value in [0.0, 1.0].
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score_working_item(self, item: WorkingMemoryItem) -> float:
        """Score a working memory item for potential promotion to episodic."""
        relevance = self._relevance_score(
            intent=item.intent,
            entity_count=len(item.entities),
            emotion_intensity=item.emotion.intensity if item.emotion else 0.0,
        )
        repetition = self._repetition_score(
            access_count=item.access_count,
            mention_count=1,  # working items track access, not mentions
        )
        recency = self._recency_score(item.created_at)

        return self._composite_score(relevance, repetition, recency)

    def score_episodic_item(self, item: EpisodicMemoryItem) -> float:
        """Score an episodic memory item for potential promotion to semantic."""
        relevance = self._relevance_score(
            intent=item.intent,
            entity_count=len(item.entities),
            emotion_intensity=item.emotion.intensity if item.emotion else 0.0,
        )
        repetition = self._repetition_score(
            access_count=item.access_count,
            mention_count=item.mention_count,
        )
        recency = self._recency_score(item.created_at)

        return self._composite_score(relevance, repetition, recency)

    def _relevance_score(
        self,
        intent: IntentType | None,
        entity_count: int,
        emotion_intensity: float,
    ) -> float:
        """Compute relevance from intent type, entity richness, and emotion."""
        w = self.weights

        # Intent score
        intent_score = 0.5  # default for unknown
        if intent is not None:
            intent_score = w.intent_scores.get(intent.value, 0.5)

        # Entity richness: diminishing returns via log
        entity_score = min(1.0, math.log1p(entity_count) / math.log1p(5))

        # Emotion intensity: normalize from 0-3 to 0-1
        emotion_score = min(1.0, emotion_intensity / 3.0)

        total_weight = (
            w.intent_weight + w.entity_richness_weight + w.emotion_intensity_weight
        )
        if total_weight == 0:
            return 0.0

        return (
            w.intent_weight * intent_score
            + w.entity_richness_weight * entity_score
            + w.emotion_intensity_weight * emotion_score
        ) / total_weight

    def _repetition_score(
        self, access_count: int, mention_count: int
    ) -> float:
        """Compute repetition score from access and mention counts."""
        w = self.weights

        # Logarithmic scaling to avoid runaway scores
        access_score = min(1.0, math.log1p(access_count) / math.log1p(10))
        mention_score = min(1.0, math.log1p(mention_count) / math.log1p(10))

        total_weight = w.access_count_weight + w.mention_count_weight
        if total_weight == 0:
            return 0.0

        return (
            w.access_count_weight * access_score
            + w.mention_count_weight * mention_score
        ) / total_weight

    def _recency_score(self, created_at: datetime) -> float:
        """Compute recency score with exponential decay."""
        now = datetime.now(timezone.utc)
        age_seconds = max(0.0, (now - created_at).total_seconds())
        half_life = self.weights.recency_half_life_seconds

        if half_life <= 0:
            return 1.0

        # Exponential decay: score = 2^(-age/half_life)
        return math.pow(2, -age_seconds / half_life)

    def _composite_score(
        self, relevance: float, repetition: float, recency: float
    ) -> float:
        """Combine sub-scores into a single importance score in [0, 1]."""
        w = self.weights

        # Relevance sub-weights already normalized internally
        rel_weight = w.intent_weight + w.entity_richness_weight + w.emotion_intensity_weight
        rep_weight = w.access_count_weight + w.mention_count_weight
        rec_weight = w.recency_weight

        total = rel_weight + rep_weight + rec_weight
        if total == 0:
            return 0.0

        score = (
            rel_weight * relevance
            + rep_weight * repetition
            + rec_weight * recency
        ) / total

        return max(0.0, min(1.0, score))
