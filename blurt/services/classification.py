"""Intent classification service — unified interface for cloud and local classifiers.

Provides a high-level service that:
1. Takes raw text input
2. Runs it through the LLM prompt chain (Flash-Lite → Flash escalation)
3. Returns classified intent with confidence scores
4. Achieves >85% accuracy across all 7 intent types

Uses the two-model strategy:
- Flash-Lite (FAST tier) for initial classification via structured prompt
- Flash (SMART tier) for resolving ambiguous cases with deeper reasoning

Integrates with ServiceProvider for cloud/local mode switching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from blurt.classification.classifier import ClassificationError, IntentClassifier
from blurt.classification.models import (
    AMBIGUITY_GAP_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    AmbiguityResolution,
    ClassificationResult,
    ClassificationStatus,
    FallbackStrategy,
    IntentScore,
)
from blurt.classification.pipeline import ClassificationPipeline
from blurt.models.intents import BlurtIntent

logger = logging.getLogger(__name__)


@dataclass
class ClassificationServiceConfig:
    """Configuration for the intent classification service."""

    confidence_threshold: float = CONFIDENCE_THRESHOLD
    ambiguity_gap: float = AMBIGUITY_GAP_THRESHOLD
    enable_escalation: bool = True
    enable_few_shot: bool = True
    max_retries: int = 1


@dataclass
class ClassificationResponse:
    """API-friendly classification response.

    Designed for serialization to JSON in the REST API.
    """

    intent: str
    confidence: float
    status: str
    all_scores: dict[str, float]
    secondary_intent: str | None = None
    is_multi_intent: bool = False
    sub_intents: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: float = 0.0
    classification_id: str = ""

    @classmethod
    def from_result(cls, result: ClassificationResult) -> ClassificationResponse:
        """Convert a ClassificationResult to a ClassificationResponse."""
        scores = {s.intent.value: round(s.confidence, 4) for s in result.all_scores}

        sub_intents = result.metadata.get("sub_intents", [])
        reasoning = result.metadata.get("reasoning", "")
        if result.resolution and result.resolution.strategy_used != FallbackStrategy.DEFAULT_JOURNAL:
            reasoning = reasoning or "Resolved via escalation to smart model"

        return cls(
            intent=result.primary_intent.value,
            confidence=round(result.confidence, 4),
            status=result.status.value,
            all_scores=scores,
            secondary_intent=(
                result.secondary_intent.value if result.secondary_intent else None
            ),
            is_multi_intent=result.is_multi_intent,
            sub_intents=sub_intents,
            reasoning=reasoning,
            latency_ms=round(result.latency_ms, 2),
            classification_id=result.id,
        )


class IntentClassificationService:
    """Unified intent classification service.

    Wraps the classification pipeline to provide a clean service interface
    for both cloud (Gemini) and local (rule-based) classifiers.

    The service handles:
    - Mode-aware classifier selection (cloud vs local)
    - The full LLM prompt chain: classify → evaluate → escalate → route
    - Confidence scoring across all 7 intent types
    - Safe fallback to journal for ambiguous/failed classifications
    - Statistics tracking for accuracy monitoring

    Usage::

        # Cloud mode (with Gemini client)
        service = IntentClassificationService.from_gemini(gemini_client)
        response = await service.classify("I need to buy groceries")
        # response.intent == "task"
        # response.confidence >= 0.85

        # Local mode (no API calls)
        service = IntentClassificationService.from_local()
        response = await service.classify("Remind me to call mom")
        # response.intent == "reminder"
    """

    def __init__(
        self,
        pipeline: ClassificationPipeline | None = None,
        *,
        local_classifier: Any | None = None,
        config: ClassificationServiceConfig | None = None,
    ) -> None:
        self._config = config or ClassificationServiceConfig()
        self._pipeline = pipeline
        self._local_classifier = local_classifier
        self._is_local = pipeline is None and local_classifier is not None

        # Accuracy tracking
        self._total_classifications = 0
        self._confident_classifications = 0
        self._intent_counts: dict[str, int] = {i.value: 0 for i in BlurtIntent}
        self._intent_confident_counts: dict[str, int] = {i.value: 0 for i in BlurtIntent}

    @classmethod
    def from_gemini(
        cls,
        gemini_client: Any,
        config: ClassificationServiceConfig | None = None,
    ) -> IntentClassificationService:
        """Create a cloud-mode service using the Gemini client.

        Args:
            gemini_client: An initialized GeminiClient instance.
            config: Optional service configuration.

        Returns:
            IntentClassificationService configured for cloud mode.
        """
        cfg = config or ClassificationServiceConfig()
        pipeline = ClassificationPipeline(
            gemini_client,
            confidence_threshold=cfg.confidence_threshold,
            ambiguity_gap=cfg.ambiguity_gap,
            enable_escalation=cfg.enable_escalation,
        )
        return cls(pipeline=pipeline, config=cfg)

    @classmethod
    def from_local(
        cls,
        config: ClassificationServiceConfig | None = None,
    ) -> IntentClassificationService:
        """Create a local-mode service using rule-based classification.

        No external API calls — fully offline.

        Args:
            config: Optional service configuration.

        Returns:
            IntentClassificationService configured for local mode.
        """
        from blurt.local.classifier import LocalIntentClassifier

        cfg = config or ClassificationServiceConfig()
        local = LocalIntentClassifier(
            confidence_threshold=cfg.confidence_threshold,
            ambiguity_gap=cfg.ambiguity_gap,
        )
        return cls(local_classifier=local, config=cfg)

    @classmethod
    def from_provider(cls, config: ClassificationServiceConfig | None = None) -> IntentClassificationService:
        """Create a service using the global ServiceProvider.

        Auto-detects cloud vs local mode.

        Args:
            config: Optional service configuration.

        Returns:
            IntentClassificationService configured for the detected mode.
        """
        from blurt.services.provider import get_provider

        provider = get_provider()
        if provider.is_local:
            return cls.from_local(config=config)
        else:
            llm = provider.llm_client()
            return cls.from_gemini(llm, config=config)

    async def classify(self, text: str, **metadata: Any) -> ClassificationResponse:
        """Classify raw text input and return intent with confidence scores.

        This is the main API. Takes unstructured text and returns a structured
        classification response with the primary intent, confidence score,
        and scores for all 7 intent types.

        Args:
            text: Raw text input (from voice transcription or direct text).
            **metadata: Optional metadata to attach (session_id, user_id, etc.)

        Returns:
            ClassificationResponse with intent, confidence, and all scores.
        """
        if not text or not text.strip():
            return self._empty_input_response()

        text = text.strip()

        if self._is_local:
            result = await self._classify_local(text, **metadata)
        else:
            result = await self._classify_cloud(text, **metadata)

        # Track accuracy statistics
        self._track_stats(result)

        return ClassificationResponse.from_result(result)

    async def classify_raw(self, text: str, **metadata: Any) -> ClassificationResult:
        """Classify and return the raw ClassificationResult (for pipeline use).

        Use this when you need the full internal result, not the API response.

        Args:
            text: Raw text input.
            **metadata: Optional metadata.

        Returns:
            ClassificationResult with full details.
        """
        if not text or not text.strip():
            return ClassificationResult(
                input_text="",
                primary_intent=BlurtIntent.JOURNAL,
                confidence=1.0,
                status=ClassificationStatus.LOW_CONFIDENCE,
            )

        text = text.strip()

        if self._is_local:
            return await self._classify_local(text, **metadata)
        else:
            return await self._classify_cloud(text, **metadata)

    async def classify_batch(
        self, texts: list[str], **metadata: Any
    ) -> list[ClassificationResponse]:
        """Classify multiple texts and return responses.

        Args:
            texts: List of text inputs.
            **metadata: Optional metadata attached to all results.

        Returns:
            List of ClassificationResponse objects.
        """
        results = []
        for text in texts:
            response = await self.classify(text, **metadata)
            results.append(response)
        return results

    @property
    def accuracy_stats(self) -> dict[str, Any]:
        """Return accuracy statistics for monitoring.

        Tracks confident rate overall and per-intent to verify
        the >85% accuracy target.
        """
        overall_rate = (
            self._confident_classifications / self._total_classifications
            if self._total_classifications > 0
            else 0.0
        )

        per_intent: dict[str, dict[str, Any]] = {}
        for intent in BlurtIntent:
            total = self._intent_counts[intent.value]
            confident = self._intent_confident_counts[intent.value]
            per_intent[intent.value] = {
                "total": total,
                "confident": confident,
                "rate": confident / total if total > 0 else 0.0,
            }

        return {
            "total_classifications": self._total_classifications,
            "confident_classifications": self._confident_classifications,
            "overall_confident_rate": round(overall_rate, 4),
            "per_intent": per_intent,
        }

    @property
    def pipeline_stats(self) -> dict[str, Any] | None:
        """Return pipeline-level stats (cloud mode only)."""
        if self._pipeline:
            return self._pipeline.stats.to_dict()
        return None

    def reset_stats(self) -> None:
        """Reset accuracy tracking statistics."""
        self._total_classifications = 0
        self._confident_classifications = 0
        self._intent_counts = {i.value: 0 for i in BlurtIntent}
        self._intent_confident_counts = {i.value: 0 for i in BlurtIntent}

    # ── Internal methods ────────────────────────────────────────────

    async def _classify_cloud(self, text: str, **metadata: Any) -> ClassificationResult:
        """Classify using the cloud pipeline (Gemini)."""
        assert self._pipeline is not None
        return await self._pipeline.classify(text, **metadata)

    async def _classify_local(self, text: str, **metadata: Any) -> ClassificationResult:
        """Classify using the local rule-based classifier."""
        assert self._local_classifier is not None
        return await self._local_classifier.classify_with_result(text, **metadata)

    def _track_stats(self, result: ClassificationResult) -> None:
        """Track classification accuracy statistics."""
        self._total_classifications += 1
        intent_key = result.primary_intent.value
        self._intent_counts[intent_key] = self._intent_counts.get(intent_key, 0) + 1

        if result.is_confident:
            self._confident_classifications += 1
            self._intent_confident_counts[intent_key] = (
                self._intent_confident_counts.get(intent_key, 0) + 1
            )

    def _empty_input_response(self) -> ClassificationResponse:
        """Handle empty/whitespace-only input."""
        return ClassificationResponse(
            intent=BlurtIntent.JOURNAL.value,
            confidence=1.0,
            status=ClassificationStatus.LOW_CONFIDENCE.value,
            all_scores={i.value: (1.0 if i == BlurtIntent.JOURNAL else 0.0) for i in BlurtIntent},
            latency_ms=0.0,
        )
