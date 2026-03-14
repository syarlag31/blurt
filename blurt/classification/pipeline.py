"""Classification pipeline controller.

Orchestrates the complete classification flow for every user input:
1. Invoke the intent classifier (Flash-Lite, fast)
2. Evaluate confidence against 85% threshold
3. Handle ambiguity/low-confidence via fallback strategies
4. Route the classified result downstream

Every blurt flows through this pipeline without exception.
No user-facing categorization is required — classification is silent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from blurt.classification.classifier import ClassificationError, IntentClassifier
from blurt.classification.models import (
    AMBIGUITY_GAP_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    MULTI_INTENT_THRESHOLD,
    AmbiguityResolution,
    ClassificationResult,
    ClassificationStatus,
    FallbackStrategy,
    IntentScore,
)
from blurt.clients.gemini import GeminiClient
from blurt.models.intents import BlurtIntent

logger = logging.getLogger(__name__)

# Type alias for downstream handlers
DownstreamHandler = Callable[[ClassificationResult], Coroutine[Any, Any, None]]


@dataclass
class PipelineStats:
    """Operational statistics for the classification pipeline."""

    total_classified: int = 0
    confident_count: int = 0
    ambiguous_count: int = 0
    low_confidence_count: int = 0
    escalated_count: int = 0
    multi_intent_count: int = 0
    error_count: int = 0
    fallback_to_journal_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_classified == 0:
            return 0.0
        return self.total_latency_ms / self.total_classified

    @property
    def confident_rate(self) -> float:
        if self.total_classified == 0:
            return 0.0
        return (self.confident_count + self.escalated_count) / self.total_classified

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_classified": self.total_classified,
            "confident_count": self.confident_count,
            "ambiguous_count": self.ambiguous_count,
            "low_confidence_count": self.low_confidence_count,
            "escalated_count": self.escalated_count,
            "multi_intent_count": self.multi_intent_count,
            "error_count": self.error_count,
            "fallback_to_journal_count": self.fallback_to_journal_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "confident_rate": round(self.confident_rate, 4),
        }


class ClassificationPipeline:
    """Orchestrates silent intent classification for every user input.

    The pipeline:
    1. Receives raw text (transcribed from voice or typed)
    2. Classifies via Flash-Lite (fast, cheap)
    3. Evaluates confidence:
       - >= 85%: Accept classification, route downstream
       - < 85% with ambiguity: Escalate to Flash (smart) for resolution
       - < 85% after escalation: Default to journal (safe, shame-free fallback)
    4. Routes the ClassificationResult to registered downstream handlers

    Design principles:
    - Zero friction: No user interaction needed for classification
    - Anti-shame: Journal is the safe default — capturing is always valuable
    - Pipeline integrity: Every input produces a ClassificationResult, no data loss

    Usage::

        pipeline = ClassificationPipeline(gemini_client)
        pipeline.register_handler(my_downstream_handler)

        result = await pipeline.classify("I need to buy groceries tomorrow")
        # result.primary_intent == BlurtIntent.TASK
        # result.confidence == 0.93
        # result.status == ClassificationStatus.CONFIDENT
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ambiguity_gap: float = AMBIGUITY_GAP_THRESHOLD,
        multi_intent_threshold: float = MULTI_INTENT_THRESHOLD,
        enable_escalation: bool = True,
    ) -> None:
        """Initialize the classification pipeline.

        Args:
            client: Gemini API client for model access.
            confidence_threshold: Minimum confidence for accepting a classification.
            ambiguity_gap: Minimum gap between top-2 intents to avoid ambiguity.
            multi_intent_threshold: Threshold for detecting multiple intents.
            enable_escalation: Whether to escalate ambiguous cases to the smart model.
        """
        self._classifier = IntentClassifier(client)
        self._client = client
        self._handlers: list[DownstreamHandler] = []
        self._confidence_threshold = confidence_threshold
        self._ambiguity_gap = ambiguity_gap
        self._multi_intent_threshold = multi_intent_threshold
        self._enable_escalation = enable_escalation
        self._stats = PipelineStats()

    @property
    def stats(self) -> PipelineStats:
        """Pipeline operational statistics."""
        return self._stats

    def register_handler(self, handler: DownstreamHandler) -> None:
        """Register a downstream handler for classified results.

        Handlers are called in registration order after classification.
        Each handler receives the complete ClassificationResult.

        Args:
            handler: Async callable that receives a ClassificationResult.
        """
        self._handlers.append(handler)

    def unregister_handler(self, handler: DownstreamHandler) -> None:
        """Remove a downstream handler.

        Args:
            handler: The handler to remove.
        """
        self._handlers = [h for h in self._handlers if h is not handler]

    async def classify(self, text: str, **metadata: Any) -> ClassificationResult:
        """Classify user input and route downstream.

        This is the main entry point. Every blurt flows through here.

        Args:
            text: User input text (transcribed from voice or typed).
            **metadata: Additional metadata to attach to the result.

        Returns:
            ClassificationResult with intent, confidence, and status.
        """
        start = time.monotonic()
        result = ClassificationResult(input_text=text, metadata=metadata)

        try:
            # Step 1: Initial classification via Flash-Lite
            scores = await self._classifier.classify(text)
            elapsed = (time.monotonic() - start) * 1000
            result.all_scores = scores
            result.model_used = "flash-lite"
            result.latency_ms = elapsed

            if not scores:
                result.status = ClassificationStatus.ERROR
                result.metadata["error"] = "No scores returned"
                self._stats.error_count += 1
                self._apply_safe_default(result)
                return await self._finalize(result)

            primary = scores[0]
            result.primary_intent = primary.intent
            result.confidence = primary.confidence

            # Step 2: Evaluate confidence
            status = self._evaluate_confidence(scores)
            result.status = status

            # Step 3: Handle based on status
            if status == ClassificationStatus.CONFIDENT:
                self._stats.confident_count += 1
            elif status in (
                ClassificationStatus.AMBIGUOUS,
                ClassificationStatus.LOW_CONFIDENCE,
            ):
                if status == ClassificationStatus.AMBIGUOUS:
                    self._stats.ambiguous_count += 1
                else:
                    self._stats.low_confidence_count += 1

                # Attempt escalation to smart model
                if self._enable_escalation:
                    result = await self._handle_escalation(result, text, scores)
                else:
                    # No escalation — use safe default
                    self._apply_safe_default(result)
                    self._stats.fallback_to_journal_count += 1

        except ClassificationError as e:
            logger.error("Classification error: %s", e)
            result.status = ClassificationStatus.ERROR
            result.metadata["error"] = str(e)
            self._apply_safe_default(result)
            self._stats.error_count += 1
        except Exception as e:
            logger.exception("Unexpected error in classification pipeline")
            result.status = ClassificationStatus.ERROR
            result.metadata["error"] = f"Unexpected: {e}"
            self._apply_safe_default(result)
            self._stats.error_count += 1

        result.latency_ms = (time.monotonic() - start) * 1000
        return await self._finalize(result)

    def _evaluate_confidence(self, scores: list[IntentScore]) -> ClassificationStatus:
        """Evaluate classification confidence and detect ambiguity.

        Args:
            scores: Sorted list of IntentScores (highest first).

        Returns:
            ClassificationStatus based on confidence analysis.
        """
        if not scores:
            return ClassificationStatus.LOW_CONFIDENCE

        primary = scores[0]

        # Clear confident classification
        if primary.confidence >= self._confidence_threshold:
            # Check for multi-intent: secondary also scores high
            if len(scores) >= 2:
                secondary = scores[1]
                if secondary.confidence >= self._multi_intent_threshold:
                    return ClassificationStatus.AMBIGUOUS  # Might be multi-intent
            return ClassificationStatus.CONFIDENT

        # Below threshold — check if it's ambiguous (close top scores)
        if len(scores) >= 2:
            gap = primary.confidence - scores[1].confidence
            if gap < self._ambiguity_gap:
                return ClassificationStatus.AMBIGUOUS

        return ClassificationStatus.LOW_CONFIDENCE

    async def _handle_escalation(
        self,
        result: ClassificationResult,
        text: str,
        original_scores: list[IntentScore],
    ) -> ClassificationResult:
        """Escalate to the smart model for ambiguity resolution.

        Args:
            result: Current classification result to update.
            text: Original user input text.
            original_scores: Scores from the initial classification.

        Returns:
            Updated ClassificationResult.
        """
        escalation_start = time.monotonic()

        try:
            resolved = await self._classifier.resolve_ambiguity(text)
            resolution_latency = (time.monotonic() - escalation_start) * 1000

            resolved_intent = resolved["primary_intent"]
            resolved_confidence = resolved["confidence"]
            is_multi = resolved.get("multi_intent", False)

            if is_multi:
                result.status = ClassificationStatus.MULTI_INTENT
                result.primary_intent = resolved_intent
                result.confidence = resolved_confidence
                result.metadata["sub_intents"] = resolved.get("intents", [])
                result.metadata["reasoning"] = resolved.get("reasoning", "")
                self._stats.multi_intent_count += 1
            elif resolved_confidence >= self._confidence_threshold:
                result.status = ClassificationStatus.RESOLVED
                result.primary_intent = resolved_intent
                result.confidence = resolved_confidence
                self._stats.escalated_count += 1
            else:
                # Smart model also unsure — use safe default
                result.status = ClassificationStatus.LOW_CONFIDENCE
                self._apply_safe_default(result)
                self._stats.fallback_to_journal_count += 1

            result.resolution = AmbiguityResolution(
                original_status=ClassificationStatus.AMBIGUOUS
                if result.was_ambiguous
                else ClassificationStatus.LOW_CONFIDENCE,
                strategy_used=(
                    FallbackStrategy.MULTI_INTENT_SPLIT
                    if is_multi
                    else FallbackStrategy.ESCALATE_TO_SMART
                ),
                original_scores=original_scores,
                resolved_intent=result.primary_intent,
                resolved_confidence=result.confidence,
                resolution_model="flash",
                resolution_latency_ms=resolution_latency,
            )

        except Exception as e:
            logger.warning("Escalation failed, using safe default: %s", e)
            self._apply_safe_default(result)
            result.resolution = AmbiguityResolution(
                original_status=result.status,
                strategy_used=FallbackStrategy.DEFAULT_JOURNAL,
                original_scores=original_scores,
                resolved_intent=result.primary_intent,
                resolved_confidence=result.confidence,
            )
            self._stats.fallback_to_journal_count += 1

        return result

    @staticmethod
    def _apply_safe_default(result: ClassificationResult) -> None:
        """Apply the safe default classification (journal).

        Journal is the anti-shame fallback — capturing a thought is always
        valuable, and the user can reclassify later via text edit.
        """
        result.primary_intent = BlurtIntent.JOURNAL
        result.confidence = 1.0  # We're 100% confident it's at least a journal entry

    async def _finalize(self, result: ClassificationResult) -> ClassificationResult:
        """Finalize and route the classification result downstream.

        Args:
            result: The completed ClassificationResult.

        Returns:
            The same result after routing.
        """
        self._stats.total_classified += 1
        self._stats.total_latency_ms += result.latency_ms

        logger.info(
            "Classified blurt: intent=%s confidence=%.2f status=%s latency=%.0fms",
            result.primary_intent.value,
            result.confidence,
            result.status.value,
            result.latency_ms,
        )

        # Route to all downstream handlers
        for handler in self._handlers:
            try:
                await handler(result)
            except Exception:
                logger.exception(
                    "Downstream handler %s failed for blurt %s",
                    handler.__name__ if hasattr(handler, "__name__") else handler,
                    result.id,
                )

        return result
