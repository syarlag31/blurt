"""Adapter to bridge ClassificationPipeline into the capture pipeline.

The capture pipeline expects a simple callable:
    async (text: str) -> tuple[str, float]

The ClassificationPipeline returns rich ClassificationResult objects.
This adapter bridges the two, running the full classification pipeline
(including ambiguity resolution) and returning (intent, confidence)
for the capture pipeline while preserving full results for downstream use.

This ensures every input is silently classified through the full pipeline
without any user-facing interruption.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from blurt.classification.models import ClassificationResult
from blurt.classification.pipeline import ClassificationPipeline
from blurt.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)

# Type for callbacks that receive the full classification result
ClassificationCallback = Callable[[ClassificationResult], Coroutine[Any, Any, None]]


class ClassificationAdapter:
    """Adapts the ClassificationPipeline for use in the capture pipeline.

    Provides a callable interface matching the capture pipeline's
    ClassifierFunc protocol: async (text: str) -> tuple[str, float]

    Additionally stores the last classification result for downstream
    consumers that need the full result object.

    Usage::

        adapter = ClassificationAdapter(gemini_client)
        intent, confidence = await adapter("I need to buy groceries")
        # intent == "task", confidence == 0.92

        # Access full result
        full_result = adapter.last_result
        # full_result.all_scores, full_result.status, etc.
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        *,
        pipeline: ClassificationPipeline | None = None,
        on_classified: ClassificationCallback | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            client: Gemini client (creates a new pipeline).
            pipeline: Existing pipeline to reuse (takes priority over client).
            on_classified: Optional callback invoked with every classification result.
        """
        if pipeline is not None:
            self._pipeline = pipeline
        elif client is not None:
            self._pipeline = ClassificationPipeline(client)
        else:
            raise ValueError("Either client or pipeline must be provided")

        self._on_classified = on_classified
        self._last_result: ClassificationResult | None = None
        self._total_calls: int = 0

    @property
    def pipeline(self) -> ClassificationPipeline:
        """Access the underlying ClassificationPipeline."""
        return self._pipeline

    @property
    def last_result(self) -> ClassificationResult | None:
        """The most recent full ClassificationResult."""
        return self._last_result

    @property
    def total_calls(self) -> int:
        """Total number of classifications performed through this adapter."""
        return self._total_calls

    async def __call__(self, text: str) -> tuple[str, float]:
        """Classify text and return (intent, confidence) tuple.

        This is the main interface matching the capture pipeline's
        ClassifierFunc protocol.

        Classification runs silently through the full pipeline:
        1. Flash-Lite initial classification
        2. Ambiguity resolution via Flash if needed
        3. Safe journal fallback if all else fails

        Args:
            text: User input text to classify.

        Returns:
            Tuple of (intent_name, confidence_score).
        """
        result = await self._pipeline.classify(text)
        self._last_result = result
        self._total_calls += 1

        # Invoke callback if registered
        if self._on_classified is not None:
            try:
                await self._on_classified(result)
            except Exception:
                logger.exception("Classification callback failed")

        return (result.primary_intent.value, result.confidence)


def create_classification_adapter(
    client: GeminiClient,
    *,
    on_classified: ClassificationCallback | None = None,
) -> ClassificationAdapter:
    """Factory for creating a ClassificationAdapter.

    Args:
        client: Gemini client for model access.
        on_classified: Optional callback for each classification result.

    Returns:
        Configured ClassificationAdapter ready for use in capture pipeline.
    """
    return ClassificationAdapter(client, on_classified=on_classified)
