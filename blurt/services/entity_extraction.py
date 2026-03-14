"""Entity extraction service for the Blurt pipeline.

Provides a unified interface for entity extraction that works seamlessly
across cloud (Gemini Flash-Lite) and local (rule-based) modes. Bridges
the extraction layer to the capture pipeline by converting extraction
results to EntityRef objects.

This service is the integration point between:
- blurt.extraction.entities.EntityExtractor (cloud, LLM-based)
- blurt.local.extractor.LocalEntityExtractor (offline, rule-based)
- blurt.services.capture.EntityExtractorFunc protocol

Usage::

    # Standalone
    service = EntityExtractionService()
    result = await service.extract("Sarah and I are meeting at Google HQ tomorrow")
    for entity in result.entities:
        print(f"{entity.name} ({entity.entity_type.value})")

    # As capture pipeline callable
    pipeline = BlurtCapturePipeline(
        entity_extractor=service.as_pipeline_extractor(),
    )

    # With ServiceProvider (auto-detects cloud vs local)
    service = EntityExtractionService.from_provider(provider)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from blurt.extraction.entities import (
    EntityExtractor,
    ExtractedEntity,
    ExtractionResult,
)
from blurt.local.extractor import LocalEntityExtractor
from blurt.memory.episodic import EntityRef

logger = logging.getLogger(__name__)


# ── Statistics tracking ──────────────────────────────────────────


@dataclass
class EntityExtractionStats:
    """Aggregate statistics for entity extraction across the service lifetime.

    Tracks extraction counts, error rates, entity type distributions, and
    latency metrics for monitoring and diagnostics.
    """

    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    fallback_extractions: int = 0
    total_entities_extracted: int = 0
    total_relationships_extracted: int = 0
    total_facts_extracted: int = 0
    total_temporal_refs_extracted: int = 0
    total_latency_ms: float = 0.0

    # Entity type distribution
    entity_type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        """Average extraction latency in milliseconds."""
        if self.total_extractions == 0:
            return 0.0
        return self.total_latency_ms / self.total_extractions

    @property
    def success_rate(self) -> float:
        """Fraction of extractions that succeeded (0.0–1.0)."""
        if self.total_extractions == 0:
            return 0.0
        return self.successful_extractions / self.total_extractions

    @property
    def avg_entities_per_extraction(self) -> float:
        """Average number of entities per extraction."""
        if self.successful_extractions == 0:
            return 0.0
        return self.total_entities_extracted / self.successful_extractions

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "total_extractions": self.total_extractions,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "fallback_extractions": self.fallback_extractions,
            "total_entities_extracted": self.total_entities_extracted,
            "total_relationships_extracted": self.total_relationships_extracted,
            "total_facts_extracted": self.total_facts_extracted,
            "total_temporal_refs_extracted": self.total_temporal_refs_extracted,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "avg_entities_per_extraction": round(self.avg_entities_per_extraction, 2),
            "entity_type_counts": dict(self.entity_type_counts),
        }


# ── Entity extraction service ───────────────────────────────────


class EntityExtractionService:
    """Unified entity extraction service for the Blurt pipeline.

    Wraps both cloud (Gemini Flash-Lite) and local (rule-based) extractors
    behind a single interface. Provides:

    - ``extract(text)`` — full extraction returning ExtractionResult
    - ``extract_entity_refs(text)`` — extract as EntityRef list (for capture pipeline)
    - ``as_pipeline_extractor()`` — returns a callable matching EntityExtractorFunc
    - ``extract_batch(texts)`` — extract from multiple texts concurrently
    - Statistics tracking for monitoring

    The service automatically falls back from cloud to local extraction
    when the LLM client is unavailable, ensuring zero-drop behavior.
    """

    def __init__(
        self,
        cloud_extractor: EntityExtractor | None = None,
        local_extractor: LocalEntityExtractor | None = None,
        *,
        min_confidence: float = 0.3,
        max_entities_per_extraction: int = 50,
        prefer_cloud: bool = True,
    ) -> None:
        """Initialize the entity extraction service.

        Args:
            cloud_extractor: LLM-based extractor (Gemini Flash-Lite).
                If None, only local extraction is available.
            local_extractor: Rule-based offline extractor.
                Created automatically if not provided.
            min_confidence: Minimum confidence threshold for entities.
            max_entities_per_extraction: Safety limit on entities per call.
            prefer_cloud: If True, try cloud extraction first with local fallback.
                If False, always use local extraction.
        """
        self._cloud = cloud_extractor
        self._local = local_extractor or LocalEntityExtractor(
            min_confidence=min_confidence,
            max_entities_per_extraction=max_entities_per_extraction,
        )
        self._min_confidence = min_confidence
        self._prefer_cloud = prefer_cloud
        self._stats = EntityExtractionStats()

    @classmethod
    def from_provider(
        cls,
        provider: Any,
        *,
        min_confidence: float = 0.3,
        max_entities_per_extraction: int = 50,
    ) -> EntityExtractionService:
        """Create an EntityExtractionService from a ServiceProvider.

        Detects cloud vs local mode and configures extractors accordingly.

        Args:
            provider: A ServiceProvider instance.
            min_confidence: Minimum confidence threshold.
            max_entities_per_extraction: Safety limit on entities.

        Returns:
            Configured EntityExtractionService.
        """
        from blurt.services.provider import ServiceProvider

        if not isinstance(provider, ServiceProvider):
            raise TypeError(
                f"Expected ServiceProvider, got {type(provider).__name__}"
            )

        # Determine if we're in cloud mode
        env_info = provider.environment
        is_cloud = env_info.mode == "cloud"

        cloud_extractor: EntityExtractor | None = None
        if is_cloud:
            try:
                llm = provider.llm_client()
                cloud_extractor = EntityExtractor(
                    gemini_client=llm,  # type: ignore[arg-type]
                    min_confidence=min_confidence,
                    max_entities_per_extraction=max_entities_per_extraction,
                )
            except Exception as e:
                logger.warning(
                    "Failed to create cloud extractor from provider: %s", e
                )

        local_extractor = LocalEntityExtractor(
            min_confidence=min_confidence,
            max_entities_per_extraction=max_entities_per_extraction,
        )

        return cls(
            cloud_extractor=cloud_extractor,
            local_extractor=local_extractor,
            min_confidence=min_confidence,
            max_entities_per_extraction=max_entities_per_extraction,
            prefer_cloud=is_cloud,
        )

    @classmethod
    def cloud(
        cls,
        gemini_client: Any,
        *,
        min_confidence: float = 0.3,
        max_entities_per_extraction: int = 50,
    ) -> EntityExtractionService:
        """Create a cloud-mode service with an existing Gemini client.

        Args:
            gemini_client: Connected GeminiClient instance.
            min_confidence: Minimum confidence threshold.
            max_entities_per_extraction: Safety limit.

        Returns:
            EntityExtractionService configured for cloud extraction.
        """
        return cls(
            cloud_extractor=EntityExtractor(
                gemini_client=gemini_client,
                min_confidence=min_confidence,
                max_entities_per_extraction=max_entities_per_extraction,
            ),
            min_confidence=min_confidence,
            max_entities_per_extraction=max_entities_per_extraction,
            prefer_cloud=True,
        )

    @classmethod
    def local(
        cls,
        *,
        min_confidence: float = 0.3,
        max_entities_per_extraction: int = 50,
    ) -> EntityExtractionService:
        """Create a local-only service (no API calls).

        Args:
            min_confidence: Minimum confidence threshold.
            max_entities_per_extraction: Safety limit.

        Returns:
            EntityExtractionService configured for local-only extraction.
        """
        return cls(
            cloud_extractor=None,
            min_confidence=min_confidence,
            max_entities_per_extraction=max_entities_per_extraction,
            prefer_cloud=False,
        )

    # ── Core extraction ──────────────────────────────────────────

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities, relationships, facts, and temporals from text.

        Tries cloud extraction first (if available and preferred), falling
        back to local extraction. Empty/whitespace input returns an empty
        result — never raises.

        Args:
            text: Natural speech text to analyze.

        Returns:
            ExtractionResult with all identified entities and metadata.
        """
        if not text or not text.strip():
            return ExtractionResult(raw_text=text or "")

        self._stats.total_extractions += 1
        start = time.monotonic()

        try:
            result = await self._do_extract(text.strip())
            self._stats.successful_extractions += 1
            self._record_result_stats(result)
            return result
        except Exception as e:
            self._stats.failed_extractions += 1
            logger.error("Entity extraction failed completely: %s", e)
            # Return empty result — never drop the blurt
            return ExtractionResult(raw_text=text)
        finally:
            elapsed = (time.monotonic() - start) * 1000
            self._stats.total_latency_ms += elapsed

    async def _do_extract(self, text: str) -> ExtractionResult:
        """Internal extraction with cloud-first, local-fallback strategy."""
        if self._prefer_cloud and self._cloud is not None:
            try:
                result = await self._cloud.extract(text)
                # If cloud returned something useful, use it
                if result.model_used != "regex-fallback":
                    return result
                # Cloud fell back to regex — use our enhanced local extractor
                self._stats.fallback_extractions += 1
            except Exception as e:
                logger.warning(
                    "Cloud extraction failed, falling back to local: %s", e
                )
                self._stats.fallback_extractions += 1

        # Local extraction (always available)
        return await self._local.extract(text)

    async def extract_batch(
        self, texts: list[str]
    ) -> list[ExtractionResult]:
        """Extract entities from multiple texts concurrently.

        Args:
            texts: List of speech texts to analyze.

        Returns:
            List of ExtractionResult, one per input text.
        """
        import asyncio

        tasks = [self.extract(text) for text in texts]
        return list(await asyncio.gather(*tasks))

    # ── Pipeline integration ─────────────────────────────────────

    async def extract_entity_refs(self, text: str) -> list[EntityRef]:
        """Extract entities and return as EntityRef list.

        This is the primary integration point with the capture pipeline.
        Converts ExtractedEntity objects to EntityRef objects matching
        the EntityExtractorFunc protocol.

        Args:
            text: Natural speech text to analyze.

        Returns:
            List of EntityRef objects for the episodic memory store.
        """
        result = await self.extract(text)
        return self._to_entity_refs(result)

    def as_pipeline_extractor(self) -> _PipelineExtractorWrapper:
        """Return a callable matching the EntityExtractorFunc protocol.

        Use this to plug the service into BlurtCapturePipeline::

            pipeline = BlurtCapturePipeline(
                entity_extractor=service.as_pipeline_extractor(),
            )

        Returns:
            A callable ``async (text: str) -> list[EntityRef]``.
        """
        return _PipelineExtractorWrapper(self)

    # ── Conversion helpers ───────────────────────────────────────

    @staticmethod
    def _to_entity_refs(result: ExtractionResult) -> list[EntityRef]:
        """Convert ExtractionResult entities to EntityRef objects."""
        refs: list[EntityRef] = []
        for entity in result.entities:
            refs.append(
                EntityRef(
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    confidence=entity.confidence,
                )
            )
        return refs

    @staticmethod
    def entity_to_ref(entity: ExtractedEntity) -> EntityRef:
        """Convert a single ExtractedEntity to an EntityRef.

        Useful for custom pipeline integrations.

        Args:
            entity: The extracted entity.

        Returns:
            An EntityRef suitable for episodic memory.
        """
        return EntityRef(
            name=entity.name,
            entity_type=entity.entity_type.value,
            confidence=entity.confidence,
        )

    # ── Statistics ───────────────────────────────────────────────

    def _record_result_stats(self, result: ExtractionResult) -> None:
        """Update internal stats from an extraction result."""
        self._stats.total_entities_extracted += result.entity_count
        self._stats.total_relationships_extracted += len(result.relationships)
        self._stats.total_facts_extracted += len(result.facts)
        self._stats.total_temporal_refs_extracted += len(
            result.temporal_references
        )

        for entity in result.entities:
            type_key = entity.entity_type.value
            self._stats.entity_type_counts[type_key] = (
                self._stats.entity_type_counts.get(type_key, 0) + 1
            )

    @property
    def stats(self) -> EntityExtractionStats:
        """Current extraction statistics."""
        return self._stats

    @property
    def mode(self) -> str:
        """Current extraction mode: 'cloud', 'local', or 'hybrid'."""
        if self._prefer_cloud and self._cloud is not None:
            return "cloud"
        return "local"

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self._stats = EntityExtractionStats()


# ── Pipeline wrapper ─────────────────────────────────────────────


class _PipelineExtractorWrapper:
    """Wraps EntityExtractionService to match EntityExtractorFunc protocol.

    This callable adapter converts extraction results to EntityRef lists,
    bridging the extraction service to the capture pipeline.
    """

    def __init__(self, service: EntityExtractionService) -> None:
        self._service = service

    async def __call__(self, text: str) -> list[EntityRef]:
        """Extract entities as EntityRef list — matches EntityExtractorFunc."""
        return await self._service.extract_entity_refs(text)
