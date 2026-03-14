"""Tests for the EntityExtractionService.

Tests cover:
- Cloud-mode extraction with mocked Gemini responses
- Local-only extraction (rule-based, no API calls)
- Cloud-to-local fallback when LLM unavailable
- Pipeline integration via as_pipeline_extractor()
- All 6 entity types: person, place, project, organization, topic, tool
- Relationship extraction
- Fact/preference/habit extraction
- Temporal reference extraction
- Batch extraction
- Statistics tracking
- Entity-to-EntityRef conversion
- Edge cases (empty input, errors, degraded mode)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from blurt.extraction.entities import (
    EntityExtractor,
    ExtractedEntity,
    ExtractionResult,
    TemporalType,
)
from blurt.local.extractor import LocalEntityExtractor
from blurt.memory.episodic import EntityRef
from blurt.models.entities import EntityType, FactType, RelationshipType
from blurt.services.entity_extraction import (
    EntityExtractionService,
    EntityExtractionStats,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_gemini_response(
    data: dict[str, Any],
    model: str = "gemini-2.5-flash-lite",
    latency_ms: float = 25.0,
) -> MagicMock:
    """Create a mock GeminiResponse."""
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.model = model
    resp.latency_ms = latency_ms
    return resp


def _make_cloud_extractor(
    response_data: dict[str, Any] | None = None,
    is_connected: bool = True,
) -> EntityExtractor:
    """Create an EntityExtractor backed by a mock Gemini client."""
    client = MagicMock()
    client.is_connected = is_connected

    if response_data is not None:
        resp = _make_gemini_response(response_data)
        client.generate = AsyncMock(return_value=resp)
    else:
        client.generate = AsyncMock(return_value=_make_gemini_response({
            "entities": [],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        }))

    return EntityExtractor(client)


def _make_cloud_response_with_entities(*entities: dict[str, Any]) -> dict[str, Any]:
    """Shorthand to build a full extraction response JSON."""
    return {
        "entities": list(entities),
        "relationships": [],
        "facts": [],
        "temporal_references": [],
    }


# ── Service initialization ───────────────────────────────────────


class TestServiceInit:
    """Test EntityExtractionService construction."""

    def test_default_creates_local_extractor(self):
        service = EntityExtractionService()
        assert service._local is not None
        assert service._cloud is None
        assert service.mode == "local"

    def test_local_factory(self):
        service = EntityExtractionService.local()
        assert service.mode == "local"
        assert service._cloud is None

    def test_cloud_factory(self):
        client = MagicMock()
        client.is_connected = True
        service = EntityExtractionService.cloud(client)
        assert service.mode == "cloud"
        assert service._cloud is not None

    def test_custom_min_confidence(self):
        service = EntityExtractionService(min_confidence=0.5)
        assert service._min_confidence == 0.5

    def test_stats_initially_zero(self):
        service = EntityExtractionService()
        assert service.stats.total_extractions == 0
        assert service.stats.success_rate == 0.0


# ── Cloud extraction ─────────────────────────────────────────────


class TestCloudExtraction:
    """Test extraction via mocked Gemini Flash-Lite."""

    @pytest.mark.asyncio
    async def test_extracts_person_entity(self):
        cloud = _make_cloud_extractor(_make_cloud_response_with_entities(
            {"name": "Sarah", "type": "person", "confidence": 0.95},
        ))
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract("I need to talk to Sarah")

        assert result.entity_count == 1
        assert result.entities[0].name == "Sarah"
        assert result.entities[0].entity_type == EntityType.PERSON
        assert result.entities[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_extracts_all_six_entity_types(self):
        cloud = _make_cloud_extractor({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.95},
                {"name": "San Francisco", "type": "place", "confidence": 0.9},
                {"name": "Project Alpha", "type": "project", "confidence": 0.88},
                {"name": "Acme Corp", "type": "organization", "confidence": 0.92},
                {"name": "machine learning", "type": "topic", "confidence": 0.85},
                {"name": "Notion", "type": "tool", "confidence": 0.98},
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract(
            "Sarah at Acme Corp in San Francisco uses Notion for "
            "Project Alpha about machine learning"
        )

        assert result.entity_count == 6
        types = result.entity_types_found
        assert EntityType.PERSON in types
        assert EntityType.PLACE in types
        assert EntityType.PROJECT in types
        assert EntityType.ORGANIZATION in types
        assert EntityType.TOPIC in types
        assert EntityType.TOOL in types

    @pytest.mark.asyncio
    async def test_extracts_relationships(self):
        cloud = _make_cloud_extractor({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.95},
            ],
            "relationships": [
                {
                    "source": "speaker",
                    "target": "Sarah",
                    "type": "managed_by",
                    "confidence": 0.9,
                },
            ],
            "facts": [],
            "temporal_references": [],
        })
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract("Sarah is my manager")

        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.source_name == "speaker"
        assert rel.target_name == "Sarah"
        assert rel.relationship_type == RelationshipType.MANAGED_BY

    @pytest.mark.asyncio
    async def test_extracts_facts(self):
        cloud = _make_cloud_extractor({
            "entities": [],
            "relationships": [],
            "facts": [
                {
                    "content": "Prefers morning meetings",
                    "type": "preference",
                    "subject": None,
                    "confidence": 0.85,
                },
            ],
            "temporal_references": [],
        })
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract("I prefer morning meetings")

        assert len(result.facts) == 1
        assert result.facts[0].fact_type == FactType.PREFERENCE

    @pytest.mark.asyncio
    async def test_extracts_temporal_references(self):
        cloud = _make_cloud_extractor({
            "entities": [],
            "relationships": [],
            "facts": [],
            "temporal_references": [
                {"text": "tomorrow at 3pm", "type": "relative"},
                {"text": "every Monday", "type": "recurring"},
            ],
        })
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract("Meet tomorrow at 3pm, every Monday")

        assert len(result.temporal_references) == 2
        assert result.temporal_references[0].temporal_type == TemporalType.RELATIVE
        assert result.temporal_references[1].temporal_type == TemporalType.RECURRING


# ── Local extraction ─────────────────────────────────────────────


class TestLocalExtraction:
    """Test local-only (rule-based) extraction."""

    @pytest.mark.asyncio
    async def test_extracts_person_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("I talked to Sarah about the project")

        names = {e.name for e in result.entities}
        assert "Sarah" in names

    @pytest.mark.asyncio
    async def test_extracts_tools_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("I need to update the Notion workspace and check Slack")

        tool_names = {e.normalized_name for e in result.entities if e.entity_type == EntityType.TOOL}
        assert "notion" in tool_names
        assert "slack" in tool_names

    @pytest.mark.asyncio
    async def test_extracts_temporal_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("Meeting tomorrow at 3pm")

        temporal_texts = {t.text.lower() for t in result.temporal_references}
        assert any("tomorrow" in t for t in temporal_texts)

    @pytest.mark.asyncio
    async def test_extracts_preferences_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("I prefer morning meetings over afternoon ones")

        pref_facts = [f for f in result.facts if f.fact_type == FactType.PREFERENCE]
        assert len(pref_facts) >= 1

    @pytest.mark.asyncio
    async def test_extracts_habits_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("I usually skip lunch on busy days")

        habit_facts = [f for f in result.facts if f.fact_type == FactType.HABIT]
        assert len(habit_facts) >= 1

    @pytest.mark.asyncio
    async def test_extracts_relationships_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("Sarah is my manager")

        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.MANAGED_BY in rel_types

    @pytest.mark.asyncio
    async def test_extracts_recurring_temporal_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("I exercise every Monday morning")

        temporal_types = {t.temporal_type.value for t in result.temporal_references}
        assert "recurring" in temporal_types

    @pytest.mark.asyncio
    async def test_extracts_deadline_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("Finish the report by next Friday")

        deadlines = [t for t in result.temporal_references if t.is_deadline]
        assert len(deadlines) > 0

    @pytest.mark.asyncio
    async def test_extracts_alias_locally(self):
        service = EntityExtractionService.local()
        result = await service.extract("the deck means Q2 planning deck")

        fact_types = {f.fact_type for f in result.facts}
        assert FactType.ALIAS in fact_types


# ── Fallback behavior ────────────────────────────────────────────


class TestFallback:
    """Test cloud-to-local fallback when LLM is unavailable."""

    @pytest.mark.asyncio
    async def test_falls_back_when_cloud_disconnected(self):
        cloud = _make_cloud_extractor(is_connected=False)
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract("Sarah mentioned the project at Google")

        # Should still produce results via local fallback
        assert result is not None
        assert result.raw_text == "Sarah mentioned the project at Google"

    @pytest.mark.asyncio
    async def test_falls_back_on_cloud_error(self):
        client = MagicMock()
        client.is_connected = True
        client.generate = AsyncMock(side_effect=Exception("API error"))
        cloud = EntityExtractor(client)

        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract("Sarah at Google")

        # Fallback should still find entities
        assert result is not None
        names = {e.name for e in result.entities}
        assert "Sarah" in names

    @pytest.mark.asyncio
    async def test_fallback_increments_stats(self):
        client = MagicMock()
        client.is_connected = True
        client.generate = AsyncMock(side_effect=Exception("API error"))
        cloud = EntityExtractor(client)

        service = EntityExtractionService(cloud_extractor=cloud)
        await service.extract("Hello Sarah")

        assert service.stats.fallback_extractions >= 1

    @pytest.mark.asyncio
    async def test_local_only_never_falls_back(self):
        service = EntityExtractionService.local()
        result = await service.extract("Sarah at Google HQ")

        assert result is not None
        assert service.stats.fallback_extractions == 0


# ── Empty / edge-case inputs ─────────────────────────────────────


class TestEdgeCases:
    """Test empty inputs and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_string(self):
        service = EntityExtractionService()
        result = await service.extract("")
        assert result.entity_count == 0
        assert not result.has_entities

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        service = EntityExtractionService()
        result = await service.extract("   \n\t  ")
        assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_none_text_treated_as_empty(self):
        service = EntityExtractionService()
        # Passing empty string (None would be a type error)
        result = await service.extract("")
        assert isinstance(result, ExtractionResult)
        assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_no_entities_in_text(self):
        service = EntityExtractionService.local()
        result = await service.extract("went for a walk today")
        # Should return result with raw_text preserved
        assert result.raw_text == "went for a walk today"

    @pytest.mark.asyncio
    async def test_raw_text_preserved(self):
        service = EntityExtractionService.local()
        text = "Hello Sarah, let's meet tomorrow"
        result = await service.extract(text)
        assert result.raw_text == text


# ── Pipeline integration ─────────────────────────────────────────


class TestPipelineIntegration:
    """Test integration with the capture pipeline's EntityExtractorFunc."""

    @pytest.mark.asyncio
    async def test_extract_entity_refs_returns_entity_refs(self):
        cloud = _make_cloud_extractor(_make_cloud_response_with_entities(
            {"name": "Sarah", "type": "person", "confidence": 0.95},
            {"name": "Notion", "type": "tool", "confidence": 0.9},
        ))
        service = EntityExtractionService(cloud_extractor=cloud)
        refs = await service.extract_entity_refs("Sarah uses Notion")

        assert len(refs) == 2
        assert all(isinstance(r, EntityRef) for r in refs)
        assert refs[0].name == "Sarah"
        assert refs[0].entity_type == "person"
        assert refs[0].confidence == 0.95
        assert refs[1].name == "Notion"
        assert refs[1].entity_type == "tool"

    @pytest.mark.asyncio
    async def test_as_pipeline_extractor_callable(self):
        cloud = _make_cloud_extractor(_make_cloud_response_with_entities(
            {"name": "Jake", "type": "person", "confidence": 0.88},
        ))
        service = EntityExtractionService(cloud_extractor=cloud)
        extractor = service.as_pipeline_extractor()

        # Should be callable with (text: str) -> list[EntityRef]
        refs = await extractor("Jake is here")
        assert len(refs) == 1
        assert refs[0].name == "Jake"
        assert refs[0].entity_type == "person"

    @pytest.mark.asyncio
    async def test_pipeline_extractor_empty_input(self):
        service = EntityExtractionService.local()
        extractor = service.as_pipeline_extractor()

        refs = await extractor("")
        assert refs == []

    @pytest.mark.asyncio
    async def test_entity_to_ref_conversion(self):
        entity = ExtractedEntity(
            name="Google",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.92,
        )
        ref = EntityExtractionService.entity_to_ref(entity)

        assert ref.name == "Google"
        assert ref.entity_type == "organization"
        assert ref.confidence == 0.92

    @pytest.mark.asyncio
    async def test_pipeline_extractor_with_local_fallback(self):
        """Pipeline extractor should work with local-only mode."""
        service = EntityExtractionService.local()
        extractor = service.as_pipeline_extractor()

        refs = await extractor("I work with Sarah at Google HQ")
        names = {r.name for r in refs}
        assert "Sarah" in names


# ── Batch extraction ─────────────────────────────────────────────


class TestBatchExtraction:
    """Test batch extraction of multiple texts."""

    @pytest.mark.asyncio
    async def test_batch_returns_correct_count(self):
        service = EntityExtractionService.local()
        texts = [
            "Sarah is my manager",
            "Meeting at Google HQ",
            "I prefer morning standups",
        ]
        results = await service.extract_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        service = EntityExtractionService.local()
        results = await service.extract_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_updates_stats(self):
        service = EntityExtractionService.local()
        await service.extract_batch(["Hello Sarah", "Hey Jake"])

        assert service.stats.total_extractions == 2


# ── Statistics tracking ──────────────────────────────────────────


class TestStatistics:
    """Test extraction statistics."""

    @pytest.mark.asyncio
    async def test_stats_increment_on_extraction(self):
        service = EntityExtractionService.local()
        await service.extract("Sarah is here")

        assert service.stats.total_extractions == 1
        assert service.stats.successful_extractions == 1
        assert service.stats.failed_extractions == 0

    @pytest.mark.asyncio
    async def test_stats_track_entity_counts(self):
        cloud = _make_cloud_extractor(_make_cloud_response_with_entities(
            {"name": "Sarah", "type": "person", "confidence": 0.9},
            {"name": "Google", "type": "organization", "confidence": 0.9},
        ))
        service = EntityExtractionService(cloud_extractor=cloud)
        await service.extract("Sarah at Google")

        assert service.stats.total_entities_extracted == 2
        assert service.stats.entity_type_counts.get("person") == 1
        assert service.stats.entity_type_counts.get("organization") == 1

    @pytest.mark.asyncio
    async def test_stats_avg_latency(self):
        service = EntityExtractionService.local()
        await service.extract("Hello")
        await service.extract("World")

        assert service.stats.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_stats_success_rate(self):
        service = EntityExtractionService.local()
        await service.extract("Hello Sarah")
        await service.extract("Hey Jake")

        assert service.stats.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_stats_avg_entities_per_extraction(self):
        cloud = _make_cloud_extractor(_make_cloud_response_with_entities(
            {"name": "A", "type": "person", "confidence": 0.9},
            {"name": "B", "type": "person", "confidence": 0.9},
        ))
        service = EntityExtractionService(cloud_extractor=cloud)
        await service.extract("A and B")

        assert service.stats.avg_entities_per_extraction == 2.0

    def test_stats_to_dict(self):
        stats = EntityExtractionStats()
        d = stats.to_dict()
        assert "total_extractions" in d
        assert "success_rate" in d
        assert "avg_latency_ms" in d
        assert "entity_type_counts" in d

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        service = EntityExtractionService.local()
        await service.extract("Hello Sarah")

        assert service.stats.total_extractions == 1
        service.reset_stats()
        assert service.stats.total_extractions == 0

    @pytest.mark.asyncio
    async def test_stats_track_relationships(self):
        cloud = _make_cloud_extractor({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.9},
            ],
            "relationships": [
                {"source": "speaker", "target": "Sarah", "type": "works_with", "confidence": 0.8},
            ],
            "facts": [
                {"content": "Sarah is the team lead", "type": "attribute", "confidence": 0.7},
            ],
            "temporal_references": [
                {"text": "tomorrow", "type": "relative"},
            ],
        })
        service = EntityExtractionService(cloud_extractor=cloud)
        await service.extract("I work with Sarah tomorrow, she's the team lead")

        assert service.stats.total_relationships_extracted == 1
        assert service.stats.total_facts_extracted == 1
        assert service.stats.total_temporal_refs_extracted == 1


# ── Mode detection ───────────────────────────────────────────────


class TestModeDetection:
    """Test mode property."""

    def test_cloud_mode(self):
        client = MagicMock()
        client.is_connected = True
        service = EntityExtractionService.cloud(client)
        assert service.mode == "cloud"

    def test_local_mode(self):
        service = EntityExtractionService.local()
        assert service.mode == "local"

    def test_no_cloud_defaults_to_local(self):
        service = EntityExtractionService(prefer_cloud=True)
        # No cloud extractor → local mode
        assert service.mode == "local"


# ── Comprehensive NLP extraction scenarios ───────────────────────


class TestNLPScenarios:
    """Test realistic natural speech extraction scenarios."""

    @pytest.mark.asyncio
    async def test_casual_conversation_with_entities(self):
        """Natural speech with casually mentioned entities."""
        service = EntityExtractionService.local()
        result = await service.extract(
            "Had a great chat with Sarah about the project, "
            "she's using Notion to track everything"
        )

        names = {e.name for e in result.entities}
        assert "Sarah" in names

    @pytest.mark.asyncio
    async def test_complex_multi_entity_utterance(self):
        """Multiple entities of different types in one utterance."""
        cloud = _make_cloud_extractor({
            "entities": [
                {"name": "Jake", "type": "person", "confidence": 0.95},
                {"name": "Sarah", "type": "person", "confidence": 0.95},
                {"name": "Notion HQ", "type": "place", "confidence": 0.9},
                {"name": "Atlas", "type": "project", "confidence": 0.88},
                {"name": "Slack", "type": "tool", "confidence": 0.92},
            ],
            "relationships": [
                {"source": "Jake", "target": "Sarah", "type": "works_with", "confidence": 0.85},
                {"source": "Jake", "target": "Atlas", "type": "collaborates_on", "confidence": 0.8},
            ],
            "facts": [],
            "temporal_references": [
                {"text": "tomorrow at 3pm", "type": "relative"},
            ],
        })
        service = EntityExtractionService(cloud_extractor=cloud)
        result = await service.extract(
            "Jake and Sarah are meeting at Notion HQ tomorrow at 3pm "
            "to discuss the Atlas project, I'll send the invite on Slack"
        )

        assert result.entity_count == 5
        assert len(result.relationships) == 2
        assert len(result.temporal_references) == 1

    @pytest.mark.asyncio
    async def test_preference_and_habit_extraction(self):
        """Extract behavioral preferences and habits."""
        service = EntityExtractionService.local()
        result = await service.extract(
            "I usually do deep work in the morning and I prefer "
            "to have meetings after lunch"
        )

        fact_types = {f.fact_type for f in result.facts}
        assert FactType.HABIT in fact_types or FactType.PREFERENCE in fact_types

    @pytest.mark.asyncio
    async def test_vague_temporal_handling(self):
        """Handle vague time expressions."""
        service = EntityExtractionService.local()
        result = await service.extract("I'll get to it eventually, maybe later")

        temporal_types = {t.temporal_type for t in result.temporal_references}
        assert TemporalType.VAGUE in temporal_types

    @pytest.mark.asyncio
    async def test_duration_extraction(self):
        """Extract duration temporal references."""
        service = EntityExtractionService.local()
        result = await service.extract("The meeting is for 30 minutes")

        durations = [t for t in result.temporal_references
                     if t.temporal_type == TemporalType.DURATION]
        assert len(durations) >= 1

    @pytest.mark.asyncio
    async def test_works_with_relationship_extraction(self):
        """Extract co-worker relationships."""
        service = EntityExtractionService.local()
        result = await service.extract("I work with Jake on the project")

        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.WORKS_WITH in rel_types

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Entities mentioned multiple times should be deduplicated."""
        service = EntityExtractionService.local()
        result = await service.extract(
            "Sarah said hi to Sarah's team at Sarah's office"
        )

        sarah_count = sum(
            1 for e in result.entities if e.normalized_name == "sarah"
        )
        assert sarah_count <= 1

    @pytest.mark.asyncio
    async def test_association_fact_extraction(self):
        """Extract association facts from text."""
        service = EntityExtractionService.local()
        result = await service.extract("React is related to JavaScript")

        fact_types = {f.fact_type for f in result.facts}
        assert FactType.ASSOCIATION in fact_types

    @pytest.mark.asyncio
    async def test_common_words_not_extracted(self):
        """Common sentence-starting words should not be extracted as entities."""
        service = EntityExtractionService.local()
        result = await service.extract("The project is going well. This is great.")

        names = {e.normalized_name for e in result.entities}
        assert "the" not in names
        assert "this" not in names


# ── ExtractionResult properties ──────────────────────────────────


class TestExtractionResultProperties:
    """Test ExtractionResult helper methods."""

    def test_empty_result(self):
        result = ExtractionResult()
        assert result.entity_count == 0
        assert not result.has_entities
        assert result.entity_types_found == set()
        assert result.entities_by_type(EntityType.PERSON) == []

    def test_entities_by_type(self):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Sarah", entity_type=EntityType.PERSON),
                ExtractedEntity(name="Jake", entity_type=EntityType.PERSON),
                ExtractedEntity(name="Google", entity_type=EntityType.ORGANIZATION),
            ]
        )
        assert len(result.entities_by_type(EntityType.PERSON)) == 2
        assert len(result.entities_by_type(EntityType.ORGANIZATION)) == 1
        assert len(result.entities_by_type(EntityType.TOOL)) == 0


# ── ExtractedEntity dataclass ────────────────────────────────────


class TestExtractedEntity:
    """Test ExtractedEntity auto-normalization and defaults."""

    def test_auto_normalizes_name(self):
        entity = ExtractedEntity(
            name="  Sarah Johnson  ",
            entity_type=EntityType.PERSON,
        )
        assert entity.normalized_name == "sarah johnson"

    def test_explicit_normalized_name_preserved(self):
        entity = ExtractedEntity(
            name="Sarah",
            entity_type=EntityType.PERSON,
            normalized_name="sarah_j",
        )
        assert entity.normalized_name == "sarah_j"

    def test_default_confidence(self):
        entity = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert entity.confidence == 1.0

    def test_default_aliases_empty(self):
        entity = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert entity.aliases == []

    def test_default_attributes_empty(self):
        entity = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert entity.attributes == {}
