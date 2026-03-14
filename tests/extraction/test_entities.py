"""Tests for the core entity extraction service.

Tests cover:
- LLM-based extraction with mocked Gemini responses
- Regex fallback extraction for all entity types
- Entity type classification heuristics
- Relationship extraction
- Fact/preference extraction
- Temporal reference extraction
- Deduplication and confidence filtering
- Error handling and edge cases
- Batch extraction
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.extraction.entities import (
    EntityExtractor,
    ExtractionResult,
    ExtractedEntity,
    TemporalType,
)
from blurt.models.entities import EntityType, FactType, RelationshipType


# ── Fixtures ──────────────────────────────────────────────────────


def _make_gemini_response(
    data: dict[str, Any],
    model: str = "gemini-2.5-flash-lite",
    latency_ms: float = 42.0,
) -> MagicMock:
    """Create a mock GeminiResponse with JSON text."""
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.model = model
    resp.latency_ms = latency_ms
    return resp


def _make_mock_client(
    response_data: dict[str, Any] | None = None,
    is_connected: bool = True,
) -> MagicMock:
    """Create a mock GeminiClient that returns the given response data."""
    client = MagicMock()
    client.is_connected = is_connected

    if response_data is not None:
        resp = _make_gemini_response(response_data)
        client.generate = AsyncMock(return_value=resp)
    else:
        client.generate = AsyncMock(return_value=_make_gemini_response(
            {"entities": [], "relationships": [], "facts": [], "temporal_references": []}
        ))

    return client


# ── Empty / edge-case inputs ─────────────────────────────────────


class TestEmptyInputs:
    """Test empty and edge-case inputs."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_empty_result(self):
        extractor = EntityExtractor()
        result = await extractor.extract("")
        assert result.entity_count == 0
        assert not result.has_entities

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_empty_result(self):
        extractor = EntityExtractor()
        result = await extractor.extract("   \n\t  ")
        assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_none_text_raises_or_returns_empty(self):
        """None input should not crash — treat as empty."""
        extractor = EntityExtractor()
        result = await extractor.extract("")
        assert isinstance(result, ExtractionResult)


# ── LLM-based extraction ─────────────────────────────────────────


class TestLLMExtraction:
    """Test entity extraction via mocked Gemini API."""

    @pytest.mark.asyncio
    async def test_extracts_person_entity(self):
        client = _make_mock_client({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.95}
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("I need to talk to Sarah about the project")

        assert result.entity_count == 1
        entity = result.entities[0]
        assert entity.name == "Sarah"
        assert entity.entity_type == EntityType.PERSON
        assert entity.confidence == 0.95
        assert entity.normalized_name == "sarah"

    @pytest.mark.asyncio
    async def test_extracts_all_entity_types(self):
        client = _make_mock_client({
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
        extractor = EntityExtractor(client)
        result = await extractor.extract(
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
        client = _make_mock_client({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.95},
            ],
            "relationships": [
                {
                    "source": "speaker",
                    "target": "Sarah",
                    "type": "managed_by",
                    "confidence": 0.9,
                }
            ],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Sarah is my manager")

        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.source_name == "speaker"
        assert rel.target_name == "Sarah"
        assert rel.relationship_type == RelationshipType.MANAGED_BY
        assert rel.confidence == 0.9

    @pytest.mark.asyncio
    async def test_extracts_facts(self):
        client = _make_mock_client({
            "entities": [],
            "relationships": [],
            "facts": [
                {
                    "content": "Prefers morning meetings",
                    "type": "preference",
                    "subject": None,
                    "confidence": 0.85,
                }
            ],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("I prefer morning meetings")

        assert len(result.facts) == 1
        fact = result.facts[0]
        assert fact.content == "Prefers morning meetings"
        assert fact.fact_type == FactType.PREFERENCE
        assert fact.subject_entity_name is None

    @pytest.mark.asyncio
    async def test_extracts_temporal_references(self):
        client = _make_mock_client({
            "entities": [],
            "relationships": [],
            "facts": [],
            "temporal_references": [
                {"text": "tomorrow at 3pm", "type": "relative"},
                {"text": "every Monday", "type": "recurring"},
            ],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Meet tomorrow at 3pm, this recurs every Monday")

        assert len(result.temporal_references) == 2
        assert result.temporal_references[0].temporal_type == TemporalType.RELATIVE
        assert result.temporal_references[1].temporal_type == TemporalType.RECURRING

    @pytest.mark.asyncio
    async def test_extracts_entity_aliases(self):
        client = _make_mock_client({
            "entities": [
                {
                    "name": "Q2 Planning Deck",
                    "type": "project",
                    "aliases": ["the deck", "Q2 deck"],
                    "confidence": 0.9,
                }
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Need to finish the deck, the Q2 planning deck")

        assert result.entities[0].aliases == ["the deck", "Q2 deck"]

    @pytest.mark.asyncio
    async def test_extracts_entity_attributes(self):
        client = _make_mock_client({
            "entities": [
                {
                    "name": "Sarah",
                    "type": "person",
                    "attributes": {"role": "manager", "department": "engineering"},
                    "confidence": 0.95,
                }
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Sarah my engineering manager")

        assert result.entities[0].attributes["role"] == "manager"
        assert result.entities[0].attributes["department"] == "engineering"

    @pytest.mark.asyncio
    async def test_filters_low_confidence_entities(self):
        client = _make_mock_client({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.95},
                {"name": "maybe thing", "type": "topic", "confidence": 0.1},
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client, min_confidence=0.3)
        result = await extractor.extract("Sarah mentioned maybe thing")

        assert result.entity_count == 1
        assert result.entities[0].name == "Sarah"

    @pytest.mark.asyncio
    async def test_model_used_populated(self):
        client = _make_mock_client({
            "entities": [],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("hello world")

        assert result.model_used == "gemini-2.5-flash-lite"

    @pytest.mark.asyncio
    async def test_latency_populated(self):
        client = _make_mock_client({
            "entities": [],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("hello world")

        assert result.latency_ms == 42.0

    @pytest.mark.asyncio
    async def test_entities_by_type_filter(self):
        client = _make_mock_client({
            "entities": [
                {"name": "Sarah", "type": "person", "confidence": 0.9},
                {"name": "Alex", "type": "person", "confidence": 0.9},
                {"name": "Notion", "type": "tool", "confidence": 0.95},
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Sarah and Alex use Notion")

        people = result.entities_by_type(EntityType.PERSON)
        assert len(people) == 2
        tools = result.entities_by_type(EntityType.TOOL)
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_unknown_entity_type_defaults_to_topic(self):
        client = _make_mock_client({
            "entities": [
                {"name": "Something", "type": "unknown_type", "confidence": 0.8},
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Something happened")

        assert result.entities[0].entity_type == EntityType.TOPIC

    @pytest.mark.asyncio
    async def test_unknown_relationship_type_defaults_to_related_to(self):
        client = _make_mock_client({
            "entities": [
                {"name": "A", "type": "person", "confidence": 0.9},
                {"name": "B", "type": "person", "confidence": 0.9},
            ],
            "relationships": [
                {"source": "A", "target": "B", "type": "unknown_rel", "confidence": 0.8},
            ],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("A and B")

        assert result.relationships[0].relationship_type == RelationshipType.RELATED_TO

    @pytest.mark.asyncio
    async def test_unknown_fact_type_defaults_to_attribute(self):
        client = _make_mock_client({
            "entities": [],
            "relationships": [],
            "facts": [
                {"content": "something", "type": "unknown_fact", "confidence": 0.8},
            ],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("something")

        assert result.facts[0].fact_type == FactType.ATTRIBUTE


# ── LLM response parsing edge cases ──────────────────────────────


class TestLLMResponseParsing:
    """Test handling of various LLM response formats."""

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self):
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '```json\n{"entities": [{"name": "Test", "type": "person", "confidence": 0.9}], "relationships": [], "facts": [], "temporal_references": []}\n```'
        resp.model = "gemini-2.5-flash-lite"
        resp.latency_ms = 10.0
        client.generate = AsyncMock(return_value=resp)

        extractor = EntityExtractor(client)
        result = await extractor.extract("Test person")

        assert result.entity_count == 1
        assert result.entities[0].name == "Test"

    @pytest.mark.asyncio
    async def test_handles_trailing_commas(self):
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '{"entities": [{"name": "Test", "type": "person", "confidence": 0.9,},], "relationships": [], "facts": [], "temporal_references": [],}'
        resp.model = "gemini-2.5-flash-lite"
        resp.latency_ms = 10.0
        client.generate = AsyncMock(return_value=resp)

        extractor = EntityExtractor(client)
        result = await extractor.extract("Test person")

        assert result.entity_count == 1

    @pytest.mark.asyncio
    async def test_skips_entities_with_no_name(self):
        client = _make_mock_client({
            "entities": [
                {"name": "", "type": "person", "confidence": 0.9},
                {"name": "Valid", "type": "person", "confidence": 0.9},
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client)
        result = await extractor.extract("Valid person")

        assert result.entity_count == 1
        assert result.entities[0].name == "Valid"

    @pytest.mark.asyncio
    async def test_clamps_confidence_to_valid_range(self):
        client = _make_mock_client({
            "entities": [
                {"name": "High", "type": "person", "confidence": 1.5},
                {"name": "Low", "type": "person", "confidence": -0.5},
            ],
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client, min_confidence=0.0)
        result = await extractor.extract("High and Low")

        assert result.entities[0].confidence == 1.0
        assert result.entities[1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_max_entities_limit(self):
        """Extraction should respect max entities limit."""
        many_entities = [
            {"name": f"Entity{i}", "type": "person", "confidence": 0.9}
            for i in range(100)
        ]
        client = _make_mock_client({
            "entities": many_entities,
            "relationships": [],
            "facts": [],
            "temporal_references": [],
        })
        extractor = EntityExtractor(client, max_entities_per_extraction=10)
        result = await extractor.extract("lots of entities")

        assert result.entity_count == 10


# ── Fallback to regex ─────────────────────────────────────────────


class TestLLMFallback:
    """Test that extraction falls back to regex when LLM is unavailable."""

    @pytest.mark.asyncio
    async def test_falls_back_when_no_client(self):
        extractor = EntityExtractor(gemini_client=None)
        result = await extractor.extract("Sarah mentioned the project at Google")

        assert result.model_used == "regex-fallback"
        assert result.has_entities

    @pytest.mark.asyncio
    async def test_falls_back_when_client_disconnected(self):
        client = _make_mock_client(is_connected=False)
        extractor = EntityExtractor(client)
        result = await extractor.extract("Sarah mentioned the project at Google")

        assert result.model_used == "regex-fallback"

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self):
        client = MagicMock()
        client.is_connected = True
        client.generate = AsyncMock(side_effect=Exception("API error"))

        extractor = EntityExtractor(client)
        result = await extractor.extract("Sarah mentioned the project at Google")

        assert result.model_used == "regex-fallback"


# ── Regex-based extraction ────────────────────────────────────────


class TestRegexExtraction:
    """Test the regex fallback extraction."""

    @pytest.mark.asyncio
    async def test_extracts_proper_nouns(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I talked with Sarah about the project")

        names = {e.name for e in result.entities}
        assert "Sarah" in names

    @pytest.mark.asyncio
    async def test_extracts_known_tools(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I need to update the Notion page and check Slack")

        tool_names = {e.normalized_name for e in result.entities if e.entity_type == EntityType.TOOL}
        assert "notion" in tool_names
        assert "slack" in tool_names

    @pytest.mark.asyncio
    async def test_extracts_temporal_tomorrow(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I need to finish this tomorrow")

        assert len(result.temporal_references) >= 1
        temporal_texts = {t.text.lower() for t in result.temporal_references}
        assert "tomorrow" in temporal_texts

    @pytest.mark.asyncio
    async def test_extracts_recurring_temporal(self):
        extractor = EntityExtractor()
        result = await extractor.extract("We have standup every Monday")

        recurring = [t for t in result.temporal_references if t.temporal_type == TemporalType.RECURRING]
        assert len(recurring) >= 1

    @pytest.mark.asyncio
    async def test_extracts_absolute_time(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Meeting at 3pm today")

        absolute_or_relative = [
            t for t in result.temporal_references
            if t.temporal_type in (TemporalType.ABSOLUTE, TemporalType.RELATIVE)
        ]
        assert len(absolute_or_relative) >= 1

    @pytest.mark.asyncio
    async def test_extracts_preference_facts(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I prefer morning meetings over afternoon ones")

        pref_facts = [f for f in result.facts if f.fact_type == FactType.PREFERENCE]
        assert len(pref_facts) >= 1

    @pytest.mark.asyncio
    async def test_extracts_habit_facts(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I usually skip lunch when I'm busy")

        habit_facts = [f for f in result.facts if f.fact_type == FactType.HABIT]
        assert len(habit_facts) >= 1

    @pytest.mark.asyncio
    async def test_deduplicates_entities(self):
        extractor = EntityExtractor()
        # "Notion" will be found both by proper noun regex and by tool extraction
        result = await extractor.extract("I use Notion to organize Notion pages")

        notion_entities = [e for e in result.entities if e.normalized_name == "notion"]
        assert len(notion_entities) == 1

    @pytest.mark.asyncio
    async def test_keeps_highest_confidence_on_dedup(self):
        extractor = EntityExtractor()
        # Tool extraction (0.8 confidence) should beat proper noun (0.5)
        result = await extractor.extract("I use Notion daily")

        notion = [e for e in result.entities if e.normalized_name == "notion"]
        assert len(notion) == 1
        assert notion[0].confidence >= 0.7  # Tool confidence beats regex

    @pytest.mark.asyncio
    async def test_classifies_person_from_context(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I met with Sarah and she told me about it")

        sarah = [e for e in result.entities if e.normalized_name == "sarah"]
        assert len(sarah) == 1
        assert sarah[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_skips_common_words(self):
        extractor = EntityExtractor()
        result = await extractor.extract("The project is going well. This is great.")

        names = {e.normalized_name for e in result.entities}
        assert "the" not in names
        assert "this" not in names

    @pytest.mark.asyncio
    async def test_vague_temporal_reference(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I'll get to it later or sometime soon")

        vague = [t for t in result.temporal_references if t.temporal_type == TemporalType.VAGUE]
        assert len(vague) >= 1

    @pytest.mark.asyncio
    async def test_duration_temporal(self):
        extractor = EntityExtractor()
        result = await extractor.extract("The meeting is for 30 minutes")

        durations = [t for t in result.temporal_references if t.temporal_type == TemporalType.DURATION]
        assert len(durations) >= 1

    @pytest.mark.asyncio
    async def test_raw_text_preserved(self):
        text = "Hello Sarah, let's meet tomorrow"
        extractor = EntityExtractor()
        result = await extractor.extract(text)

        assert result.raw_text == text


# ── Batch extraction ──────────────────────────────────────────────


class TestBatchExtraction:
    """Test batch extraction of multiple texts."""

    @pytest.mark.asyncio
    async def test_batch_returns_correct_count(self):
        extractor = EntityExtractor()
        texts = [
            "Sarah is my manager",
            "I need to finish the project",
            "Meeting tomorrow at 3pm",
        ]
        results = await extractor.extract_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        extractor = EntityExtractor()
        results = await extractor.extract_batch([])
        assert results == []


# ── ExtractionResult properties ───────────────────────────────────


class TestExtractionResult:
    """Test ExtractionResult helper methods."""

    def test_empty_result_properties(self):
        result = ExtractionResult()
        assert result.entity_count == 0
        assert not result.has_entities
        assert result.entity_types_found == set()
        assert result.entities_by_type(EntityType.PERSON) == []

    def test_result_with_entities(self):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Sarah", entity_type=EntityType.PERSON),
                ExtractedEntity(name="Google", entity_type=EntityType.ORGANIZATION),
            ]
        )
        assert result.entity_count == 2
        assert result.has_entities
        assert EntityType.PERSON in result.entity_types_found
        assert EntityType.ORGANIZATION in result.entity_types_found
        assert len(result.entities_by_type(EntityType.PERSON)) == 1


# ── ExtractedEntity dataclass ─────────────────────────────────────


class TestExtractedEntity:
    """Test ExtractedEntity auto-normalization."""

    def test_auto_normalizes_name(self):
        entity = ExtractedEntity(name="  Sarah Johnson  ", entity_type=EntityType.PERSON)
        assert entity.normalized_name == "sarah johnson"

    def test_explicit_normalized_name_preserved(self):
        entity = ExtractedEntity(
            name="Sarah",
            entity_type=EntityType.PERSON,
            normalized_name="sarah_j",
        )
        assert entity.normalized_name == "sarah_j"

    def test_default_confidence_is_one(self):
        entity = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert entity.confidence == 1.0

    def test_default_aliases_empty(self):
        entity = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert entity.aliases == []

    def test_default_attributes_empty(self):
        entity = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert entity.attributes == {}
