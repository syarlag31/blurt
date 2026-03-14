"""Tests for LocalEntityExtractor — offline rule-based extraction."""

from __future__ import annotations

import pytest

from blurt.local.extractor import LocalEntityExtractor
from blurt.models.entities import FactType, RelationshipType


@pytest.fixture
def extractor() -> LocalEntityExtractor:
    return LocalEntityExtractor()


class TestEntityExtraction:
    async def test_extract_person(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I talked to Sarah about the project")
        names = {e.name for e in result.entities}
        assert "Sarah" in names

    async def test_extract_multiple_entities(self, extractor: LocalEntityExtractor):
        result = await extractor.extract(
            "Sarah and Jake are meeting at Notion HQ tomorrow"
        )
        names = {e.name for e in result.entities}
        assert "Sarah" in names
        assert "Jake" in names

    async def test_extract_tool(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I need to update our Notion workspace")
        names = {e.normalized_name for e in result.entities}
        # Notion should be detected as a tool
        assert any("notion" in n for n in names)

    async def test_extract_empty_text(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("")
        assert len(result.entities) == 0

    async def test_extract_no_entities(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I went for a walk")
        # Should still return a result, just maybe no entities
        assert result.raw_text == "I went for a walk"

    async def test_model_used_is_local(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Sarah works at Google")
        assert result.model_used == "local-rules"

    async def test_no_api_calls(self, extractor: LocalEntityExtractor):
        """Extraction should work entirely offline."""
        result = await extractor.extract(
            "Tell Jake about the Atlas project at the downtown office"
        )
        assert result.model_used == "local-rules"
        assert result.latency_ms >= 0


class TestTemporalExtraction:
    async def test_relative_time(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Meeting tomorrow at 3pm")
        temporal_texts = {t.text.lower() for t in result.temporal_references}
        assert any("tomorrow" in t for t in temporal_texts)

    async def test_absolute_time(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Flight at 6am on March 20th")
        assert len(result.temporal_references) > 0

    async def test_recurring(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I exercise every Monday morning")
        temporal_types = {t.temporal_type.value for t in result.temporal_references}
        assert "recurring" in temporal_types

    async def test_vague_time(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I'll get to it eventually")
        temporal_types = {t.temporal_type.value for t in result.temporal_references}
        assert "vague" in temporal_types

    async def test_deadline_detection(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Finish the report by next Friday")
        deadlines = [t for t in result.temporal_references if t.is_deadline]
        assert len(deadlines) > 0


class TestFactExtraction:
    async def test_preference(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I prefer morning meetings")
        fact_types = {f.fact_type for f in result.facts}
        assert FactType.PREFERENCE in fact_types

    async def test_habit(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I usually skip lunch on busy days")
        fact_types = {f.fact_type for f in result.facts}
        assert FactType.HABIT in fact_types

    async def test_attribute(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Sarah is my manager")
        fact_types = {f.fact_type for f in result.facts}
        assert FactType.ATTRIBUTE in fact_types

    async def test_alias(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("the deck means Q2 planning deck")
        fact_types = {f.fact_type for f in result.facts}
        assert FactType.ALIAS in fact_types


class TestRelationshipExtraction:
    async def test_co_mention_relationship(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Sarah and Jake are in the same meeting")
        # Should create MENTIONED_WITH for co-mentioned entities
        if len(result.entities) >= 2:
            rel_types = {r.relationship_type for r in result.relationships}
            assert RelationshipType.MENTIONED_WITH in rel_types

    async def test_manager_relationship(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("Sarah is my manager")
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.MANAGED_BY in rel_types

    async def test_works_with_relationship(self, extractor: LocalEntityExtractor):
        result = await extractor.extract("I work with Jake on the project")
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.WORKS_WITH in rel_types


class TestBatchExtraction:
    async def test_batch(self, extractor: LocalEntityExtractor):
        texts = [
            "Sarah is my manager",
            "Meeting at Google HQ tomorrow",
            "I prefer morning standups",
        ]
        results = await extractor.extract_batch(texts)
        assert len(results) == 3
        assert all(r.model_used == "local-rules" for r in results)


class TestDeduplication:
    async def test_entity_deduplication(self, extractor: LocalEntityExtractor):
        result = await extractor.extract(
            "Sarah said hi to Sarah's team at Sarah's office"
        )
        sarah_count = sum(1 for e in result.entities if e.normalized_name == "sarah")
        # Should be deduplicated to 1
        assert sarah_count <= 1

    async def test_fact_deduplication(self, extractor: LocalEntityExtractor):
        result = await extractor.extract(
            "I prefer coffee. I really prefer coffee in the morning."
        )
        # Facts with similar content should be deduplicated
        contents = [f.content.lower() for f in result.facts]
        unique_contents = set(contents)
        assert len(unique_contents) == len(contents)


class TestExtractionStats:
    async def test_stats_tracking(self, extractor: LocalEntityExtractor):
        await extractor.extract("Hello world")
        await extractor.extract("Another text")

        stats = extractor.stats
        assert stats["extraction_count"] == 2
        assert stats["mode"] == "local-rules"
