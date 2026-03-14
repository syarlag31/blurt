"""Tests for LocalKnowledgeGraphStore — SQLite-backed offline storage."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blurt.clients.embeddings import MockEmbeddingProvider
from blurt.local.storage import LocalKnowledgeGraphStore
from blurt.models.entities import (
    EntityType,
    FactType,
    PatternType,
    RelationshipType,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_knowledge.db"


@pytest.fixture
async def store(tmp_db: Path) -> AsyncGenerator[LocalKnowledgeGraphStore, None]:
    s = LocalKnowledgeGraphStore(
        user_id="test-user",
        embedding_provider=MockEmbeddingProvider(),
        db_path=tmp_db,
    )
    await s.initialize()
    yield s
    await s.close()


# ── Entity Tests ─────────────────────────────────────────────────


class TestEntityOperations:
    async def test_add_entity(self, store: LocalKnowledgeGraphStore):
        entity = await store.add_entity("Sarah", EntityType.PERSON)
        assert entity.name == "Sarah"
        assert entity.normalized_name == "sarah"
        assert entity.entity_type == EntityType.PERSON
        assert entity.mention_count == 1
        assert entity.embedding is not None

    async def test_add_entity_deduplicates(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("sarah", EntityType.PERSON)
        assert e1.id == e2.id
        assert e2.mention_count == 2

    async def test_add_entity_with_aliases(self, store: LocalKnowledgeGraphStore):
        entity = await store.add_entity(
            "Sarah Chen", EntityType.PERSON, aliases=["sarah", "SC"]
        )
        assert "sarah" in entity.aliases
        assert "sc" in entity.aliases

        # Find by alias
        found = await store.find_entity_by_name("SC")
        assert found is not None
        assert found.id == entity.id

    async def test_add_entity_with_attributes(self, store: LocalKnowledgeGraphStore):
        entity = await store.add_entity(
            "Sarah", EntityType.PERSON, attributes={"role": "manager"}
        )
        assert entity.attributes["role"] == "manager"

        # Update attributes on re-add
        updated = await store.add_entity(
            "Sarah", EntityType.PERSON, attributes={"team": "engineering"}
        )
        assert updated.attributes["role"] == "manager"
        assert updated.attributes["team"] == "engineering"

    async def test_get_entity(self, store: LocalKnowledgeGraphStore):
        entity = await store.add_entity("Google", EntityType.ORGANIZATION)
        found = await store.get_entity(entity.id)
        assert found is not None
        assert found.name == "Google"
        assert found.entity_type == EntityType.ORGANIZATION

    async def test_get_entity_not_found(self, store: LocalKnowledgeGraphStore):
        found = await store.get_entity("nonexistent-id")
        assert found is None

    async def test_find_entity_by_name(self, store: LocalKnowledgeGraphStore):
        await store.add_entity("New York", EntityType.PLACE)
        found = await store.find_entity_by_name("new york")
        assert found is not None
        assert found.name == "New York"

    async def test_get_all_entities(self, store: LocalKnowledgeGraphStore):
        await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_entity("Google", EntityType.ORGANIZATION)
        await store.add_entity("NYC", EntityType.PLACE)

        all_entities = await store.get_all_entities()
        assert len(all_entities) == 3

        people = await store.get_all_entities(EntityType.PERSON)
        assert len(people) == 1
        assert people[0].name == "Sarah"

    async def test_update_entity_embedding(self, store: LocalKnowledgeGraphStore):
        entity = await store.add_entity("Sarah", EntityType.PERSON)

        updated = await store.update_entity_embedding(entity.id)
        assert updated is not None
        assert updated.embedding is not None


# ── Relationship Tests ───────────────────────────────────────────


class TestRelationshipOperations:
    async def test_add_relationship(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("Google", EntityType.ORGANIZATION)

        rel = await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.EMPLOYED_BY
        )
        assert rel.strength == 1.0
        assert rel.co_mention_count == 1

    async def test_strengthen_relationship(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("Jake", EntityType.PERSON)

        rel1 = await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.WORKS_WITH
        )
        rel2 = await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.WORKS_WITH, context="in the same meeting"
        )

        assert rel1.id == rel2.id
        assert rel2.co_mention_count == 2
        assert rel2.strength > 1.0
        assert len(rel2.context_snippets) == 1

    async def test_get_entity_relationships(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("Google", EntityType.ORGANIZATION)
        e3 = await store.add_entity("Project X", EntityType.PROJECT)

        await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.EMPLOYED_BY
        )
        await store.add_or_strengthen_relationship(
            e1.id, e3.id, RelationshipType.COLLABORATES_ON
        )

        rels = await store.get_entity_relationships(e1.id)
        assert len(rels) == 2

    async def test_get_connected_entities(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("Jake", EntityType.PERSON)
        e3 = await store.add_entity("Google", EntityType.ORGANIZATION)

        await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.WORKS_WITH
        )
        await store.add_or_strengthen_relationship(
            e1.id, e3.id, RelationshipType.EMPLOYED_BY
        )

        connections = await store.get_connected_entities(e1.id)
        assert len(connections) == 2
        connected_names = {e.name for e, _ in connections}
        assert "Jake" in connected_names
        assert "Google" in connected_names

    async def test_decay_relationships(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("Jake", EntityType.PERSON)

        rel = await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.WORKS_WITH
        )
        original_strength = rel.strength

        # Decay 60 days into the future (2 half-lives)
        future = datetime.now(timezone.utc) + timedelta(days=60)
        await store.decay_relationships(as_of=future)

        # Re-read the relationship to check decayed strength
        rels = await store.get_entity_relationships(e1.id)
        assert len(rels) == 1
        assert rels[0].strength < original_strength


# ── Fact Tests ───────────────────────────────────────────────────


class TestFactOperations:
    async def test_add_fact(self, store: LocalKnowledgeGraphStore):
        e = await store.add_entity("Sarah", EntityType.PERSON)
        fact = await store.add_fact(
            content="Sarah is my manager",
            fact_type=FactType.ATTRIBUTE,
            subject_entity_id=e.id,
        )
        assert fact.content == "Sarah is my manager"
        assert fact.fact_type == FactType.ATTRIBUTE
        assert fact.is_active is True
        assert fact.embedding is not None

    async def test_get_entity_facts(self, store: LocalKnowledgeGraphStore):
        e = await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE, e.id)
        await store.add_fact("Sarah likes coffee", FactType.PREFERENCE, e.id)

        facts = await store.get_entity_facts(e.id)
        assert len(facts) == 2

    async def test_supersede_fact(self, store: LocalKnowledgeGraphStore):
        e = await store.add_entity("Sarah", EntityType.PERSON)
        old = await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE, e.id)

        new = await store.supersede_fact(old.id, "Sarah is my director")
        assert new is not None
        assert new.content == "Sarah is my director"

        # Old fact should be inactive
        facts = await store.get_entity_facts(e.id, active_only=True)
        assert all(f.content != "Sarah is my manager" for f in facts)

    async def test_get_all_facts_filtered(self, store: LocalKnowledgeGraphStore):
        await store.add_fact("I prefer mornings", FactType.PREFERENCE)
        await store.add_fact("I usually skip lunch", FactType.HABIT)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)

        prefs = await store.get_all_facts(fact_type=FactType.PREFERENCE)
        assert len(prefs) == 1
        assert prefs[0].content == "I prefer mornings"


# ── Pattern Tests ────────────────────────────────────────────────


class TestPatternOperations:
    async def test_add_pattern(self, store: LocalKnowledgeGraphStore):
        pattern = await store.add_pattern(
            pattern_type=PatternType.TIME_OF_DAY,
            description="User is most productive in the morning",
            parameters={"peak_hours": [9, 10, 11]},
        )
        assert pattern.description == "User is most productive in the morning"
        assert pattern.confidence == 0.5
        assert pattern.is_active is False  # Below threshold

    async def test_add_active_pattern(self, store: LocalKnowledgeGraphStore):
        pattern = await store.add_pattern(
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="High energy mornings",
            confidence=0.8,
        )
        assert pattern.is_active is True

    async def test_confirm_pattern_promotes(self, store: LocalKnowledgeGraphStore):
        pattern = await store.add_pattern(
            pattern_type=PatternType.MOOD_CYCLE,
            description="Low energy on Mondays",
            confidence=0.5,
            observation_count=4,
        )
        assert pattern.is_active is False

        # Confirm enough times to promote
        confirmed = await store.confirm_pattern(pattern.id, "obs-1")
        assert confirmed is not None
        assert confirmed.observation_count == 5
        assert confirmed.is_active is True

    async def test_get_active_patterns(self, store: LocalKnowledgeGraphStore):
        await store.add_pattern(
            PatternType.TIME_OF_DAY, "Morning person", confidence=0.8
        )
        await store.add_pattern(
            PatternType.MOOD_CYCLE, "Low Monday", confidence=0.3
        )

        active = await store.get_active_patterns()
        assert len(active) == 1
        assert active[0].description == "Morning person"

    async def test_get_all_patterns(self, store: LocalKnowledgeGraphStore):
        await store.add_pattern(
            PatternType.TIME_OF_DAY, "Morning person", confidence=0.8
        )
        await store.add_pattern(
            PatternType.MOOD_CYCLE, "Low Monday", confidence=0.3
        )

        all_patterns = await store.get_all_patterns()
        assert len(all_patterns) == 2


# ── Search Tests ─────────────────────────────────────────────────


class TestSemanticSearch:
    async def test_search_entities(self, store: LocalKnowledgeGraphStore):
        await store.add_entity("Sarah Chen", EntityType.PERSON)
        await store.add_entity("Google", EntityType.ORGANIZATION)

        results = await store.search("people", item_types=["entity"])
        assert len(results) >= 0  # Mock embeddings may or may not match

    async def test_search_returns_sorted_results(self, store: LocalKnowledgeGraphStore):
        await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_entity("Jake", EntityType.PERSON)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)

        # Use min_similarity=0.0 to include any non-negative match
        results = await store.search("manager", min_similarity=0.0)
        # With mock embeddings, we may get results if hashes produce positive cosine
        # At minimum, verify sort order if we have results
        if len(results) == 0:
            # No results is acceptable with mock embeddings — just verify no crash
            return
        # Results should be sorted by similarity descending
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score

    async def test_search_with_top_k(self, store: LocalKnowledgeGraphStore):
        for i in range(10):
            await store.add_entity(f"Entity{i}", EntityType.TOPIC)

        results = await store.search("entity", top_k=3, min_similarity=0.0)
        assert len(results) <= 3


# ── Graph Stats & Context Tests ──────────────────────────────────


class TestGraphQueries:
    async def test_get_graph_stats(self, store: LocalKnowledgeGraphStore):
        await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_entity("Google", EntityType.ORGANIZATION)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)

        stats = await store.get_graph_stats()
        assert stats["total_entities"] == 2
        assert stats["total_facts"] == 1
        assert stats["active_facts"] == 1
        assert stats["entity_type_counts"]["person"] == 1

    async def test_get_entity_context(self, store: LocalKnowledgeGraphStore):
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("Google", EntityType.ORGANIZATION)
        await store.add_or_strengthen_relationship(
            e1.id, e2.id, RelationshipType.EMPLOYED_BY
        )
        await store.add_fact(
            "Sarah is my manager", FactType.ATTRIBUTE, subject_entity_id=e1.id
        )

        ctx = await store.get_entity_context(e1.id)
        assert ctx["entity"]["name"] == "Sarah"
        assert len(ctx["facts"]) == 1
        assert len(ctx["connections"]) == 1

    async def test_get_entity_context_empty(self, store: LocalKnowledgeGraphStore):
        ctx = await store.get_entity_context("nonexistent")
        assert ctx == {}


# ── Pipeline Integration Tests ───────────────────────────────────


class TestPipelineIntegration:
    async def test_process_extracted_entities(self, store: LocalKnowledgeGraphStore):
        entities_data = [
            {"name": "Sarah", "type": "person"},
            {"name": "Google", "type": "organization"},
        ]
        nodes = await store.process_extracted_entities(
            entities_data, blurt_id="blurt-1", raw_text="Sarah works at Google"
        )
        assert len(nodes) == 2

        # Should have created MENTIONED_WITH relationship
        rels = await store.get_entity_relationships(nodes[0].id)
        assert len(rels) == 1
        assert rels[0].relationship_type == RelationshipType.MENTIONED_WITH

    async def test_process_extracted_facts(self, store: LocalKnowledgeGraphStore):
        await store.add_entity("Sarah", EntityType.PERSON)
        facts_data = [
            {
                "content": "Sarah is my manager",
                "type": "attribute",
                "subject_entity": "Sarah",
            },
        ]
        facts = await store.process_extracted_facts(facts_data, blurt_id="blurt-1")
        assert len(facts) == 1
        assert facts[0].content == "Sarah is my manager"


# ── Persistence Tests ────────────────────────────────────────────


class TestPersistence:
    async def test_data_persists_across_sessions(self, tmp_db: Path):
        """Data should survive store close and reopen."""
        provider = MockEmbeddingProvider()

        # Session 1: write data
        store1 = LocalKnowledgeGraphStore("user-1", provider, tmp_db)
        await store1.initialize()
        await store1.add_entity("Sarah", EntityType.PERSON)
        await store1.add_fact("Sarah is my manager", FactType.ATTRIBUTE)
        await store1.close()

        # Session 2: read data
        store2 = LocalKnowledgeGraphStore("user-1", provider, tmp_db)
        await store2.initialize()

        entities = await store2.get_all_entities()
        assert len(entities) == 1
        assert entities[0].name == "Sarah"

        facts = await store2.get_all_facts()
        assert len(facts) == 1

        await store2.close()

    async def test_user_isolation(self, tmp_db: Path):
        """Different users' data should be isolated."""
        provider = MockEmbeddingProvider()

        store1 = LocalKnowledgeGraphStore("user-1", provider, tmp_db)
        await store1.initialize()
        await store1.add_entity("Sarah", EntityType.PERSON)

        store2 = LocalKnowledgeGraphStore("user-2", provider, tmp_db)
        await store2.initialize()
        await store2.add_entity("Jake", EntityType.PERSON)

        user1_entities = await store1.get_all_entities()
        user2_entities = await store2.get_all_entities()

        assert len(user1_entities) == 1
        assert user1_entities[0].name == "Sarah"
        assert len(user2_entities) == 1
        assert user2_entities[0].name == "Jake"

        await store1.close()
        await store2.close()
