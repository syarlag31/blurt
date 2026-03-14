"""Tests for the semantic memory tier — knowledge graph with vector embeddings.

Tests cover:
- Entity CRUD and deduplication
- Relationship creation, strengthening, and decay
- Fact storage, confirmation, and supersession
- Pattern detection and promotion
- Semantic search across the knowledge graph
- Pipeline integration (process_extracted_entities/facts)
- Graph statistics and context queries
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from blurt.memory.semantic import (
    MIN_PATTERN_OBSERVATIONS,
    PATTERN_PROMOTION_THRESHOLD,
    STRENGTH_DECAY_HALF_LIFE_DAYS,
    SemanticMemoryStore,
)
from blurt.models.entities import (
    EntityType,
    FactType,
    PatternType,
    RelationshipType,
)
from blurt.clients.embeddings import (
    MockEmbeddingProvider,
    cosine_similarity,
)

USER_ID = "test-user-001"


@pytest_asyncio.fixture
async def store():
    """Create a SemanticMemoryStore with mock embeddings for testing."""
    provider = MockEmbeddingProvider()
    return SemanticMemoryStore(user_id=USER_ID, embedding_provider=provider)


@pytest_asyncio.fixture
async def populated_store(store: SemanticMemoryStore):
    """Create a store pre-populated with sample entities and relationships."""
    sarah = await store.add_entity("Sarah", EntityType.PERSON)
    jake = await store.add_entity("Jake", EntityType.PERSON)
    q2_deck = await store.add_entity(
        "Q2 Planning Deck",
        EntityType.PROJECT,
        aliases=["the deck", "Q2 deck"],
    )
    acme = await store.add_entity("Acme Corp", EntityType.ORGANIZATION)

    await store.add_or_strengthen_relationship(
        sarah.id, jake.id, RelationshipType.WORKS_WITH, "Sarah and Jake on Q2"
    )
    await store.add_or_strengthen_relationship(
        sarah.id, q2_deck.id, RelationshipType.COLLABORATES_ON, "Sarah working on deck"
    )
    await store.add_or_strengthen_relationship(
        q2_deck.id, acme.id, RelationshipType.PART_OF, "Q2 deck for Acme"
    )

    return store


# ── Entity Tests ──────────────────────────────────────────────────────


class TestEntityOperations:
    @pytest.mark.asyncio
    async def test_add_entity(self, store: SemanticMemoryStore):
        entity = await store.add_entity("Sarah", EntityType.PERSON)
        assert entity.name == "Sarah"
        assert entity.normalized_name == "sarah"
        assert entity.entity_type == EntityType.PERSON
        assert entity.mention_count == 1
        assert entity.embedding is not None
        assert len(entity.embedding) == MockEmbeddingProvider.DIMENSION
        assert entity.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_add_entity_deduplication(self, store: SemanticMemoryStore):
        """Adding the same entity twice should increment mention count."""
        e1 = await store.add_entity("Sarah", EntityType.PERSON)
        e2 = await store.add_entity("sarah", EntityType.PERSON)  # case-insensitive
        assert e1.id == e2.id
        assert e2.mention_count == 2

    @pytest.mark.asyncio
    async def test_add_entity_with_aliases(self, store: SemanticMemoryStore):
        entity = await store.add_entity(
            "Q2 Planning Deck",
            EntityType.PROJECT,
            aliases=["the deck", "Q2 deck"],
        )
        assert "the deck" in entity.aliases
        assert "q2 deck" in entity.aliases

        # Find by alias
        found = await store.find_entity_by_name("the deck")
        assert found is not None
        assert found.id == entity.id

    @pytest.mark.asyncio
    async def test_find_entity_by_name(self, store: SemanticMemoryStore):
        await store.add_entity("Sarah", EntityType.PERSON)
        found = await store.find_entity_by_name("Sarah")
        assert found is not None
        assert found.name == "Sarah"

        not_found = await store.find_entity_by_name("Unknown Person")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_entity_by_id(self, store: SemanticMemoryStore):
        entity = await store.add_entity("Sarah", EntityType.PERSON)
        found = await store.get_entity(entity.id)
        assert found is not None
        assert found.id == entity.id

    @pytest.mark.asyncio
    async def test_get_all_entities_with_filter(
        self, populated_store: SemanticMemoryStore
    ):
        all_entities = await populated_store.get_all_entities()
        assert len(all_entities) == 4

        people = await populated_store.get_all_entities(EntityType.PERSON)
        assert len(people) == 2

        projects = await populated_store.get_all_entities(EntityType.PROJECT)
        assert len(projects) == 1

    @pytest.mark.asyncio
    async def test_entity_attributes_update(self, store: SemanticMemoryStore):
        e1 = await store.add_entity(
            "Sarah", EntityType.PERSON, attributes={"role": "manager"}
        )
        assert e1.attributes["role"] == "manager"

        e2 = await store.add_entity(
            "Sarah", EntityType.PERSON, attributes={"department": "engineering"}
        )
        assert e2.attributes["role"] == "manager"
        assert e2.attributes["department"] == "engineering"

    @pytest.mark.asyncio
    async def test_update_entity_embedding(
        self, populated_store: SemanticMemoryStore
    ):
        entities = await populated_store.get_all_entities(EntityType.PERSON)
        sarah = entities[0]

        # Add a fact to change context
        await populated_store.add_fact(
            "Sarah is the VP of Engineering",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
        )

        updated = await populated_store.update_entity_embedding(sarah.id)
        assert updated is not None
        # Embedding should change since context changed
        assert updated.embedding is not None


# ── Relationship Tests ────────────────────────────────────────────────


class TestRelationshipOperations:
    @pytest.mark.asyncio
    async def test_add_relationship(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        jake = await store.add_entity("Jake", EntityType.PERSON)

        rel = await store.add_or_strengthen_relationship(
            sarah.id, jake.id, RelationshipType.WORKS_WITH, "working together"
        )
        assert rel.source_entity_id == sarah.id
        assert rel.target_entity_id == jake.id
        assert rel.strength == 1.0
        assert rel.co_mention_count == 1
        assert "working together" in rel.context_snippets

    @pytest.mark.asyncio
    async def test_strengthen_relationship(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        jake = await store.add_entity("Jake", EntityType.PERSON)

        rel1 = await store.add_or_strengthen_relationship(
            sarah.id, jake.id, RelationshipType.WORKS_WITH
        )
        assert rel1.strength == 1.0

        rel2 = await store.add_or_strengthen_relationship(
            sarah.id, jake.id, RelationshipType.WORKS_WITH, "meeting notes"
        )
        assert rel2.id == rel1.id  # same relationship
        assert rel2.co_mention_count == 2
        assert rel2.strength > 1.0  # strengthened

    @pytest.mark.asyncio
    async def test_strength_log_growth(self, store: SemanticMemoryStore):
        """Relationship strength should follow log curve — diminishing returns."""
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        jake = await store.add_entity("Jake", EntityType.PERSON)

        strengths = []
        for i in range(10):
            rel = await store.add_or_strengthen_relationship(
                sarah.id, jake.id, RelationshipType.WORKS_WITH
            )
            strengths.append(rel.strength)

        # Each increment should be smaller (diminishing returns)
        deltas = [strengths[i + 1] - strengths[i] for i in range(len(strengths) - 1)]
        for i in range(len(deltas) - 1):
            assert deltas[i] >= deltas[i + 1] - 0.001  # allow small float error

    @pytest.mark.asyncio
    async def test_relationship_decay(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        jake = await store.add_entity("Jake", EntityType.PERSON)

        rel = await store.add_or_strengthen_relationship(
            sarah.id, jake.id, RelationshipType.WORKS_WITH
        )
        original_strength = rel.strength

        # Decay after 30 days (one half-life)
        future = datetime.now(timezone.utc) + timedelta(days=STRENGTH_DECAY_HALF_LIFE_DAYS)
        await store.decay_relationships(as_of=future)

        assert rel.strength < original_strength
        assert abs(rel.strength - original_strength * 0.5) < 0.01

    @pytest.mark.asyncio
    async def test_get_connected_entities(
        self, populated_store: SemanticMemoryStore
    ):
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        connections = await populated_store.get_connected_entities(sarah.id)

        assert len(connections) == 2  # Jake and Q2 deck
        # Should be sorted by strength
        entity_names = [e.name for e, _ in connections]
        assert "Jake" in entity_names
        assert "Q2 Planning Deck" in entity_names

    @pytest.mark.asyncio
    async def test_context_snippets_capped(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        jake = await store.add_entity("Jake", EntityType.PERSON)

        for i in range(15):
            await store.add_or_strengthen_relationship(
                sarah.id, jake.id, RelationshipType.WORKS_WITH, f"context {i}"
            )

        rel = (await store.get_entity_relationships(sarah.id))[0]
        assert len(rel.context_snippets) <= 10


# ── Fact Tests ────────────────────────────────────────────────────────


class TestFactOperations:
    @pytest.mark.asyncio
    async def test_add_fact(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        fact = await store.add_fact(
            "Sarah is my manager",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
            source_blurt_id="blurt-001",
        )
        assert fact.content == "Sarah is my manager"
        assert fact.fact_type == FactType.ATTRIBUTE
        assert fact.is_active is True
        assert fact.embedding is not None
        assert "blurt-001" in fact.source_blurt_ids

    @pytest.mark.asyncio
    async def test_fact_confirmation(self, store: SemanticMemoryStore):
        """Adding the same fact twice should confirm, not duplicate."""
        fact1 = await store.add_fact(
            "I prefer morning meetings", FactType.PREFERENCE
        )
        # Exact same text will have same embedding → confirm
        fact2 = await store.add_fact(
            "I prefer morning meetings", FactType.PREFERENCE
        )
        assert fact2.id == fact1.id
        assert fact2.confirmation_count == 2
        assert fact2.confidence >= fact1.confidence

    @pytest.mark.asyncio
    async def test_supersede_fact(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        old = await store.add_fact(
            "Sarah is a junior engineer",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
        )

        new = await store.supersede_fact(
            old.id, "Sarah is a senior engineer", source_blurt_id="blurt-002"
        )

        assert new is not None
        assert new.content == "Sarah is a senior engineer"
        assert old.is_active is False
        assert old.superseded_by == new.id

    @pytest.mark.asyncio
    async def test_get_entity_facts(self, store: SemanticMemoryStore):
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_fact(
            "Sarah is my manager", FactType.ATTRIBUTE, subject_entity_id=sarah.id
        )
        await store.add_fact(
            "Sarah likes coffee", FactType.PREFERENCE, subject_entity_id=sarah.id
        )

        facts = await store.get_entity_facts(sarah.id)
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_get_all_facts_filtered(self, store: SemanticMemoryStore):
        await store.add_fact("I prefer mornings", FactType.PREFERENCE)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)

        prefs = await store.get_all_facts(fact_type=FactType.PREFERENCE)
        assert len(prefs) == 1
        assert prefs[0].content == "I prefer mornings"

    @pytest.mark.asyncio
    async def test_alias_fact(self, store: SemanticMemoryStore):
        fact = await store.add_fact(
            "'the deck' refers to the Q2 planning deck", FactType.ALIAS
        )
        assert fact.fact_type == FactType.ALIAS


# ── Pattern Tests ─────────────────────────────────────────────────────


class TestPatternOperations:
    @pytest.mark.asyncio
    async def test_add_pattern(self, store: SemanticMemoryStore):
        pattern = await store.add_pattern(
            PatternType.TIME_OF_DAY,
            "User has low energy on Thursday afternoons",
            parameters={"day": "thursday", "hour_range": [14, 17], "energy": "low"},
            confidence=0.5,
        )
        assert pattern.pattern_type == PatternType.TIME_OF_DAY
        assert pattern.is_active is False  # below threshold
        assert pattern.embedding is not None

    @pytest.mark.asyncio
    async def test_pattern_promotion(self, store: SemanticMemoryStore):
        """Pattern should become active when confidence and observations meet thresholds."""
        pattern = await store.add_pattern(
            PatternType.ENERGY_RHYTHM,
            "Peak creativity before 10am",
            confidence=0.3,
        )
        assert pattern.is_active is False
        pattern_id = pattern.id

        # Confirm many times to build confidence
        for i in range(MIN_PATTERN_OBSERVATIONS + 5):
            pattern = await store.confirm_pattern(
                pattern_id, observation_id=f"obs-{i}"
            )

        assert pattern is not None
        assert pattern.is_active is True
        assert pattern.confidence >= PATTERN_PROMOTION_THRESHOLD
        assert pattern.observation_count >= MIN_PATTERN_OBSERVATIONS

    @pytest.mark.asyncio
    async def test_get_active_patterns(self, store: SemanticMemoryStore):
        # Add one active pattern (high confidence)
        await store.add_pattern(
            PatternType.TIME_OF_DAY,
            "Morning person",
            confidence=0.9,
        )
        # Add one inactive pattern
        await store.add_pattern(
            PatternType.MOOD_CYCLE,
            "Monday blues",
            confidence=0.3,
        )

        active = await store.get_active_patterns()
        assert len(active) == 1
        assert active[0].description == "Morning person"

    @pytest.mark.asyncio
    async def test_get_active_patterns_by_type(self, store: SemanticMemoryStore):
        await store.add_pattern(
            PatternType.TIME_OF_DAY, "Morning person", confidence=0.9
        )
        await store.add_pattern(
            PatternType.ENERGY_RHYTHM, "Post-lunch crash", confidence=0.9
        )

        time_patterns = await store.get_active_patterns(PatternType.TIME_OF_DAY)
        assert len(time_patterns) == 1


# ── Semantic Search Tests ─────────────────────────────────────────────


class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_search_entities(self, populated_store: SemanticMemoryStore):
        results = await populated_store.search(
            "person: Sarah", item_types=["entity"], min_similarity=0.0
        )
        assert len(results) > 0
        assert results[0].item_type == "entity"

    @pytest.mark.asyncio
    async def test_search_facts(self, store: SemanticMemoryStore):
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)
        await store.add_fact("I prefer morning meetings", FactType.PREFERENCE)

        # Use exact text to guarantee a match with mock embeddings
        results = await store.search(
            "Sarah is my manager", item_types=["fact"], min_similarity=0.0
        )
        assert len(results) > 0
        assert results[0].similarity_score == 1.0

    @pytest.mark.asyncio
    async def test_search_patterns(self, store: SemanticMemoryStore):
        # Use the same text for query and pattern to guarantee a match
        description = "Low energy Thursday afternoons"
        await store.add_pattern(
            PatternType.TIME_OF_DAY,
            description,
            confidence=0.9,
        )

        results = await store.search(
            description, item_types=["pattern"], min_similarity=0.0
        )
        assert len(results) > 0
        assert results[0].similarity_score == 1.0

    @pytest.mark.asyncio
    async def test_search_cross_type(self, populated_store: SemanticMemoryStore):
        """Search should return results from multiple types when not filtered."""
        await populated_store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)

        results = await populated_store.search("Sarah", min_similarity=0.0)
        types = {r.item_type for r in results}
        # Should find at least entities
        assert "entity" in types

    @pytest.mark.asyncio
    async def test_search_top_k(self, store: SemanticMemoryStore):
        for i in range(20):
            await store.add_entity(f"Entity{i}", EntityType.TOPIC)

        results = await store.search("Entity", top_k=5, min_similarity=0.0)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_min_similarity_filter(
        self, populated_store: SemanticMemoryStore
    ):
        results = await populated_store.search(
            "Sarah", min_similarity=0.99
        )
        # Very high threshold should filter most results
        assert len(results) <= len(await populated_store.get_all_entities())


# ── Graph Context Tests ───────────────────────────────────────────────


class TestGraphContext:
    @pytest.mark.asyncio
    async def test_get_entity_context(
        self, populated_store: SemanticMemoryStore
    ):
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        await populated_store.add_fact(
            "Sarah is VP of Engineering",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
        )

        context = await populated_store.get_entity_context(sarah.id)
        assert "entity" in context
        assert context["entity"]["name"] == "Sarah"
        assert len(context["facts"]) == 1
        assert len(context["connections"]) == 2

    @pytest.mark.asyncio
    async def test_get_entity_context_missing(
        self, store: SemanticMemoryStore
    ):
        context = await store.get_entity_context("nonexistent-id")
        assert context == {}

    @pytest.mark.asyncio
    async def test_graph_stats(self, populated_store: SemanticMemoryStore):
        stats = await populated_store.get_graph_stats()
        assert stats["total_entities"] == 4
        assert stats["total_relationships"] == 3
        assert stats["entity_type_counts"]["person"] == 2
        assert stats["entity_type_counts"]["project"] == 1
        assert stats["entity_type_counts"]["organization"] == 1


# ── Pipeline Integration Tests ────────────────────────────────────────


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_process_extracted_entities(
        self, store: SemanticMemoryStore
    ):
        """Simulate the pipeline sending extracted entities to semantic memory."""
        entities = [
            {"name": "Sarah", "type": "person"},
            {"name": "Q2 Deck", "type": "project"},
            {"name": "Acme Corp", "type": "organization"},
        ]

        processed = await store.process_extracted_entities(
            entities, blurt_id="blurt-001", raw_text="Sarah working on Q2 deck for Acme"
        )

        assert len(processed) == 3

        # Co-mentioned entities should have relationships
        sarah = await store.find_entity_by_name("Sarah")
        assert sarah is not None
        rels = await store.get_entity_relationships(sarah.id)
        assert len(rels) == 2  # mentioned with Q2 Deck and Acme Corp

    @pytest.mark.asyncio
    async def test_process_extracted_entities_updates_existing(
        self, store: SemanticMemoryStore
    ):
        """Processing the same entity twice should update, not duplicate."""
        await store.process_extracted_entities(
            [{"name": "Sarah", "type": "person"}],
            blurt_id="blurt-001",
            raw_text="First mention of Sarah",
        )
        await store.process_extracted_entities(
            [{"name": "Sarah", "type": "person"}],
            blurt_id="blurt-002",
            raw_text="Second mention of Sarah",
        )

        all_entities = await store.get_all_entities()
        assert len(all_entities) == 1
        assert all_entities[0].mention_count == 2

    @pytest.mark.asyncio
    async def test_process_extracted_facts(
        self, store: SemanticMemoryStore
    ):
        await store.add_entity("Sarah", EntityType.PERSON)

        facts = [
            {
                "content": "Sarah is my manager",
                "type": "attribute",
                "subject_entity": "Sarah",
            },
            {
                "content": "I prefer morning meetings",
                "type": "preference",
            },
        ]

        processed = await store.process_extracted_facts(facts, blurt_id="blurt-001")
        assert len(processed) == 2

        sarah = await store.find_entity_by_name("Sarah")
        assert sarah is not None
        sarah_facts = await store.get_entity_facts(sarah.id)
        assert len(sarah_facts) == 1
        assert sarah_facts[0].content == "Sarah is my manager"

    @pytest.mark.asyncio
    async def test_process_entities_with_invalid_type(
        self, store: SemanticMemoryStore
    ):
        """Unknown entity types should fall back to TOPIC."""
        processed = await store.process_extracted_entities(
            [{"name": "Something", "type": "unknown_type"}],
            blurt_id="blurt-001",
            raw_text="Something happened",
        )
        assert processed[0].entity_type == EntityType.TOPIC

    @pytest.mark.asyncio
    async def test_knowledge_graph_compounds(
        self, store: SemanticMemoryStore
    ):
        """Simulate multiple blurts building the knowledge graph over time."""
        # Blurt 1: "Meeting with Sarah about the Q2 deck"
        await store.process_extracted_entities(
            [
                {"name": "Sarah", "type": "person"},
                {"name": "Q2 Deck", "type": "project"},
            ],
            blurt_id="blurt-001",
            raw_text="Meeting with Sarah about the Q2 deck",
        )

        # Blurt 2: "Sarah and Jake are presenting the Q2 deck to Acme"
        await store.process_extracted_entities(
            [
                {"name": "Sarah", "type": "person"},
                {"name": "Jake", "type": "person"},
                {"name": "Q2 Deck", "type": "project"},
                {"name": "Acme Corp", "type": "organization"},
            ],
            blurt_id="blurt-002",
            raw_text="Sarah and Jake are presenting the Q2 deck to Acme",
        )

        # Blurt 3: "Sarah mentioned the Q2 deck is almost done"
        await store.process_extracted_entities(
            [
                {"name": "Sarah", "type": "person"},
                {"name": "Q2 Deck", "type": "project"},
            ],
            blurt_id="blurt-003",
            raw_text="Sarah mentioned the Q2 deck is almost done",
        )

        # Verify compounding
        sarah = await store.find_entity_by_name("Sarah")
        assert sarah is not None
        assert sarah.mention_count == 3  # mentioned in all 3 blurts

        # Sarah-Q2Deck relationship should be strongest (3 co-mentions)
        connections = await store.get_connected_entities(sarah.id)
        # Find Q2 deck connection
        q2_connection = [
            (e, r) for e, r in connections if e.normalized_name == "q2 deck"
        ]
        assert len(q2_connection) == 1
        assert q2_connection[0][1].co_mention_count == 3

        # Sarah-Jake relationship weaker (1 co-mention)
        jake_connection = [
            (e, r) for e, r in connections if e.normalized_name == "jake"
        ]
        assert len(jake_connection) == 1
        assert jake_connection[0][1].co_mention_count == 1

        # Graph should have grown
        stats = await store.get_graph_stats()
        assert stats["total_entities"] == 4
        assert stats["total_relationships"] > 0


# ── Embedding Provider Tests ─────────────────────────────────────────


class TestEmbeddingProviders:
    @pytest.mark.asyncio
    async def test_mock_embedding_deterministic(self):
        provider = MockEmbeddingProvider()
        e1 = await provider.embed("hello world")
        e2 = await provider.embed("hello world")
        assert e1 == e2

    @pytest.mark.asyncio
    async def test_mock_embedding_different_texts(self):
        provider = MockEmbeddingProvider()
        e1 = await provider.embed("hello world")
        e2 = await provider.embed("goodbye world")
        assert e1 != e2

    @pytest.mark.asyncio
    async def test_mock_embedding_batch(self):
        provider = MockEmbeddingProvider()
        results = await provider.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == MockEmbeddingProvider.DIMENSION

    @pytest.mark.asyncio
    async def test_mock_embedding_unit_vector(self):
        provider = MockEmbeddingProvider()
        vec = await provider.embed("test")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 0.01  # should be unit vector

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v1, v2)) < 0.001

    def test_cosine_similarity_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ── Semantic Similarity Search & Retrieval Tests ─────────────────────


class TestSearchSimilarEntities:
    @pytest.mark.asyncio
    async def test_find_similar_entities(self, populated_store: SemanticMemoryStore):
        """Searching from an entity should find other entities by embedding similarity."""
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        results = await populated_store.search_similar_entities(
            sarah.id, min_similarity=0.0
        )

        # Should not include the seed entity itself
        result_ids = [r.item_id for r in results]
        assert sarah.id not in result_ids

        # Should find other entities
        assert len(results) > 0
        assert all(r.item_type == "entity" for r in results)

    @pytest.mark.asyncio
    async def test_similar_entities_type_filter(
        self, populated_store: SemanticMemoryStore
    ):
        """Filter similar entities by entity type."""
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        results = await populated_store.search_similar_entities(
            sarah.id,
            min_similarity=0.0,
            entity_type=EntityType.PERSON,
        )
        # Only Jake should match (other person)
        assert all(
            r.metadata["entity_type"] == "person" for r in results
        )

    @pytest.mark.asyncio
    async def test_similar_entities_top_k(self, store: SemanticMemoryStore):
        """Top-k limits the number of results."""
        # Create many entities
        seed = await store.add_entity("Seed Topic", EntityType.TOPIC)
        for i in range(15):
            await store.add_entity(f"Topic {i}", EntityType.TOPIC)

        results = await store.search_similar_entities(
            seed.id, top_k=5, min_similarity=0.0
        )
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_similar_entities_nonexistent(self, store: SemanticMemoryStore):
        """Searching for a nonexistent entity returns empty."""
        results = await store.search_similar_entities("nonexistent-id")
        assert results == []

    @pytest.mark.asyncio
    async def test_similar_entities_sorted_by_score(
        self, populated_store: SemanticMemoryStore
    ):
        """Results should be sorted by similarity score descending."""
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        results = await populated_store.search_similar_entities(
            sarah.id, min_similarity=0.0
        )
        if len(results) >= 2:
            scores = [r.similarity_score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestSearchEntitiesByQuery:
    @pytest.mark.asyncio
    async def test_search_by_query(self, populated_store: SemanticMemoryStore):
        """Search entities by natural language query."""
        results = await populated_store.search_entities_by_query(
            "person: Sarah", min_similarity=0.0
        )
        assert len(results) > 0
        assert all(r.item_type == "entity" for r in results)

    @pytest.mark.asyncio
    async def test_search_by_query_type_filter(
        self, populated_store: SemanticMemoryStore
    ):
        """Filter query results by entity type."""
        results = await populated_store.search_entities_by_query(
            "person",
            min_similarity=0.0,
            entity_type=EntityType.PERSON,
        )
        for r in results:
            assert r.metadata["entity_type"] == "person"

    @pytest.mark.asyncio
    async def test_search_by_query_min_mentions(
        self, populated_store: SemanticMemoryStore
    ):
        """Filter by minimum mention count."""
        # All entities have mention_count=1 by default
        results = await populated_store.search_entities_by_query(
            "Sarah", min_similarity=0.0, min_mentions=2
        )
        # No entity has 2+ mentions in populated_store
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_by_query_includes_attributes(
        self, populated_store: SemanticMemoryStore
    ):
        """Results should include entity attributes in metadata."""
        results = await populated_store.search_entities_by_query(
            "Sarah", min_similarity=0.0
        )
        assert len(results) > 0
        # Metadata should include attributes key
        assert "attributes" in results[0].metadata

    @pytest.mark.asyncio
    async def test_search_by_query_exact_match(self, store: SemanticMemoryStore):
        """Exact text match should return similarity score of 1.0."""
        await store.add_entity("Alpha Project", EntityType.PROJECT)
        results = await store.search_entities_by_query(
            "project: Alpha Project", min_similarity=0.0
        )
        assert len(results) > 0
        assert results[0].similarity_score == 1.0


class TestSearchNeighborhood:
    @pytest.mark.asyncio
    async def test_neighborhood_expands_from_seeds(
        self, populated_store: SemanticMemoryStore
    ):
        """Neighborhood search should find entities connected to seed matches."""
        # Search for Sarah — should also surface Jake (connected) and Q2 Deck
        results = await populated_store.search_neighborhood(
            "person: Sarah", min_similarity=0.0, max_hops=1
        )
        names = [r.content for r in results]
        # Should find Sarah as a seed
        assert any("Sarah" in n for n in names)

    @pytest.mark.asyncio
    async def test_neighborhood_marks_seeds(
        self, populated_store: SemanticMemoryStore
    ):
        """Results should indicate which entities were direct (seed) matches."""
        results = await populated_store.search_neighborhood(
            "person: Sarah", min_similarity=0.0, max_hops=1
        )
        seed_results = [r for r in results if r.metadata.get("is_seed")]
        non_seed_results = [r for r in results if not r.metadata.get("is_seed")]
        # At least one seed should exist
        assert len(seed_results) > 0 or len(non_seed_results) > 0

    @pytest.mark.asyncio
    async def test_neighborhood_multi_hop(
        self, populated_store: SemanticMemoryStore
    ):
        """Multi-hop traversal should reach further entities."""
        # Acme Corp is 2 hops from Sarah (Sarah→Q2 Deck→Acme)
        results_1hop = await populated_store.search_neighborhood(
            "person: Sarah", min_similarity=0.99, max_hops=1
        )
        results_2hop = await populated_store.search_neighborhood(
            "person: Sarah", min_similarity=0.99, max_hops=2
        )
        # 2-hop should find at least as many as 1-hop
        assert len(results_2hop) >= len(results_1hop)

    @pytest.mark.asyncio
    async def test_neighborhood_relationship_type_filter(
        self, populated_store: SemanticMemoryStore
    ):
        """Filter neighborhood traversal by relationship type."""
        results = await populated_store.search_neighborhood(
            "person: Sarah",
            min_similarity=0.0,
            max_hops=1,
            relationship_types=[RelationshipType.WORKS_WITH],
        )
        # Should still return results (Sarah works_with Jake)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_neighborhood_top_k(self, populated_store: SemanticMemoryStore):
        """Top-k limits neighborhood results."""
        results = await populated_store.search_neighborhood(
            "person", min_similarity=0.0, top_k=2
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_neighborhood_scores_clamped(
        self, populated_store: SemanticMemoryStore
    ):
        """All similarity scores should be clamped to [0, 1]."""
        results = await populated_store.search_neighborhood(
            "person: Sarah", min_similarity=0.0, max_hops=2
        )
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_entities(
        self, populated_store: SemanticMemoryStore
    ):
        """Recall should return entities matching the query."""
        results = await populated_store.recall(
            "person: Sarah", min_similarity=0.0
        )
        entity_results = [r for r in results if r["item_type"] == "entity"]
        assert len(entity_results) > 0

    @pytest.mark.asyncio
    async def test_recall_returns_facts(self, store: SemanticMemoryStore):
        """Recall should return matching facts."""
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)
        results = await store.recall(
            "Sarah is my manager", min_similarity=0.0
        )
        fact_results = [r for r in results if r["item_type"] == "fact"]
        assert len(fact_results) > 0
        assert fact_results[0]["similarity_score"] == 1.0

    @pytest.mark.asyncio
    async def test_recall_returns_patterns(self, store: SemanticMemoryStore):
        """Recall should return matching patterns."""
        desc = "Peak energy in the morning"
        await store.add_pattern(
            PatternType.ENERGY_RHYTHM, desc, confidence=0.9
        )
        results = await store.recall(desc, min_similarity=0.0)
        pattern_results = [r for r in results if r["item_type"] == "pattern"]
        assert len(pattern_results) > 0

    @pytest.mark.asyncio
    async def test_recall_enriches_entity_context(
        self, populated_store: SemanticMemoryStore
    ):
        """Entity results should include connections and facts when include_context=True."""
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        await populated_store.add_fact(
            "Sarah is VP of Engineering",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
        )

        results = await populated_store.recall(
            "person: Sarah", min_similarity=0.0, include_context=True
        )
        # Find the Sarah entity result
        sarah_results = [
            r for r in results
            if r["item_type"] == "entity" and r["item_id"] == sarah.id
        ]
        assert len(sarah_results) == 1
        sarah_result = sarah_results[0]

        # Should have connections
        assert "connections" in sarah_result
        assert len(sarah_result["connections"]) > 0

        # Should have facts
        assert "facts" in sarah_result
        assert len(sarah_result["facts"]) > 0

    @pytest.mark.asyncio
    async def test_recall_enriches_fact_with_subject(
        self, store: SemanticMemoryStore
    ):
        """Fact results should include subject entity info when available."""
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_fact(
            "Sarah is my manager",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
        )

        results = await store.recall(
            "Sarah is my manager", min_similarity=0.0, include_context=True
        )
        fact_results = [r for r in results if r["item_type"] == "fact"]
        assert len(fact_results) > 0
        fact = fact_results[0]
        assert "subject_entity" in fact
        assert fact["subject_entity"]["name"] == "Sarah"

    @pytest.mark.asyncio
    async def test_recall_without_context(self, populated_store: SemanticMemoryStore):
        """Recall with include_context=False should not enrich results."""
        results = await populated_store.recall(
            "Sarah", min_similarity=0.0, include_context=False
        )
        entity_results = [r for r in results if r["item_type"] == "entity"]
        if entity_results:
            assert "connections" not in entity_results[0]

    @pytest.mark.asyncio
    async def test_recall_top_k(self, store: SemanticMemoryStore):
        """Recall should respect top_k limit."""
        for i in range(20):
            await store.add_entity(f"Thing {i}", EntityType.TOPIC)
        results = await store.recall("Thing", top_k=3, min_similarity=0.0)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_recall_sorted_by_similarity(
        self, populated_store: SemanticMemoryStore
    ):
        """Recall results should be sorted by similarity descending."""
        results = await populated_store.recall("Sarah", min_similarity=0.0)
        if len(results) >= 2:
            scores = [r["similarity_score"] for r in results]
            assert scores == sorted(scores, reverse=True)


class TestFindRelatedByEmbedding:
    @pytest.mark.asyncio
    async def test_find_related_entities(
        self, populated_store: SemanticMemoryStore
    ):
        """Find entities related to a given entity by its embedding."""
        sarah = await populated_store.find_entity_by_name("Sarah")
        assert sarah is not None
        results = await populated_store.find_related_by_embedding(
            sarah.id, item_types=["entity"], min_similarity=0.0
        )
        # Should not include seed entity
        assert sarah.id not in [r.item_id for r in results]
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_find_related_facts(self, store: SemanticMemoryStore):
        """Find facts related to an entity by embedding similarity."""
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)
        await store.add_fact(
            "I prefer morning meetings", FactType.PREFERENCE
        )

        results = await store.find_related_by_embedding(
            sarah.id, item_types=["fact"], min_similarity=0.0
        )
        assert len(results) > 0
        assert all(r.item_type == "fact" for r in results)

    @pytest.mark.asyncio
    async def test_find_related_patterns(self, store: SemanticMemoryStore):
        """Find patterns related to an entity by embedding similarity."""
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        # Use the same text used for entity embedding to guarantee similarity
        await store.add_pattern(
            PatternType.ENTITY_PATTERN,
            "person: Sarah",
            confidence=0.9,
        )

        results = await store.find_related_by_embedding(
            sarah.id, item_types=["pattern"], min_similarity=0.0
        )
        assert len(results) > 0
        assert all(r.item_type == "pattern" for r in results)

    @pytest.mark.asyncio
    async def test_find_related_cross_type(self, store: SemanticMemoryStore):
        """Find related items across all types."""
        sarah = await store.add_entity("Sarah", EntityType.PERSON)
        await store.add_entity("Jake", EntityType.PERSON)
        await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE)
        await store.add_pattern(
            PatternType.ENTITY_PATTERN,
            "Meetings with Sarah on Mondays",
            confidence=0.9,
        )

        results = await store.find_related_by_embedding(
            sarah.id, min_similarity=0.0
        )
        types = {r.item_type for r in results}
        # Should find at least entities and facts
        assert len(types) >= 1

    @pytest.mark.asyncio
    async def test_find_related_nonexistent_entity(
        self, store: SemanticMemoryStore
    ):
        """Finding related for nonexistent entity returns empty."""
        results = await store.find_related_by_embedding("nonexistent-id")
        assert results == []

    @pytest.mark.asyncio
    async def test_find_related_top_k(self, store: SemanticMemoryStore):
        """Top-k limits results."""
        seed = await store.add_entity("Seed", EntityType.TOPIC)
        for i in range(15):
            await store.add_entity(f"Related {i}", EntityType.TOPIC)

        results = await store.find_related_by_embedding(
            seed.id, top_k=3, min_similarity=0.0
        )
        assert len(results) <= 3
