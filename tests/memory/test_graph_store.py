"""Tests for entity graph storage layer — graph operations, traversal, merging.

Tests cover:
- EntityGraphStore protocol compliance
- BFS traversal with filters
- Shortest path finding
- Subgraph extraction
- Entity merging and deduplication detection
- Semantic entity discovery
- Batch embedding operations
- Rich entity profile assembly
- Graph export/import
- Relationship-level semantic search
- Entity importance scoring
- Most connected entities ranking
"""

from __future__ import annotations


import pytest
import pytest_asyncio

from blurt.clients.embeddings import MockEmbeddingProvider
from blurt.memory.graph_store import (
    EntityGraphOperations,
    EntityGraphStore,
    EntityMergeResult,
    Subgraph,
    TraversalNode,
)
from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import (
    EntityNode,
    EntityType,
    FactType,
    PatternType,
    RelationshipType,
)

USER_ID = "test-graph-user"


@pytest_asyncio.fixture
async def embedding_provider():
    return MockEmbeddingProvider()


@pytest_asyncio.fixture
async def store(embedding_provider):
    return SemanticMemoryStore(user_id=USER_ID, embedding_provider=embedding_provider)


@pytest_asyncio.fixture
async def graph_ops(store, embedding_provider):
    return EntityGraphOperations(store=store, embedding_provider=embedding_provider)


@pytest_asyncio.fixture
async def populated_graph(graph_ops: EntityGraphOperations):
    """Build a small knowledge graph for traversal tests.

    Graph structure:
        Sarah -- WORKS_WITH --> Jake
        Sarah -- COLLABORATES_ON --> Q2 Deck
        Jake -- COLLABORATES_ON --> Q2 Deck
        Q2 Deck -- PART_OF --> Acme Corp
        Acme Corp -- BASED_IN --> San Francisco
        Jake -- KNOWS --> Bob
    """
    store = graph_ops.store

    sarah = await store.add_entity("Sarah", EntityType.PERSON)
    jake = await store.add_entity("Jake", EntityType.PERSON)
    bob = await store.add_entity("Bob", EntityType.PERSON)
    q2_deck = await store.add_entity("Q2 Deck", EntityType.PROJECT)
    acme = await store.add_entity("Acme Corp", EntityType.ORGANIZATION)
    sf = await store.add_entity("San Francisco", EntityType.PLACE)

    await store.add_or_strengthen_relationship(
        sarah.id, jake.id, RelationshipType.WORKS_WITH, "team members"
    )
    await store.add_or_strengthen_relationship(
        sarah.id, q2_deck.id, RelationshipType.COLLABORATES_ON, "sarah on deck"
    )
    await store.add_or_strengthen_relationship(
        jake.id, q2_deck.id, RelationshipType.COLLABORATES_ON, "jake on deck"
    )
    await store.add_or_strengthen_relationship(
        q2_deck.id, acme.id, RelationshipType.PART_OF, "deck for acme"
    )
    await store.add_or_strengthen_relationship(
        acme.id, sf.id, RelationshipType.BASED_IN, "acme in sf"
    )
    await store.add_or_strengthen_relationship(
        jake.id, bob.id, RelationshipType.KNOWS, "jake knows bob"
    )

    return graph_ops


# ── Protocol Compliance ──────────────────────────────────────────────


class TestProtocolCompliance:
    def test_semantic_memory_store_implements_protocol(self):
        """SemanticMemoryStore should satisfy the EntityGraphStore protocol."""
        provider = MockEmbeddingProvider()
        store = SemanticMemoryStore(user_id="u1", embedding_provider=provider)
        assert isinstance(store, EntityGraphStore)

    def test_protocol_has_required_methods(self):
        """Verify the protocol defines all expected methods."""
        required_methods = [
            "add_entity", "get_entity", "find_entity_by_name",
            "get_all_entities", "update_entity_embedding",
            "add_or_strengthen_relationship", "get_entity_relationships",
            "get_connected_entities", "decay_relationships",
            "add_fact", "get_entity_facts", "get_all_facts", "supersede_fact",
            "add_pattern", "confirm_pattern", "get_active_patterns",
            "search", "get_entity_context", "get_graph_stats",
            "process_extracted_entities", "process_extracted_facts",
        ]
        for method in required_methods:
            assert hasattr(EntityGraphStore, method)


# ── BFS Traversal ────────────────────────────────────────────────────


class TestBFSTraversal:
    @pytest.mark.asyncio
    async def test_basic_traversal(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        result = await populated_graph.traverse_bfs(sarah.id, max_depth=1)

        assert len(result) >= 1  # At least Sarah herself
        assert result[0].entity.id == sarah.id
        assert result[0].depth == 0

        # Depth-1 neighbors should include Jake and Q2 Deck
        depth_1_names = {n.entity.name for n in result if n.depth == 1}
        assert "Jake" in depth_1_names
        assert "Q2 Deck" in depth_1_names

    @pytest.mark.asyncio
    async def test_traversal_depth_2(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        result = await populated_graph.traverse_bfs(sarah.id, max_depth=2)

        # Depth 2 should reach Acme Corp (through Q2 Deck) and Bob (through Jake)
        all_names = {n.entity.name for n in result}
        assert "Acme Corp" in all_names
        assert "Bob" in all_names

    @pytest.mark.asyncio
    async def test_traversal_max_nodes(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        result = await populated_graph.traverse_bfs(sarah.id, max_depth=5, max_nodes=3)
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_traversal_entity_type_filter(
        self, populated_graph: EntityGraphOperations
    ):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        result = await populated_graph.traverse_bfs(
            sarah.id, max_depth=3, entity_type_filter=EntityType.PERSON
        )

        # Only Sarah (depth 0, unfiltered) and other PERSON entities
        for node in result:
            if node.depth > 0:
                assert node.entity.entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_traversal_relationship_type_filter(
        self, populated_graph: EntityGraphOperations
    ):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        result = await populated_graph.traverse_bfs(
            sarah.id,
            max_depth=3,
            relationship_type_filter=RelationshipType.WORKS_WITH,
        )

        # Should only follow WORKS_WITH edges (Sarah -> Jake)
        depth_1_names = {n.entity.name for n in result if n.depth == 1}
        assert "Jake" in depth_1_names
        assert "Q2 Deck" not in depth_1_names

    @pytest.mark.asyncio
    async def test_traversal_nonexistent_entity(
        self, populated_graph: EntityGraphOperations
    ):
        result = await populated_graph.traverse_bfs("nonexistent-id")
        assert result == []

    @pytest.mark.asyncio
    async def test_traversal_path_tracking(
        self, populated_graph: EntityGraphOperations
    ):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        result = await populated_graph.traverse_bfs(sarah.id, max_depth=3)

        for node in result:
            assert node.path[0] == sarah.id
            assert len(node.path) == node.depth + 1
            assert node.path[-1] == node.entity.id


# ── Shortest Path ────────────────────────────────────────────────────


class TestFindPath:
    @pytest.mark.asyncio
    async def test_direct_path(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        jake = await populated_graph.store.find_entity_by_name("Jake")
        assert sarah is not None
        assert jake is not None

        path = await populated_graph.find_path(sarah.id, jake.id)
        assert path is not None
        assert len(path) == 2
        assert path[0].entity.id == sarah.id
        assert path[-1].entity.id == jake.id

    @pytest.mark.asyncio
    async def test_multi_hop_path(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        sf = await populated_graph.store.find_entity_by_name("San Francisco")
        assert sarah is not None
        assert sf is not None

        path = await populated_graph.find_path(sarah.id, sf.id)
        assert path is not None
        # Sarah -> Q2 Deck -> Acme Corp -> San Francisco
        assert len(path) >= 3

    @pytest.mark.asyncio
    async def test_same_entity_path(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        path = await populated_graph.find_path(sarah.id, sarah.id)
        assert path is not None
        assert len(path) == 1

    @pytest.mark.asyncio
    async def test_no_path(self, populated_graph: EntityGraphOperations):
        """No path when entities are disconnected."""
        store = populated_graph.store
        isolated = await store.add_entity("Isolated Node", EntityType.TOPIC)
        sarah = await store.find_entity_by_name("Sarah")
        assert sarah is not None

        path = await populated_graph.find_path(sarah.id, isolated.id)
        assert path is None

    @pytest.mark.asyncio
    async def test_nonexistent_source(self, populated_graph: EntityGraphOperations):
        jake = await populated_graph.store.find_entity_by_name("Jake")
        assert jake is not None
        path = await populated_graph.find_path("nonexistent", jake.id)
        assert path is None


# ── Subgraph Extraction ──────────────────────────────────────────────


class TestSubgraphExtraction:
    @pytest.mark.asyncio
    async def test_extract_neighborhood(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        subgraph = await populated_graph.extract_neighborhood(sarah.id, depth=1)

        assert isinstance(subgraph, Subgraph)
        assert subgraph.center_entity_id == sarah.id
        assert subgraph.entity_count >= 3  # Sarah, Jake, Q2 Deck
        assert subgraph.relationship_count >= 2

    @pytest.mark.asyncio
    async def test_subgraph_depth_2(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        subgraph = await populated_graph.extract_neighborhood(sarah.id, depth=2)

        # Depth 2 should include more entities
        assert subgraph.entity_count > 3
        entity_names = {e.name for e in subgraph.entities}
        assert "Acme Corp" in entity_names or "Bob" in entity_names

    @pytest.mark.asyncio
    async def test_subgraph_to_dict(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        subgraph = await populated_graph.extract_neighborhood(sarah.id, depth=1)

        d = subgraph.to_dict()
        assert "entities" in d
        assert "relationships" in d
        assert d["center_entity_id"] == sarah.id
        assert d["entity_count"] == subgraph.entity_count

    @pytest.mark.asyncio
    async def test_subgraph_get_entity(self, populated_graph: EntityGraphOperations):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        subgraph = await populated_graph.extract_neighborhood(sarah.id, depth=1)

        found = subgraph.get_entity(sarah.id)
        assert found is not None
        assert found.id == sarah.id

        not_found = subgraph.get_entity("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_subgraph_only_internal_relationships(
        self, populated_graph: EntityGraphOperations
    ):
        """Subgraph should only contain relationships between included entities."""
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        subgraph = await populated_graph.extract_neighborhood(sarah.id, depth=1)

        entity_ids = {e.id for e in subgraph.entities}
        for rel in subgraph.relationships:
            assert rel.source_entity_id in entity_ids
            assert rel.target_entity_id in entity_ids


# ── Entity Merging ───────────────────────────────────────────────────


class TestEntityMerging:
    @pytest.mark.asyncio
    async def test_merge_entities(self, graph_ops: EntityGraphOperations):
        store = graph_ops.store

        sarah_full = await store.add_entity(
            "Sarah Chen", EntityType.PERSON,
            attributes={"role": "manager", "team": "platform"},
        )
        sarah_short = await store.add_entity(
            "Sarah", EntityType.PERSON,
            attributes={"email": "sarah@acme.com"},
        )

        # Add some data to the entity being removed
        project = await store.add_entity("Project X", EntityType.PROJECT)
        await store.add_or_strengthen_relationship(
            sarah_short.id, project.id, RelationshipType.COLLABORATES_ON
        )
        await store.add_fact(
            "Sarah prefers morning standups",
            FactType.PREFERENCE,
            subject_entity_id=sarah_short.id,
        )

        result = await graph_ops.merge_entities(sarah_full.id, sarah_short.id)

        assert result is not None
        assert isinstance(result, EntityMergeResult)
        assert result.merged_entity.id == sarah_full.id
        assert result.removed_entity_id == sarah_short.id
        assert result.relationships_transferred >= 1
        assert result.facts_transferred >= 1

        # Merged entity should have combined attributes
        merged = result.merged_entity
        assert merged.attributes.get("role") == "manager"
        assert merged.attributes.get("email") == "sarah@acme.com"

        # Aliases should include the removed entity's name
        assert "sarah" in merged.aliases

    @pytest.mark.asyncio
    async def test_merge_accumulates_mentions(self, graph_ops: EntityGraphOperations):
        store = graph_ops.store
        e1 = await store.add_entity("Alice", EntityType.PERSON)
        e2 = await store.add_entity("Alice B", EntityType.PERSON)
        # Mention e1 again to increment count
        await store.add_entity("Alice", EntityType.PERSON)

        result = await graph_ops.merge_entities(e1.id, e2.id)
        assert result is not None
        assert result.merged_entity.mention_count == 3  # 2 + 1

    @pytest.mark.asyncio
    async def test_merge_nonexistent(self, graph_ops: EntityGraphOperations):
        store = graph_ops.store
        entity = await store.add_entity("Test", EntityType.TOPIC)
        result = await graph_ops.merge_entities(entity.id, "nonexistent")
        assert result is None


# ── Duplicate Detection ──────────────────────────────────────────────


class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_find_potential_duplicates(self, graph_ops: EntityGraphOperations):
        store = graph_ops.store
        # Two entities with very similar names should be flagged
        await store.add_entity("Sarah Chen", EntityType.PERSON)
        await store.add_entity("Bob Smith", EntityType.PERSON)

        # With mock embeddings, exact text = exact match
        duplicates = await graph_ops.find_potential_duplicates(
            similarity_threshold=0.0  # Low threshold to catch anything
        )
        # The entities have different names so mock embeddings differ
        assert isinstance(duplicates, list)

    @pytest.mark.asyncio
    async def test_no_cross_type_duplicates(self, graph_ops: EntityGraphOperations):
        """Duplicate detection only compares within same entity type."""
        store = graph_ops.store
        await store.add_entity("Python", EntityType.TOOL)
        await store.add_entity("Python", EntityType.TOPIC)

        duplicates = await graph_ops.find_potential_duplicates(
            similarity_threshold=0.9
        )
        # Different types, should not be flagged as duplicates
        assert all(d[0].entity_type == d[1].entity_type for d in duplicates)


# ── Semantic Entity Discovery ────────────────────────────────────────


class TestSemanticDiscovery:
    @pytest.mark.asyncio
    async def test_find_similar_entities(
        self, populated_graph: EntityGraphOperations
    ):
        # Search with exact entity name to guarantee match with mock embeddings
        results = await populated_graph.find_similar_entities(
            "person: Sarah", min_similarity=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], EntityNode) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    @pytest.mark.asyncio
    async def test_find_similar_with_type_filter(
        self, populated_graph: EntityGraphOperations
    ):
        results = await populated_graph.find_similar_entities(
            "person: Sarah",
            entity_type=EntityType.PERSON,
            min_similarity=0.0,
        )
        for entity, score in results:
            assert entity.entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_find_similar_top_k(
        self, populated_graph: EntityGraphOperations
    ):
        results = await populated_graph.find_similar_entities(
            "person", top_k=2, min_similarity=0.0
        )
        assert len(results) <= 2


# ── Batch Embedding Operations ───────────────────────────────────────


class TestBatchEmbeddings:
    @pytest.mark.asyncio
    async def test_reindex_all_embeddings(
        self, populated_graph: EntityGraphOperations
    ):
        count = await populated_graph.reindex_all_embeddings()
        assert count == 6  # All 6 entities in populated graph


# ── Rich Entity Profile ─────────────────────────────────────────────


class TestEntityProfile:
    @pytest.mark.asyncio
    async def test_build_entity_profile(
        self, populated_graph: EntityGraphOperations
    ):
        store = populated_graph.store
        sarah = await store.find_entity_by_name("Sarah")
        assert sarah is not None

        # Add a fact and pattern to make the profile richer
        await store.add_fact(
            "Sarah is VP of Engineering",
            FactType.ATTRIBUTE,
            subject_entity_id=sarah.id,
        )
        await store.add_pattern(
            PatternType.ENTITY_PATTERN,
            "Sarah is frequently mentioned with Q2 Deck",
            confidence=0.9,
        )

        profile = await populated_graph.build_entity_profile(sarah.id)

        assert "entity" in profile
        assert profile["entity"]["name"] == "Sarah"
        assert "facts" in profile
        assert "connections" in profile
        assert "patterns" in profile
        assert "semantic_neighbors" in profile

    @pytest.mark.asyncio
    async def test_build_profile_nonexistent(
        self, populated_graph: EntityGraphOperations
    ):
        profile = await populated_graph.build_entity_profile("nonexistent")
        assert profile == {}


# ── Graph Export/Import ──────────────────────────────────────────────


class TestGraphExportImport:
    @pytest.mark.asyncio
    async def test_export_graph(self, populated_graph: EntityGraphOperations):
        data = await populated_graph.export_graph()

        assert data["version"] == "1.0"
        assert data["user_id"] == USER_ID
        assert "exported_at" in data
        assert len(data["entities"]) == 6
        assert len(data["relationships"]) == 6
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_export_without_embeddings(
        self, populated_graph: EntityGraphOperations
    ):
        data = await populated_graph.export_graph(include_embeddings=False)

        for entity in data["entities"]:
            assert "embedding" not in entity

    @pytest.mark.asyncio
    async def test_export_with_embeddings(
        self, populated_graph: EntityGraphOperations
    ):
        data = await populated_graph.export_graph(include_embeddings=True)

        # At least some entities should have embeddings
        entities_with_embeddings = [
            e for e in data["entities"] if e.get("embedding") is not None
        ]
        assert len(entities_with_embeddings) > 0

    @pytest.mark.asyncio
    async def test_import_into_empty_store(
        self, populated_graph: EntityGraphOperations, embedding_provider
    ):
        # Export from populated graph
        data = await populated_graph.export_graph()

        # Import into a fresh store
        new_store = SemanticMemoryStore(
            user_id="import-test-user", embedding_provider=embedding_provider
        )
        new_ops = EntityGraphOperations(
            store=new_store, embedding_provider=embedding_provider
        )

        counts = await new_ops.import_entities(data, regenerate_embeddings=True)

        assert counts["entities"] == 6
        assert counts["relationships"] == 6

        # Verify imported data
        stats = await new_store.get_graph_stats()
        assert stats["total_entities"] == 6
        assert stats["total_relationships"] >= 6


# ── Relationship-Level Search ────────────────────────────────────────


class TestRelationshipSearch:
    @pytest.mark.asyncio
    async def test_search_relationships(
        self, populated_graph: EntityGraphOperations
    ):
        results = await populated_graph.search_relationships(
            "team members", min_similarity=0.0
        )
        assert isinstance(results, list)
        # Should find relationships with context snippets
        if results:
            assert "relationship" in results[0]
            assert "similarity_score" in results[0]

    @pytest.mark.asyncio
    async def test_search_relationships_top_k(
        self, populated_graph: EntityGraphOperations
    ):
        results = await populated_graph.search_relationships(
            "working", top_k=2, min_similarity=0.0
        )
        assert len(results) <= 2


# ── Entity Importance ────────────────────────────────────────────────


class TestEntityImportance:
    @pytest.mark.asyncio
    async def test_entity_importance_score(
        self, populated_graph: EntityGraphOperations
    ):
        sarah = await populated_graph.store.find_entity_by_name("Sarah")
        assert sarah is not None
        score = await populated_graph.get_entity_importance(sarah.id)

        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Sarah has connections

    @pytest.mark.asyncio
    async def test_importance_nonexistent(
        self, populated_graph: EntityGraphOperations
    ):
        score = await populated_graph.get_entity_importance("nonexistent")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_connected_entity_more_important(
        self, populated_graph: EntityGraphOperations
    ):
        """Entity with more connections should have higher importance."""
        store = populated_graph.store
        sarah = await store.find_entity_by_name("Sarah")  # 2 connections
        sf = await store.find_entity_by_name("San Francisco")  # 1 connection
        assert sarah is not None
        assert sf is not None

        sarah_score = await populated_graph.get_entity_importance(sarah.id)
        sf_score = await populated_graph.get_entity_importance(sf.id)

        assert sarah_score >= sf_score


# ── Most Connected Entities ──────────────────────────────────────────


class TestMostConnected:
    @pytest.mark.asyncio
    async def test_most_connected(self, populated_graph: EntityGraphOperations):
        results = await populated_graph.get_most_connected_entities(top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)

        # Should be sorted descending by connection count
        counts = [count for _, count in results]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.asyncio
    async def test_most_connected_with_type_filter(
        self, populated_graph: EntityGraphOperations
    ):
        results = await populated_graph.get_most_connected_entities(
            entity_type=EntityType.PERSON
        )

        for entity, count in results:
            assert entity.entity_type == EntityType.PERSON


# ── TraversalNode Model ──────────────────────────────────────────────


class TestTraversalNode:
    def test_repr(self):
        entity = EntityNode(
            user_id="u1", name="Test", entity_type=EntityType.TOPIC
        )
        node = TraversalNode(entity=entity, depth=2, path=["a", "b", "c"])
        repr_str = repr(node)
        assert "Test" in repr_str
        assert "depth=2" in repr_str

    def test_relationship_optional(self):
        entity = EntityNode(
            user_id="u1", name="Test", entity_type=EntityType.TOPIC
        )
        node = TraversalNode(entity=entity, depth=0, path=["a"])
        assert node.relationship is None


# ── Vector Embedding Fields ──────────────────────────────────────────


class TestVectorEmbeddingFields:
    """Verify that all graph data models carry vector embedding fields."""

    def test_entity_node_has_embedding_field(self):
        node = EntityNode(
            user_id="u1", name="Test", entity_type=EntityType.PERSON
        )
        assert hasattr(node, "embedding")
        assert node.embedding is None  # Optional until set

    @pytest.mark.asyncio
    async def test_entity_gets_embedding_on_creation(self, store):
        entity = await store.add_entity("Test Person", EntityType.PERSON)
        assert entity.embedding is not None
        assert len(entity.embedding) == MockEmbeddingProvider.DIMENSION

    @pytest.mark.asyncio
    async def test_fact_gets_embedding_on_creation(self, store):
        fact = await store.add_fact("Test fact", FactType.ATTRIBUTE)
        assert fact.embedding is not None
        assert len(fact.embedding) == MockEmbeddingProvider.DIMENSION

    @pytest.mark.asyncio
    async def test_pattern_gets_embedding_on_creation(self, store):
        pattern = await store.add_pattern(
            PatternType.TIME_OF_DAY, "Morning person"
        )
        assert pattern.embedding is not None
        assert len(pattern.embedding) == MockEmbeddingProvider.DIMENSION

    @pytest.mark.asyncio
    async def test_embedding_enables_similarity_search(self, store):
        """Entities with embeddings should be findable via semantic search."""
        await store.add_entity("Alice Manager", EntityType.PERSON)
        await store.add_entity("Bob Engineer", EntityType.PERSON)

        results = await store.search("person: Alice Manager", min_similarity=0.0)
        assert len(results) > 0
        # Exact match should have score 1.0
        assert results[0].similarity_score == 1.0
