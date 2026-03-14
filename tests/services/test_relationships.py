"""Tests for the relationship tracking service.

Validates that the service correctly detects, scores, and stores
semantic relationships between entities using embeddings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blurt.clients.embeddings import MockEmbeddingProvider
from blurt.models.entities import (
    EntityNode,
    EntityType,
    RelationshipEdge,
    RelationshipType,
    is_valid_relationship,
)
from blurt.services.relationships import (
    RelationshipConfig,
    RelationshipDetectionMode,
    RelationshipDetectionResult,
    RelationshipScore,
    RelationshipTrackingService,
    infer_relationship_type,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def embedding_provider():
    return MockEmbeddingProvider()


@pytest.fixture
def config():
    return RelationshipConfig()


@pytest.fixture
def service(embedding_provider, config):
    return RelationshipTrackingService(
        embedding_provider=embedding_provider,
        config=config,
    )


@pytest.fixture
def user_id():
    return "test-user-123"


async def _make_entity(
    provider: MockEmbeddingProvider,
    name: str,
    entity_type: EntityType,
    user_id: str = "test-user-123",
    mention_count: int = 3,
    aliases: list[str] | None = None,
) -> EntityNode:
    """Helper to create an EntityNode with a real embedding."""
    embedding = await provider.embed(f"{entity_type.value}: {name}")
    return EntityNode(
        user_id=user_id,
        name=name,
        entity_type=entity_type,
        mention_count=mention_count,
        embedding=embedding,
        aliases=aliases or [],
    )


# ── Service Initialization ───────────────────────────────────────────


class TestServiceInit:
    def test_creates_with_defaults(self, embedding_provider):
        svc = RelationshipTrackingService(embedding_provider=embedding_provider)
        assert svc.config.mode == RelationshipDetectionMode.BALANCED
        assert svc.relationship_count == 0

    def test_creates_with_custom_config(self, embedding_provider):
        cfg = RelationshipConfig(mode=RelationshipDetectionMode.CONSERVATIVE)
        svc = RelationshipTrackingService(embedding_provider=embedding_provider, config=cfg)
        assert svc.config.implicit_similarity_threshold == 0.80
        assert svc.config.min_mentions_for_implicit == 3

    def test_aggressive_mode_lowers_thresholds(self, embedding_provider):
        cfg = RelationshipConfig(mode=RelationshipDetectionMode.AGGRESSIVE)
        svc = RelationshipTrackingService(embedding_provider=embedding_provider, config=cfg)
        assert svc.config.implicit_similarity_threshold == 0.50
        assert svc.config.max_implicit_per_batch == 20
        assert svc.config.min_mentions_for_implicit == 1


# ── Relationship Type Inference ──────────────────────────────────────


class TestRelationshipTypeInference:
    def test_person_person_defaults_to_knows(self):
        result = infer_relationship_type(EntityType.PERSON, EntityType.PERSON)
        assert result == RelationshipType.KNOWS

    def test_person_org_defaults_to_member_of(self):
        result = infer_relationship_type(EntityType.PERSON, EntityType.ORGANIZATION)
        assert result == RelationshipType.MEMBER_OF

    def test_person_project_defaults_to_collaborates_on(self):
        result = infer_relationship_type(EntityType.PERSON, EntityType.PROJECT)
        assert result == RelationshipType.COLLABORATES_ON

    def test_person_place_defaults_to_located_at(self):
        result = infer_relationship_type(EntityType.PERSON, EntityType.PLACE)
        assert result == RelationshipType.LOCATED_AT

    def test_org_place_defaults_to_based_in(self):
        result = infer_relationship_type(EntityType.ORGANIZATION, EntityType.PLACE)
        assert result == RelationshipType.BASED_IN

    def test_project_project_defaults_to_depends_on(self):
        result = infer_relationship_type(EntityType.PROJECT, EntityType.PROJECT)
        assert result == RelationshipType.DEPENDS_ON

    def test_unknown_pair_defaults_to_related_to(self):
        result = infer_relationship_type(EntityType.TOOL, EntityType.TOOL)
        assert result == RelationshipType.RELATED_TO

    def test_context_overrides_default_for_work(self):
        result = infer_relationship_type(
            EntityType.PERSON, EntityType.PERSON,
            context="I work with Sarah on the project"
        )
        assert result == RelationshipType.WORKS_WITH

    def test_context_overrides_for_manages(self):
        result = infer_relationship_type(
            EntityType.PERSON, EntityType.PERSON,
            context="Sarah manages the engineering team"
        )
        assert result == RelationshipType.MANAGES

    def test_context_friend(self):
        result = infer_relationship_type(
            EntityType.PERSON, EntityType.PERSON,
            context="She is a friend from college"
        )
        assert result == RelationshipType.FRIEND_OF

    def test_context_company_employment(self):
        result = infer_relationship_type(
            EntityType.PERSON, EntityType.ORGANIZATION,
            context="I work at the company"
        )
        # "work" maps to WORKS_WITH, but that's not valid for person->org,
        # so it falls through to "company" -> EMPLOYED_BY
        assert result in (RelationshipType.EMPLOYED_BY, RelationshipType.MEMBER_OF)

    def test_context_with_invalid_pair_falls_through(self):
        # "manage" is not valid for TOOL->TOOL, should fall through
        result = infer_relationship_type(
            EntityType.TOOL, EntityType.TOOL,
            context="I manage all my tools"
        )
        assert result == RelationshipType.RELATED_TO


# ── Explicit Relationship Processing ─────────────────────────────────


class TestExplicitRelationships:
    async def test_process_explicit_relationship(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        result = await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            context="Sarah works with Bob on the project",
            user_id=user_id,
        )

        assert len(result.explicit_relationships) == 1
        edge = result.explicit_relationships[0]
        assert edge.source_entity_id == sarah.id
        assert edge.target_entity_id == bob.id
        assert edge.relationship_type == RelationshipType.WORKS_WITH
        assert edge.strength > 0
        assert service.relationship_count == 1

    async def test_strengthens_existing_relationship(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        # First mention
        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        edge_before = (await service.get_entity_relationships(sarah.id))[0]
        strength_before = edge_before.strength

        # Second mention (strengthens)
        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        edge_after = (await service.get_entity_relationships(sarah.id))[0]
        assert edge_after.strength > strength_before
        assert edge_after.co_mention_count == 2
        # Should still be just 1 relationship (strengthened, not duplicated)
        assert service.relationship_count == 1

    async def test_invalid_relationship_type_gets_inferred(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        acme = await _make_entity(embedding_provider, "Acme Corp", EntityType.ORGANIZATION, user_id)

        result = await service.process_extraction(
            entities=[sarah, acme],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "acme corp",
                "relationship_type": "works_with",  # Not valid for person->org
                "confidence": 0.9,
            }],
            context="Sarah works at the company Acme Corp",
            user_id=user_id,
        )

        assert len(result.explicit_relationships) == 1
        edge = result.explicit_relationships[0]
        # Should infer a valid type for person->org
        assert is_valid_relationship(
            EntityType.PERSON, EntityType.ORGANIZATION, edge.relationship_type
        )

    async def test_skips_speaker_source(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)

        result = await service.process_extraction(
            entities=[sarah],
            explicit_relationships=[{
                "source_name": "speaker",
                "target_name": "sarah",
                "relationship_type": "knows",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        assert len(result.explicit_relationships) == 0

    async def test_skips_self_referential(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)

        result = await service.process_extraction(
            entities=[sarah],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "sarah",
                "relationship_type": "related_to",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        assert len(result.explicit_relationships) == 0

    async def test_context_stored_in_snippets(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)
        context = "Sarah and Bob discussed the launch plan"

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            context=context,
            user_id=user_id,
        )

        edge = (await service.get_entity_relationships(sarah.id))[0]
        assert context in edge.context_snippets


# ── Implicit Relationship Detection ──────────────────────────────────


class TestImplicitRelationships:
    async def test_detects_no_implicit_with_single_entity(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)

        result = await service.process_extraction(
            entities=[sarah],
            user_id=user_id,
        )

        assert len(result.implicit_relationships) == 0

    async def test_entities_below_mention_threshold_excluded(self, embedding_provider, user_id):
        """Entities with too few mentions are not eligible for implicit detection."""
        config = RelationshipConfig(min_mentions_for_implicit=5)
        svc = RelationshipTrackingService(embedding_provider=embedding_provider, config=config)

        # Only 2 mentions, threshold is 5
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id, mention_count=2)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id, mention_count=2)

        result = await svc.process_extraction(
            entities=[sarah, bob],
            user_id=user_id,
        )

        assert len(result.implicit_relationships) == 0

    async def test_respects_max_implicit_per_batch(self, embedding_provider, user_id):
        config = RelationshipConfig(
            max_implicit_per_batch=1,
            implicit_similarity_threshold=0.0,  # Accept everything
            min_mentions_for_implicit=1,
        )
        svc = RelationshipTrackingService(embedding_provider=embedding_provider, config=config)

        entities = [
            await _make_entity(embedding_provider, f"Entity{i}", EntityType.PERSON, user_id, mention_count=5)
            for i in range(5)
        ]

        result = await svc.process_extraction(entities=entities, user_id=user_id)
        assert len(result.implicit_relationships) <= 1

    async def test_no_implicit_when_explicit_exists(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        # Create explicit relationship first
        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        # Second pass should not create implicit on top of explicit
        result = await service.process_extraction(
            entities=[sarah, bob],
            user_id=user_id,
        )
        assert len(result.implicit_relationships) == 0


# ── Relationship Scoring ─────────────────────────────────────────────


class TestRelationshipScoring:
    async def test_score_computation(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        result = await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        assert len(result.scores) >= 1
        score = result.scores[0]
        assert score.source_entity_id == sarah.id
        assert score.target_entity_id == bob.id
        assert score.composite_score >= 0
        assert score.embedding_similarity >= 0
        assert score.co_mention_score >= 0
        assert score.recency_score >= 0

    async def test_score_entity_pair(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        score = await service.score_entity_pair(sarah, bob)
        assert score.embedding_similarity >= 0
        assert score.composite_score >= 0
        assert score.relationship_type == RelationshipType.RELATED_TO  # No explicit yet

    async def test_score_with_existing_relationship(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        # Create relationship
        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        score = await service.score_entity_pair(sarah, bob)
        assert score.is_explicit is True
        assert score.relationship_type == RelationshipType.WORKS_WITH
        assert score.co_mention_score > 0

    async def test_composite_score_uses_weights(self):
        config = RelationshipConfig()
        score = RelationshipScore(
            source_entity_id="a",
            target_entity_id="b",
            relationship_type=RelationshipType.RELATED_TO,
            embedding_similarity=0.8,
            co_mention_score=0.5,
            recency_score=1.0,
        )
        composite = score.compute_composite(config)
        expected = (
            config.embedding_weight * 0.8
            + config.co_mention_weight * 0.5
            + config.recency_weight * 1.0
        )
        assert abs(composite - expected) < 1e-6


# ── Queries ──────────────────────────────────────────────────────────


class TestQueries:
    async def test_find_related_entities(self, service, embedding_provider, user_id):
        entities = [
            await _make_entity(embedding_provider, f"Person{i}", EntityType.PERSON, user_id)
            for i in range(5)
        ]

        related = await service.find_related_entities(
            entity=entities[0],
            all_entities=entities,
            min_similarity=-1.0,  # Accept any similarity (mock embeddings can be negative)
        )

        # Should find some of the others (not itself)
        assert len(related) > 0
        entity_ids = [e.id for e, _ in related]
        assert entities[0].id not in entity_ids

    async def test_find_related_excludes_no_embedding(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        no_embed = EntityNode(
            user_id=user_id, name="NoEmbed", entity_type=EntityType.PERSON,
            embedding=None,
        )

        related = await service.find_related_entities(
            entity=sarah,
            all_entities=[sarah, no_embed],
        )

        entity_ids = [e.id for e, _ in related]
        assert no_embed.id not in entity_ids

    async def test_get_relationship_between(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        rel = await service.get_relationship_between(sarah.id, bob.id)
        assert rel is not None
        assert rel.relationship_type == RelationshipType.WORKS_WITH

        # Also works in reverse direction
        rel_rev = await service.get_relationship_between(bob.id, sarah.id)
        assert rel_rev is not None

    async def test_get_relationship_between_nonexistent(self, service, user_id):
        rel = await service.get_relationship_between("no-id-a", "no-id-b")
        assert rel is None

    async def test_get_strongest_connections(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)
        alice = await _make_entity(embedding_provider, "Alice", EntityType.PERSON, user_id)

        # Create two relationships with different strengths
        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )
        # Strengthen sarah-bob
        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        await service.process_extraction(
            entities=[sarah, alice],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "alice",
                "relationship_type": "knows",
                "confidence": 0.5,
            }],
            user_id=user_id,
        )

        connections = await service.get_strongest_connections(sarah.id)
        assert len(connections) == 2
        # Bob should be stronger (2 mentions)
        assert connections[0][0] == bob.id

    async def test_get_strongest_connections_with_min_strength(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        # Very high threshold should filter everything
        connections = await service.get_strongest_connections(sarah.id, min_strength=999.0)
        assert len(connections) == 0


# ── Entity Clusters ──────────────────────────────────────────────────


class TestEntityClusters:
    async def test_find_clusters_groups_similar(self, service, embedding_provider, user_id):
        # Create entities with varied types
        entities = [
            await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id),
            await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id),
            await _make_entity(embedding_provider, "Notion HQ", EntityType.PLACE, user_id),
        ]

        # Use negative threshold to ensure clustering works with mock embeddings
        # (mock embeddings can produce negative cosine similarity)
        clusters = await service.find_entity_clusters(
            entities, similarity_threshold=-1.0
        )

        # With threshold -1, all entities form one cluster
        assert len(clusters) >= 1
        total_in_clusters = sum(len(c) for c in clusters)
        assert total_in_clusters == 3

    async def test_find_clusters_excludes_singletons(self, service, embedding_provider, user_id):
        # With high threshold, no clusters should form
        entities = [
            await _make_entity(embedding_provider, f"Unique_{i}", EntityType.PERSON, user_id)
            for i in range(3)
        ]

        clusters = await service.find_entity_clusters(
            entities, similarity_threshold=0.9999
        )
        # Very high threshold means no entities cluster
        # (could be 0 if embeddings are very different)
        for cluster in clusters:
            assert len(cluster) >= 2

    async def test_empty_entities_returns_no_clusters(self, service):
        clusters = await service.find_entity_clusters([])
        assert clusters == []


# ── Lifecycle: Decay and Pruning ─────────────────────────────────────


class TestLifecycle:
    async def test_decay_reduces_strength(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        edge = (await service.get_entity_relationships(sarah.id))[0]
        original_strength = edge.strength

        # Decay as if 60 days passed (2 half-lives)
        future = datetime.now(timezone.utc) + timedelta(days=60)
        await service.decay_relationships(as_of=future)

        assert edge.strength < original_strength
        # After 2 half-lives, strength should be ~25% of original
        assert edge.strength < original_strength * 0.3

    async def test_decay_no_effect_on_recent(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        edge = (await service.get_entity_relationships(sarah.id))[0]
        original_strength = edge.strength

        # Decay with the exact last_seen time (0 days passed)
        await service.decay_relationships(as_of=edge.last_seen)
        assert edge.strength == original_strength

    async def test_prune_removes_dormant(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        assert service.relationship_count == 1

        # Decay far into the future
        far_future = datetime.now(timezone.utc) + timedelta(days=365)
        await service.decay_relationships(as_of=far_future)

        # Now prune
        pruned = await service.prune_dormant_relationships()
        assert pruned == 1
        assert service.relationship_count == 0

    async def test_prune_keeps_strong_relationships(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        # Prune without decay — relationship is still strong
        pruned = await service.prune_dormant_relationships()
        assert pruned == 0
        assert service.relationship_count == 1

    async def test_prune_cleans_up_indexes(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        # Decay and prune
        far_future = datetime.now(timezone.utc) + timedelta(days=365)
        await service.decay_relationships(as_of=far_future)
        await service.prune_dormant_relationships()

        # Indexes should be clean
        assert len(await service.get_entity_relationships(sarah.id)) == 0
        assert len(await service.get_entity_relationships(bob.id)) == 0


# ── Statistics ───────────────────────────────────────────────────────


class TestStats:
    async def test_stats_empty(self, service):
        stats = await service.get_stats()
        assert stats["total_relationships"] == 0
        assert stats["active_relationships"] == 0
        assert stats["average_strength"] == 0

    async def test_stats_with_relationships(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)
        alice = await _make_entity(embedding_provider, "Alice", EntityType.PERSON, user_id)

        await service.process_extraction(
            entities=[sarah, bob],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "bob",
                "relationship_type": "works_with",
                "confidence": 0.9,
            }],
            user_id=user_id,
        )

        await service.process_extraction(
            entities=[sarah, alice],
            explicit_relationships=[{
                "source_name": "sarah",
                "target_name": "alice",
                "relationship_type": "knows",
                "confidence": 0.7,
            }],
            user_id=user_id,
        )

        stats = await service.get_stats()
        assert stats["total_relationships"] == 2
        assert stats["active_relationships"] == 2
        assert stats["average_strength"] > 0
        assert "works_with" in stats["relationship_type_counts"]
        assert "knows" in stats["relationship_type_counts"]


# ── Detection Result Model ───────────────────────────────────────────


class TestDetectionResult:
    def test_total_relationships(self):
        result = RelationshipDetectionResult()
        assert result.total_relationships == 0

        result.explicit_relationships = [
            RelationshipEdge(user_id="u", source_entity_id="a", target_entity_id="b",
                           relationship_type=RelationshipType.KNOWS),
        ]
        result.implicit_relationships = [
            RelationshipEdge(user_id="u", source_entity_id="c", target_entity_id="d",
                           relationship_type=RelationshipType.RELATED_TO),
        ]
        assert result.total_relationships == 2

    async def test_process_extraction_returns_latency(self, service, embedding_provider, user_id):
        sarah = await _make_entity(embedding_provider, "Sarah", EntityType.PERSON, user_id)
        bob = await _make_entity(embedding_provider, "Bob", EntityType.PERSON, user_id)

        result = await service.process_extraction(
            entities=[sarah, bob],
            user_id=user_id,
        )

        assert result.latency_ms >= 0
        assert result.entities_processed == 2


# ── Config Modes ─────────────────────────────────────────────────────


class TestConfigModes:
    def test_balanced_defaults(self):
        cfg = RelationshipConfig(mode=RelationshipDetectionMode.BALANCED)
        assert cfg.implicit_similarity_threshold == 0.65
        assert cfg.max_implicit_per_batch == 10

    def test_conservative_raises_thresholds(self):
        cfg = RelationshipConfig(mode=RelationshipDetectionMode.CONSERVATIVE)
        assert cfg.implicit_similarity_threshold == 0.80
        assert cfg.min_mentions_for_implicit == 3
        assert cfg.max_implicit_per_batch == 5

    def test_aggressive_lowers_thresholds(self):
        cfg = RelationshipConfig(mode=RelationshipDetectionMode.AGGRESSIVE)
        assert cfg.implicit_similarity_threshold == 0.50
        assert cfg.min_mentions_for_implicit == 1
        assert cfg.max_implicit_per_batch == 20

    def test_weight_sum_is_one(self):
        cfg = RelationshipConfig()
        total = cfg.embedding_weight + cfg.co_mention_weight + cfg.recency_weight
        assert abs(total - 1.0) < 1e-6
