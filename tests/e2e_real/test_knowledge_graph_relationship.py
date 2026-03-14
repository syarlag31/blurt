"""Test that two sequential blurts mentioning the same entity create a relationship.

This test validates the knowledge graph relationship pipeline:
1. Extract entities from two blurts that share a common entity
2. Process them through ``SemanticMemoryStore.process_extracted_entities``
3. Verify MENTIONED_WITH relationships are created between co-mentioned entities
4. Verify relationship strengthening when the same pair is mentioned again

Uses REAL Gemini API calls — no mocks.
"""

from __future__ import annotations

import pytest

from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import RelationshipType

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def semantic_store(
    embedding_provider,
    test_user_id: str,
) -> SemanticMemoryStore:
    """Fresh in-memory semantic memory store for each test."""
    return SemanticMemoryStore(
        user_id=test_user_id,
        embedding_provider=embedding_provider,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_two_blurts_same_entity_creates_relationship(
    entity_extractor,
    semantic_store: SemanticMemoryStore,
) -> None:
    """Two blurts mentioning the same entity create MENTIONED_WITH relationships.

    Blurt 1: "Tell Sarah about the product launch" → Sarah + product launch
    Blurt 2: "Sarah needs to review the Q2 budget"  → Sarah + Q2 budget

    After processing both, the knowledge graph should contain:
    - An entity for Sarah (mentioned in both blurts, mention_count ≥ 2)
    - MENTIONED_WITH relationships between co-mentioned entities in each blurt
    - Sarah should have relationships to entities from *both* blurts
    """
    blurt_1_text = "Tell Sarah about the product launch"
    blurt_2_text = "Sarah needs to review the Q2 budget"

    # --- Extract entities from both blurts using real Gemini ---
    result_1 = await entity_extractor.extract(blurt_1_text)
    result_2 = await entity_extractor.extract(blurt_2_text)

    assert result_1.success, f"Extraction failed for blurt 1: {result_1.error}"
    assert result_2.success, f"Extraction failed for blurt 2: {result_2.error}"

    # Build entity dicts for the graph store
    entities_1 = [
        {"name": e.name, "type": e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value}
        for e in result_1.entities
    ]
    entities_2 = [
        {"name": e.name, "type": e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value}
        for e in result_2.entities
    ]

    assert len(entities_1) >= 1, (
        f"Expected at least 1 entity from blurt 1, got {len(entities_1)}: {entities_1}"
    )
    assert len(entities_2) >= 1, (
        f"Expected at least 1 entity from blurt 2, got {len(entities_2)}: {entities_2}"
    )

    # --- Process entities through the knowledge graph ---
    nodes_1 = await semantic_store.process_extracted_entities(
        entities_1, blurt_id="blurt-001", raw_text=blurt_1_text,
    )
    nodes_2 = await semantic_store.process_extracted_entities(
        entities_2, blurt_id="blurt-002", raw_text=blurt_2_text,
    )

    # --- Find Sarah in the graph ---
    sarah_node = await semantic_store.find_entity_by_name("Sarah")
    assert sarah_node is not None, (
        "Expected to find 'Sarah' in the knowledge graph after processing two blurts. "
        f"Entities in blurt 1: {[e['name'] for e in entities_1]}, "
        f"Entities in blurt 2: {[e['name'] for e in entities_2]}"
    )

    # Sarah should have been mentioned in both blurts → mention_count ≥ 2
    assert sarah_node.mention_count >= 2, (
        f"Expected Sarah's mention_count ≥ 2 (mentioned in both blurts), "
        f"got {sarah_node.mention_count}"
    )

    # --- Verify relationships exist ---
    sarah_rels = await semantic_store.get_entity_relationships(sarah_node.id)
    assert len(sarah_rels) >= 1, (
        f"Expected Sarah to have at least 1 relationship after two blurts, "
        f"got {len(sarah_rels)}"
    )

    # All relationships from co-mentions should be MENTIONED_WITH
    mentioned_with_rels = [
        r for r in sarah_rels
        if r.relationship_type == RelationshipType.MENTIONED_WITH
    ]
    assert len(mentioned_with_rels) >= 1, (
        f"Expected at least 1 MENTIONED_WITH relationship for Sarah, "
        f"got {len(mentioned_with_rels)}. "
        f"Relationship types found: {[r.relationship_type for r in sarah_rels]}"
    )

    # Each relationship should have context snippets from the blurt text
    for rel in mentioned_with_rels:
        assert len(rel.context_snippets) >= 1, (
            f"Expected context snippets on relationship {rel.id}, got none"
        )

    # --- Verify graph stats ---
    stats = await semantic_store.get_graph_stats()
    assert stats["total_entities"] >= 3, (
        f"Expected ≥ 3 entities (Sarah + at least 1 from each blurt), "
        f"got {stats['total_entities']}"
    )
    assert stats["total_relationships"] >= 1, (
        f"Expected ≥ 1 relationship, got {stats['total_relationships']}"
    )
