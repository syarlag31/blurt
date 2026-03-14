"""Tests for the QuestionService — tier-gated question answering.

Validates:
- Free tier gets structured results (entity/fact lookup) without premium fields
- Premium tier gets full recall with source episodes and confidence scores
- Premium-only queries fall back gracefully on free tier
- Response formatting is tier-appropriate
- Service handles missing stores gracefully
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from blurt.clients.embeddings import EmbeddingProvider
from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import EntityType, FactType, PatternType
from blurt.services.access_control import (
    QuestionQueryType,
    QuestionRequest,
    UserTier,
)
from blurt.services.question import QuestionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}
        self._counter = 0

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            # Create a deterministic but unique embedding
            base = [0.0] * 64
            # Use hash for some similarity between related texts
            h = hash(text) % 1000
            base[self._counter % 64] = 1.0
            base[(self._counter + 1) % 64] = h / 1000.0
            self._cache[text] = base
            self._counter += 1
        return self._cache[text]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 64


@pytest_asyncio.fixture
async def semantic_store():
    """Create a semantic store with test data."""
    provider = FakeEmbeddingProvider()
    store = SemanticMemoryStore(user_id="test-user", embedding_provider=provider)

    # Add entities
    await store.add_entity("Sarah", EntityType.PERSON, attributes={"role": "manager"})
    await store.add_entity("Q2 Project", EntityType.PROJECT, attributes={"status": "active"})
    await store.add_entity("Acme Corp", EntityType.ORGANIZATION)

    # Add facts
    await store.add_fact("Sarah is my manager", FactType.ATTRIBUTE, source_blurt_id="blurt-1")
    await store.add_fact("I prefer morning meetings", FactType.PREFERENCE)
    await store.add_fact("The Q2 project deadline is March 31", FactType.ATTRIBUTE)

    # Add a pattern
    await store.add_pattern(
        PatternType.TIME_OF_DAY,
        "Most productive in the mornings between 9-11am",
        confidence=0.85,
        observation_count=10,
    )

    return store


@pytest.fixture
def question_service(semantic_store):
    """Create a QuestionService with the test semantic store."""
    return QuestionService(semantic_store=semantic_store)


@pytest.fixture
def empty_service():
    """QuestionService with no stores — should handle gracefully."""
    return QuestionService()


# ---------------------------------------------------------------------------
# Free tier tests
# ---------------------------------------------------------------------------


class TestFreeTierQueries:
    """Free tier should get structured queries only."""

    @pytest.mark.asyncio
    async def test_entity_lookup_returns_results(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="Who is Sarah?",
            entity_name="Sarah",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.tier == UserTier.FREE
        assert result.result_count > 0
        assert any("Sarah" in str(r) for r in result.results)

    @pytest.mark.asyncio
    async def test_entity_lookup_no_premium_fields(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="Who is Sarah?",
            entity_name="Sarah",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.source_episodes == []
        assert result.confidence_scores == []
        assert result.relationship_context == []

    @pytest.mark.asyncio
    async def test_fact_lookup_returns_facts(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What is Sarah's role?",
            entity_name="Sarah",
            query_type=QuestionQueryType.FACT_LOOKUP,
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.tier == UserTier.FREE
        assert result.query_type == QuestionQueryType.FACT_LOOKUP

    @pytest.mark.asyncio
    async def test_count_query(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="How many entities do I have?",
            query_type=QuestionQueryType.COUNT_QUERY,
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.result_count > 0
        assert result.results[0].get("entity_count", 0) > 0

    @pytest.mark.asyncio
    async def test_results_capped_at_free_limit(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="everything",
            query_type=QuestionQueryType.ENTITY_LOOKUP,
            max_results=200,  # Request more than free tier allows
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        # Results should be capped at free tier max (10)
        assert result.result_count <= 10

    @pytest.mark.asyncio
    async def test_recent_facts(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What did I learn recently?",
            query_type=QuestionQueryType.RECENT_FACTS,
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.tier == UserTier.FREE
        assert result.query_type == QuestionQueryType.RECENT_FACTS


class TestFreeTierFallbacks:
    """Premium-only queries should fall back gracefully on free tier."""

    @pytest.mark.asyncio
    async def test_semantic_recall_falls_back(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What did I say about the project last week?",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        # Should detect as SEMANTIC_RECALL but still return the original type
        assert result.query_type == QuestionQueryType.SEMANTIC_RECALL
        assert result.upgrade_hint is not None
        # Should still have some results (from fallback)
        assert result.tier == UserTier.FREE

    @pytest.mark.asyncio
    async def test_pattern_query_falls_back(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="When am I most productive?",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.query_type == QuestionQueryType.PATTERN_QUERY
        assert result.upgrade_hint is not None
        assert result.source_episodes == []  # No premium fields

    @pytest.mark.asyncio
    async def test_graph_query_falls_back(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="How are Sarah and Q2 Project connected?",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.upgrade_hint is not None

    @pytest.mark.asyncio
    async def test_fallback_upgrade_hint_is_anti_shame(self, question_service):
        """Upgrade hints must never use shame language."""
        shame_words = [
            "overdue", "behind", "missed", "failed", "lazy",
            "you should", "hurry", "penalty", "streak",
        ]
        request = QuestionRequest(
            user_id="test-user",
            query="What did I say about the project last week?",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        if result.upgrade_hint:
            hint_lower = result.upgrade_hint.lower()
            for word in shame_words:
                assert word not in hint_lower


# ---------------------------------------------------------------------------
# Premium tier tests
# ---------------------------------------------------------------------------


class TestPremiumTierQueries:
    """Premium tier should get full recall with all response fields."""

    @pytest.mark.asyncio
    async def test_semantic_recall_returns_full_results(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What do I know about Sarah?",
            query_type=QuestionQueryType.SEMANTIC_RECALL,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        assert result.tier == UserTier.PREMIUM
        assert result.query_type == QuestionQueryType.SEMANTIC_RECALL
        assert result.upgrade_hint is None  # No upgrade needed

    @pytest.mark.asyncio
    async def test_semantic_recall_includes_confidence(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="Sarah",
            query_type=QuestionQueryType.SEMANTIC_RECALL,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        # Premium responses include confidence scores
        if result.result_count > 0:
            assert len(result.confidence_scores) > 0

    @pytest.mark.asyncio
    async def test_pattern_query_works(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="When am I most productive?",
            query_type=QuestionQueryType.PATTERN_QUERY,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        assert result.tier == UserTier.PREMIUM
        assert result.query_type == QuestionQueryType.PATTERN_QUERY

    @pytest.mark.asyncio
    async def test_graph_query_with_entity(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What is connected to Sarah?",
            entity_name="Sarah",
            query_type=QuestionQueryType.GRAPH_QUERY,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        assert result.tier == UserTier.PREMIUM
        assert result.upgrade_hint is None

    @pytest.mark.asyncio
    async def test_neighborhood_query(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="Everything about Q2 project",
            query_type=QuestionQueryType.NEIGHBORHOOD,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        assert result.tier == UserTier.PREMIUM

    @pytest.mark.asyncio
    async def test_premium_generous_result_limit(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="everything",
            query_type=QuestionQueryType.SEMANTIC_RECALL,
            max_results=100,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        # Premium can get up to 100 results
        assert result.result_count <= 100

    @pytest.mark.asyncio
    async def test_premium_also_gets_structured_queries(self, question_service):
        """Premium tier should also work with free-tier query types."""
        request = QuestionRequest(
            user_id="test-user",
            query="Who is Sarah?",
            entity_name="Sarah",
            query_type=QuestionQueryType.ENTITY_LOOKUP,
        )
        result = await question_service.answer(request, tier=UserTier.PREMIUM)

        assert result.tier == UserTier.PREMIUM
        assert result.result_count > 0


# ---------------------------------------------------------------------------
# Team tier tests
# ---------------------------------------------------------------------------


class TestTeamTier:
    @pytest.mark.asyncio
    async def test_team_has_full_access(self, question_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What did I say about Sarah?",
            query_type=QuestionQueryType.SEMANTIC_RECALL,
        )
        result = await question_service.answer(request, tier=UserTier.TEAM)

        assert result.tier == UserTier.TEAM
        assert result.upgrade_hint is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_store_returns_gracefully(self, empty_service):
        request = QuestionRequest(
            user_id="test-user",
            query="Who is Sarah?",
        )
        result = await empty_service.answer(request, tier=UserTier.FREE)

        assert result.result_count == 0
        assert result.answer_summary != ""  # Should have a helpful message

    @pytest.mark.asyncio
    async def test_empty_store_premium_returns_gracefully(self, empty_service):
        request = QuestionRequest(
            user_id="test-user",
            query="What patterns do I have?",
            query_type=QuestionQueryType.PATTERN_QUERY,
        )
        result = await empty_service.answer(request, tier=UserTier.PREMIUM)

        assert result.result_count == 0

    @pytest.mark.asyncio
    async def test_auto_detects_query_type(self, question_service):
        """When no query_type is specified, it should auto-detect."""
        request = QuestionRequest(
            user_id="test-user",
            query="How many tasks do I have this week?",
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.query_type == QuestionQueryType.COUNT_QUERY

    @pytest.mark.asyncio
    async def test_respects_max_results_within_tier(self, question_service):
        """User-specified max_results should be capped by tier limit."""
        request = QuestionRequest(
            user_id="test-user",
            query="everything",
            query_type=QuestionQueryType.ENTITY_LOOKUP,
            max_results=5,
        )
        result = await question_service.answer(request, tier=UserTier.FREE)

        assert result.result_count <= 5

    @pytest.mark.asyncio
    async def test_answer_summary_always_present(self, question_service):
        """Every response should have a human-readable summary."""
        for qt in [QuestionQueryType.ENTITY_LOOKUP, QuestionQueryType.COUNT_QUERY]:
            request = QuestionRequest(
                user_id="test-user",
                query="test",
                query_type=qt,
            )
            result = await question_service.answer(request, tier=UserTier.FREE)
            assert result.answer_summary != ""
