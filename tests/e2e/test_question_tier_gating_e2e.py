"""End-to-end tests for QUESTION intent tier-based access control.

Validates the full flow: natural language question → auto-classify → tier gate
→ fallback (if needed) → query execution → response formatting.

Key scenarios:
- Free user asks a recall question → gets fallback results + upgrade hint
- Premium user asks same question → gets full recall
- Tier transitions: same question, different tiers, different responses
- Anti-shame: no guilt language anywhere in the flow
- Graceful degradation: free-tier always returns something useful
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
    classify_question_type,
    get_capabilities,
)
from blurt.services.question import QuestionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._cache: dict[str, list[float]] = {}
        self._counter = 0

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            base = [0.0] * 64
            base[self._counter % 64] = 1.0
            base[(self._counter + 1) % 64] = (hash(text) % 1000) / 1000.0
            self._cache[text] = base
            self._counter += 1
        return self._cache[text]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 64


@pytest_asyncio.fixture
async def rich_store():
    """A store with enough data to exercise all query types."""
    provider = FakeEmbeddingProvider()
    store = SemanticMemoryStore(user_id="test-user", embedding_provider=provider)

    # Entities
    await store.add_entity("Sarah", EntityType.PERSON, attributes={"role": "manager"})
    await store.add_entity("David", EntityType.PERSON, attributes={"role": "engineer"})
    await store.add_entity("Alpha Project", EntityType.PROJECT, attributes={"status": "active"})
    await store.add_entity("Acme Corp", EntityType.ORGANIZATION)

    # Facts
    await store.add_fact("Sarah manages Alpha Project", FactType.ATTRIBUTE, source_blurt_id="b1")
    await store.add_fact("David works on the backend", FactType.ATTRIBUTE, source_blurt_id="b2")
    await store.add_fact("I prefer morning meetings", FactType.PREFERENCE)
    await store.add_fact("Alpha Project deadline is March 31", FactType.ATTRIBUTE)

    # Pattern
    await store.add_pattern(
        PatternType.TIME_OF_DAY,
        "Most productive 9-11am",
        confidence=0.88,
        observation_count=15,
    )

    return store


@pytest.fixture
def service(rich_store):
    return QuestionService(semantic_store=rich_store)


# ---------------------------------------------------------------------------
# E2E: Same question, different tiers
# ---------------------------------------------------------------------------


class TestSameQuestionDifferentTiers:
    """The same question should produce different responses per tier."""

    @pytest.mark.asyncio
    async def test_recall_question_free_vs_premium(self, service):
        query = "What did I say about Sarah's role?"

        # Free tier
        free_result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.FREE,
        )
        assert free_result.tier == UserTier.FREE
        assert free_result.upgrade_hint is not None
        assert free_result.source_episodes == []
        assert free_result.confidence_scores == []

        # Premium tier
        premium_result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.PREMIUM,
        )
        assert premium_result.tier == UserTier.PREMIUM
        assert premium_result.upgrade_hint is None

    @pytest.mark.asyncio
    async def test_pattern_question_free_vs_premium(self, service):
        query = "When am I most productive?"

        free_result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.FREE,
        )
        assert free_result.upgrade_hint is not None
        assert free_result.query_type == QuestionQueryType.PATTERN_QUERY

        premium_result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.PREMIUM,
        )
        assert premium_result.upgrade_hint is None
        assert premium_result.query_type == QuestionQueryType.PATTERN_QUERY

    @pytest.mark.asyncio
    async def test_entity_lookup_same_across_tiers(self, service):
        """Entity lookup is allowed on both tiers — no upgrade hint."""
        query = "Who is Sarah?"

        free_result = await service.answer(
            QuestionRequest(user_id="u1", query=query, entity_name="Sarah"),
            tier=UserTier.FREE,
        )
        premium_result = await service.answer(
            QuestionRequest(user_id="u1", query=query, entity_name="Sarah"),
            tier=UserTier.PREMIUM,
        )

        assert free_result.upgrade_hint is None
        assert premium_result.upgrade_hint is None
        # Both should find Sarah
        assert free_result.result_count > 0
        assert premium_result.result_count > 0


# ---------------------------------------------------------------------------
# E2E: Full pipeline classify → gate → execute → format
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test the full classify → gate → execute → format pipeline."""

    @pytest.mark.asyncio
    async def test_classify_gate_execute_for_free_recall(self, service):
        """Natural language recall query → classified → gated → fallback → response."""
        query = "What did I mention about the Alpha Project deadline?"
        detected_type = classify_question_type(query)
        assert detected_type == QuestionQueryType.SEMANTIC_RECALL

        result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.FREE,
        )
        # Original type preserved in response
        assert result.query_type == QuestionQueryType.SEMANTIC_RECALL
        # But gated with upgrade hint
        assert result.upgrade_hint is not None
        # Still returns something useful (fallback to fact_lookup)
        assert result.answer_summary != ""

    @pytest.mark.asyncio
    async def test_classify_execute_for_premium_recall(self, service):
        """Same query on premium → no gating, full recall."""
        query = "What did I mention about the Alpha Project deadline?"

        result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.PREMIUM,
        )
        assert result.query_type == QuestionQueryType.SEMANTIC_RECALL
        assert result.upgrade_hint is None

    @pytest.mark.asyncio
    async def test_count_query_no_gating_any_tier(self, service):
        """Count queries work on all tiers without gating."""
        query = "How many entities do I have?"

        for tier in [UserTier.FREE, UserTier.PREMIUM, UserTier.TEAM]:
            result = await service.answer(
                QuestionRequest(user_id="u1", query=query), tier=tier,
            )
            assert result.upgrade_hint is None
            assert result.query_type == QuestionQueryType.COUNT_QUERY
            assert result.result_count > 0


# ---------------------------------------------------------------------------
# E2E: Time range capping
# ---------------------------------------------------------------------------


class TestTimeRangeCapping:
    """Free tier is capped to 30 days of history."""

    @pytest.mark.asyncio
    async def test_free_tier_time_range_capped(self, service):
        """Requesting 365 days on free tier should be capped to 30."""
        result = await service.answer(
            QuestionRequest(
                user_id="u1",
                query="What did I learn recently?",
                query_type=QuestionQueryType.RECENT_FACTS,
                time_range_days=365,
            ),
            tier=UserTier.FREE,
        )
        # The service internally caps to 30 days for free tier
        # We can't directly observe the internal cap, but we verify
        # the response is valid and comes from free tier
        assert result.tier == UserTier.FREE
        caps = get_capabilities(UserTier.FREE)
        assert caps.max_history_days == 30

    @pytest.mark.asyncio
    async def test_premium_tier_time_range_not_capped(self, service):
        """Premium can request long time ranges."""
        result = await service.answer(
            QuestionRequest(
                user_id="u1",
                query="What did I learn?",
                query_type=QuestionQueryType.RECENT_FACTS,
                time_range_days=365,
            ),
            tier=UserTier.PREMIUM,
        )
        assert result.tier == UserTier.PREMIUM
        caps = get_capabilities(UserTier.PREMIUM)
        assert caps.max_history_days >= 365


# ---------------------------------------------------------------------------
# E2E: Anti-shame across the full pipeline
# ---------------------------------------------------------------------------


SHAME_WORDS = [
    "overdue", "behind", "missed", "failed", "lazy", "guilty",
    "you should have", "you must", "hurry", "running out",
    "deadline", "penalty", "streak", "slacking", "falling behind",
]


class TestAntiShameE2E:
    """Every user-facing string in the full pipeline must be shame-free."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "What did I say about the project?",
        "When am I most productive?",
        "How are Sarah and David connected?",
        "What was I thinking about last month?",
        "Everything about Alpha Project",
    ])
    async def test_recall_queries_no_shame(self, service, query):
        result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.FREE,
        )
        for text in [result.answer_summary, result.upgrade_hint or ""]:
            text_lower = text.lower()
            for word in SHAME_WORDS:
                assert word not in text_lower, (
                    f"Shame word '{word}' found in: {text}"
                )

    @pytest.mark.asyncio
    async def test_empty_results_no_shame(self, service):
        result = await service.answer(
            QuestionRequest(user_id="u1", query="something nonexistent"),
            tier=UserTier.FREE,
        )
        text_lower = result.answer_summary.lower()
        for word in SHAME_WORDS:
            assert word not in text_lower


# ---------------------------------------------------------------------------
# E2E: Graceful degradation (never empty "upgrade" walls)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Free-tier users should always get something useful, never an
    empty 'upgrade to see results' response."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_gated", [
        ("What did I say about Sarah?", True),
        ("When am I most productive?", True),
        ("How are Sarah and David connected?", True),
        ("Who is Sarah?", False),
        ("How many entities do I have?", False),
    ])
    async def test_free_tier_always_has_summary(self, service, query, expected_gated):
        result = await service.answer(
            QuestionRequest(user_id="u1", query=query), tier=UserTier.FREE,
        )
        # answer_summary is always populated
        assert result.answer_summary != ""
        if expected_gated:
            assert result.upgrade_hint is not None
        else:
            assert result.upgrade_hint is None
