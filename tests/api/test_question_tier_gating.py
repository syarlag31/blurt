"""API-level tests for QUESTION intent tier-based access control.

Validates the FastAPI endpoints enforce tier gating:
- Free tier gets structured queries, no premium fields
- Premium tier gets full recall with source episodes
- Recall-style queries from free users return upgrade prompts
- Upgrade prompts follow anti-shame design
- /types endpoint returns correct types per tier
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from blurt.api.question import set_question_service
from blurt.clients.embeddings import EmbeddingProvider
from blurt.core.app import create_app
from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import EntityType, FactType, PatternType
from blurt.services.question import QuestionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for API tests."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}
        self._counter = 0

    async def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            base = [0.0] * 64
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
    provider = FakeEmbeddingProvider()
    store = SemanticMemoryStore(user_id="test-user", embedding_provider=provider)
    await store.add_entity("Sarah", EntityType.PERSON, attributes={"role": "manager"})
    await store.add_entity("Alpha Project", EntityType.PROJECT)
    await store.add_fact(
        "Sarah manages the Alpha Project", FactType.ATTRIBUTE, source_blurt_id="blurt-1"
    )
    await store.add_fact("I prefer working in the mornings", FactType.PREFERENCE)
    await store.add_pattern(
        PatternType.TIME_OF_DAY,
        "Most productive between 9-11am",
        confidence=0.85,
        observation_count=10,
    )
    return store


@pytest_asyncio.fixture
async def client(semantic_store):
    app = create_app()
    service = QuestionService(semantic_store=semantic_store)
    set_question_service(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_question_service(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# POST /api/v1/question — free tier
# ---------------------------------------------------------------------------


class TestQuestionAPIFreeTier:
    """Free-tier users get structured queries only, no premium data."""

    @pytest.mark.asyncio
    async def test_entity_lookup_succeeds(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "Who is Sarah?",
            "tier": "free",
            "entity_name": "Sarah",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "free"
        assert body["result_count"] >= 1
        assert body["source_episodes"] == []
        assert body["confidence_scores"] == []
        assert body["relationship_context"] == []

    @pytest.mark.asyncio
    async def test_fact_lookup_succeeds(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What is Sarah's role?",
            "tier": "free",
            "query_type": "fact_lookup",
            "entity_name": "Sarah",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["query_type"] == "fact_lookup"
        assert body["tier"] == "free"

    @pytest.mark.asyncio
    async def test_count_query_succeeds(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "How many entities do I have?",
            "tier": "free",
            "query_type": "count_query",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["query_type"] == "count_query"
        assert body["result_count"] >= 1

    @pytest.mark.asyncio
    async def test_free_tier_results_capped(self, client: AsyncClient):
        """Free-tier results are capped at 10 regardless of max_results."""
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "everything",
            "tier": "free",
            "query_type": "entity_lookup",
            "max_results": 200,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result_count"] <= 10


# ---------------------------------------------------------------------------
# POST /api/v1/question — free tier recall attempts → upgrade prompts
# ---------------------------------------------------------------------------


class TestQuestionAPIFreeTierRecallGating:
    """Free-tier recall-style queries return upgrade prompts with fallback results."""

    @pytest.mark.asyncio
    async def test_semantic_recall_returns_upgrade_hint(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What did I say about the project last week?",
            "tier": "free",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["upgrade_hint"] is not None
        assert "Premium" in body["upgrade_hint"]
        assert body["tier"] == "free"

    @pytest.mark.asyncio
    async def test_pattern_query_returns_upgrade_hint(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "When am I most productive?",
            "tier": "free",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["upgrade_hint"] is not None
        assert body["query_type"] == "pattern_query"

    @pytest.mark.asyncio
    async def test_graph_query_returns_upgrade_hint(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "How are Sarah and Alpha Project connected?",
            "tier": "free",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["upgrade_hint"] is not None

    @pytest.mark.asyncio
    async def test_temporal_recall_returns_upgrade_hint(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What was I thinking about last month?",
            "tier": "free",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["upgrade_hint"] is not None
        assert body["query_type"] == "temporal_recall"

    @pytest.mark.asyncio
    async def test_neighborhood_returns_upgrade_hint(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "Everything about Alpha Project",
            "tier": "free",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["upgrade_hint"] is not None

    @pytest.mark.asyncio
    async def test_upgrade_hint_is_anti_shame(self, client: AsyncClient):
        """Upgrade hints must never contain shame language."""
        shame_words = [
            "overdue", "behind", "missed", "failed", "lazy", "guilty",
            "you should have", "you must", "hurry", "running out",
            "deadline", "penalty", "streak", "slacking",
        ]
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What did I say about the project last week?",
            "tier": "free",
        })
        body = resp.json()
        hint = body.get("upgrade_hint", "")
        hint_lower = hint.lower()
        for word in shame_words:
            assert word not in hint_lower, (
                f"Shame word '{word}' found in upgrade hint: {hint}"
            )

    @pytest.mark.asyncio
    async def test_gated_query_still_returns_results(self, client: AsyncClient):
        """Even when gated, the response should contain fallback results, not
        an empty 'upgrade to see' wall."""
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What did I say about the project last week?",
            "tier": "free",
        })
        body = resp.json()
        # answer_summary is always populated — never an empty "upgrade" wall
        assert body["answer_summary"] != ""


# ---------------------------------------------------------------------------
# POST /api/v1/question — premium tier
# ---------------------------------------------------------------------------


class TestQuestionAPIPremiumTier:
    """Premium tier gets full recall with all response fields."""

    @pytest.mark.asyncio
    async def test_semantic_recall_no_upgrade_hint(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What do I know about Sarah?",
            "tier": "premium",
            "query_type": "semantic_recall",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "premium"
        assert body["upgrade_hint"] is None

    @pytest.mark.asyncio
    async def test_pattern_query_works(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "When am I most productive?",
            "tier": "premium",
            "query_type": "pattern_query",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "premium"
        assert body["query_type"] == "pattern_query"
        assert body["upgrade_hint"] is None

    @pytest.mark.asyncio
    async def test_premium_includes_confidence_scores(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "Sarah",
            "tier": "premium",
            "query_type": "semantic_recall",
        })
        body = resp.json()
        if body["result_count"] > 0:
            assert len(body["confidence_scores"]) > 0

    @pytest.mark.asyncio
    async def test_premium_result_limit_generous(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "everything",
            "tier": "premium",
            "query_type": "semantic_recall",
            "max_results": 100,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result_count"] <= 100


# ---------------------------------------------------------------------------
# POST /api/v1/question — team tier
# ---------------------------------------------------------------------------


class TestQuestionAPITeamTier:
    @pytest.mark.asyncio
    async def test_team_has_full_access(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What did I say about Sarah?",
            "tier": "team",
            "query_type": "semantic_recall",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "team"
        assert body["upgrade_hint"] is None


# ---------------------------------------------------------------------------
# GET /api/v1/question/types
# ---------------------------------------------------------------------------


class TestQueryTypesEndpoint:
    """The /types endpoint should list available query types per tier."""

    @pytest.mark.asyncio
    async def test_free_tier_types(self, client: AsyncClient):
        resp = await client.get("/api/v1/question/types?tier=free")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "free"
        type_names = {t["type"] for t in body["available_types"]}
        assert "entity_lookup" in type_names
        assert "fact_lookup" in type_names
        assert "recent_facts" in type_names
        assert "count_query" in type_names
        # Premium-only types should NOT be listed
        assert "semantic_recall" not in type_names
        assert "graph_query" not in type_names
        assert "pattern_query" not in type_names

    @pytest.mark.asyncio
    async def test_premium_tier_types(self, client: AsyncClient):
        resp = await client.get("/api/v1/question/types?tier=premium")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "premium"
        type_names = {t["type"] for t in body["available_types"]}
        # Premium has all types
        assert "semantic_recall" in type_names
        assert "graph_query" in type_names
        assert "pattern_query" in type_names
        assert "entity_lookup" in type_names

    @pytest.mark.asyncio
    async def test_team_tier_types(self, client: AsyncClient):
        resp = await client.get("/api/v1/question/types?tier=team")
        assert resp.status_code == 200
        body = resp.json()
        type_names = {t["type"] for t in body["available_types"]}
        assert "semantic_recall" in type_names
        assert body["total_types"] >= 9

    @pytest.mark.asyncio
    async def test_free_tier_has_fewer_types(self, client: AsyncClient):
        free_resp = await client.get("/api/v1/question/types?tier=free")
        premium_resp = await client.get("/api/v1/question/types?tier=premium")
        assert free_resp.json()["total_types"] < premium_resp.json()["total_types"]


# ---------------------------------------------------------------------------
# Auto-detection via API
# ---------------------------------------------------------------------------


class TestQuestionAutoDetection:
    """Query type auto-detection works through the API."""

    @pytest.mark.asyncio
    async def test_who_is_detected_as_entity_lookup(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "Who is Sarah?",
            "tier": "free",
        })
        assert resp.json()["query_type"] == "entity_lookup"

    @pytest.mark.asyncio
    async def test_how_many_detected_as_count(self, client: AsyncClient):
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "How many tasks do I have?",
            "tier": "free",
        })
        assert resp.json()["query_type"] == "count_query"

    @pytest.mark.asyncio
    async def test_recall_query_detected_and_gated(self, client: AsyncClient):
        """A recall-style query auto-detected as semantic_recall is gated
        for free users but still returns the detected type in the response."""
        resp = await client.post("/api/v1/question", json={
            "user_id": "u1",
            "query": "What did I say about the project?",
            "tier": "free",
        })
        body = resp.json()
        assert body["query_type"] == "semantic_recall"
        assert body["upgrade_hint"] is not None
        assert body["tier"] == "free"
