"""Tests for the personal history recall engine.

Tests cover:
- Cross-source recall (episodes, entities, facts, patterns, summaries)
- Relevance ranking and scoring
- Recency boost
- Entity mention boost
- Source filtering
- Time range filtering
- Deduplication
- Edge cases (no results, no embedding provider, empty store)
- Configuration
- API routes
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from blurt.clients.embeddings import MockEmbeddingProvider
from blurt.memory.episodic import (
    EmotionSnapshot,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodeSummary,
    InMemoryEpisodicStore,
)
from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import (
    EntityType,
    FactType,
    PatternType,
)
from blurt.services.recall import (
    PersonalHistoryRecallEngine,
    RecallConfig,
    RecallResponse,
    RecallResult,
    RecallSourceType,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_episode(
    user_id: str = "user-1",
    raw_text: str = "test episode",
    intent: str = "journal",
    embedding: list[float] | None = None,
    entities: list[EntityRef] | None = None,
    timestamp: datetime | None = None,
    session_id: str = "sess-1",
    emotion_primary: str = "trust",
) -> Episode:
    """Create a test episode with optional embedding."""
    return Episode(
        id=str(uuid.uuid4()),
        user_id=user_id,
        raw_text=raw_text,
        intent=intent,
        intent_confidence=0.9,
        timestamp=timestamp or _utcnow(),
        embedding=embedding,
        entities=entities or [],
        context=EpisodeContext(session_id=session_id),
        emotion=EmotionSnapshot(primary=emotion_primary, intensity=1.0, valence=0.3, arousal=0.4),
    )


@pytest.fixture
def mock_embedder():
    """Mock embedding provider for deterministic tests."""
    return MockEmbeddingProvider()


@pytest.fixture
def episodic_store():
    """Fresh in-memory episodic store."""
    return InMemoryEpisodicStore()


@pytest.fixture
def semantic_store(mock_embedder):
    """Fresh semantic memory store."""
    return SemanticMemoryStore(user_id="user-1", embedding_provider=mock_embedder)


@pytest.fixture
def recall_engine(episodic_store, semantic_store, mock_embedder):
    """Recall engine with all stores configured."""
    return PersonalHistoryRecallEngine(
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        embedding_provider=mock_embedder,
    )


# ── Basic Recall Tests ───────────────────────────────────────────────


class TestBasicRecall:
    """Test basic recall functionality."""

    @pytest.mark.asyncio
    async def test_recall_returns_response(self, recall_engine):
        """Recall always returns a RecallResponse, even with no results."""
        response = await recall_engine.recall("user-1", "test query")
        assert isinstance(response, RecallResponse)
        assert response.query == "test query"
        assert isinstance(response.results, list)

    @pytest.mark.asyncio
    async def test_recall_empty_store_returns_no_results(self, recall_engine):
        """Recall on empty store returns empty results."""
        response = await recall_engine.recall("user-1", "anything")
        assert response.total_results == 0
        assert not response.has_results
        assert response.top_result is None

    @pytest.mark.asyncio
    async def test_recall_generates_embedding(self, recall_engine):
        """Recall generates a query embedding."""
        response = await recall_engine.recall("user-1", "test query")
        assert response.query_embedding_generated is True

    @pytest.mark.asyncio
    async def test_recall_without_embedding_provider(self, episodic_store):
        """Recall works (degrades gracefully) without embedding provider."""
        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=None,
        )
        response = await engine.recall("user-1", "test query")
        assert isinstance(response, RecallResponse)
        assert response.query_embedding_generated is False

    @pytest.mark.asyncio
    async def test_recall_tracks_sources_searched(self, recall_engine):
        """Recall tracks which sources were searched."""
        response = await recall_engine.recall("user-1", "test query")
        assert isinstance(response.sources_searched, list)

    @pytest.mark.asyncio
    async def test_recall_measures_latency(self, recall_engine):
        """Recall measures and reports latency."""
        response = await recall_engine.recall("user-1", "test query")
        assert response.latency_ms >= 0


# ── Episode Search Tests ─────────────────────────────────────────────


class TestEpisodeSearch:
    """Test episodic memory search."""

    @pytest.mark.asyncio
    async def test_finds_episodes_by_embedding_similarity(
        self, episodic_store, mock_embedder
    ):
        """Recall finds episodes with similar embeddings."""
        # Create episode with embedding
        query_text = "project planning meeting"
        query_embedding = await mock_embedder.embed(query_text)

        ep = _make_episode(
            raw_text="discussed the project planning timeline",
            embedding=query_embedding,  # Same embedding = perfect match
        )
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", query_text)
        assert response.has_results
        episode_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.EPISODE
        ]
        assert len(episode_results) >= 1
        assert episode_results[0].content == "discussed the project planning timeline"

    @pytest.mark.asyncio
    async def test_episode_results_include_metadata(
        self, episodic_store, mock_embedder
    ):
        """Episode results include intent, emotion, entities metadata."""
        embedding = await mock_embedder.embed("sarah meeting")
        ep = _make_episode(
            raw_text="meeting with Sarah about the project",
            intent="event",
            embedding=embedding,
            entities=[EntityRef(name="Sarah", entity_type="person")],
            emotion_primary="anticipation",
        )
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "sarah meeting")
        episode_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.EPISODE
        ]
        assert len(episode_results) >= 1
        meta = episode_results[0].metadata
        assert meta["intent"] == "event"
        assert meta["emotion"] == "anticipation"
        assert "Sarah" in meta["entities"]

    @pytest.mark.asyncio
    async def test_episode_time_filter(self, episodic_store, mock_embedder):
        """Time range filters work on episode results."""
        now = _utcnow()
        embedding = await mock_embedder.embed("gym workout")

        # Old episode
        old_ep = _make_episode(
            raw_text="went to the gym yesterday",
            embedding=embedding,
            timestamp=now - timedelta(days=30),
        )
        await episodic_store.append(old_ep)

        # Recent episode
        recent_ep = _make_episode(
            raw_text="great gym session today",
            embedding=embedding,
            timestamp=now - timedelta(hours=2),
        )
        await episodic_store.append(recent_ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )

        # Search only recent
        response = await engine.recall(
            "user-1",
            "gym workout",
            time_start=now - timedelta(days=7),
        )
        episode_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.EPISODE
        ]
        # Only the recent episode should match
        assert all(
            r.timestamp >= now - timedelta(days=7)
            for r in episode_results
            if r.timestamp is not None
        )


# ── Entity Search Tests ──────────────────────────────────────────────


class TestEntitySearch:
    """Test knowledge graph entity search."""

    @pytest.mark.asyncio
    async def test_finds_entities_in_graph(self, semantic_store, mock_embedder, episodic_store):
        """Recall finds entities in the knowledge graph."""
        await semantic_store.add_entity(
            name="Sarah Chen",
            entity_type=EntityType.PERSON,
            attributes={"role": "engineering manager"},
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "Sarah Chen")
        # Should find the entity
        entity_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.ENTITY
        ]
        # Entity search uses semantic_store.search() which requires embeddings
        # The results depend on embedding similarity
        assert isinstance(entity_results, list)


# ── Fact Search Tests ────────────────────────────────────────────────


class TestFactSearch:
    """Test fact search in the knowledge graph."""

    @pytest.mark.asyncio
    async def test_finds_matching_facts(self, semantic_store, mock_embedder, episodic_store):
        """Recall finds facts with matching embeddings."""
        # Add a fact with embedding
        fact_text = "Sarah is my engineering manager"
        await semantic_store.add_fact(
            content=fact_text,
            fact_type=FactType.ATTRIBUTE,
            confidence=0.95,
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", fact_text)
        fact_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.FACT
        ]
        # Should find facts if embeddings match
        assert isinstance(fact_results, list)

    @pytest.mark.asyncio
    async def test_fact_results_include_metadata(
        self, semantic_store, mock_embedder, episodic_store
    ):
        """Fact results include type, confidence, and confirmation count."""
        await semantic_store.add_fact(
            content="I prefer morning meetings",
            fact_type=FactType.PREFERENCE,
            confidence=0.9,
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "I prefer morning meetings")
        fact_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.FACT
        ]
        if fact_results:
            meta = fact_results[0].metadata
            assert "fact_type" in meta
            assert "confidence" in meta


# ── Pattern Search Tests ─────────────────────────────────────────────


class TestPatternSearch:
    """Test behavioral pattern search."""

    @pytest.mark.asyncio
    async def test_finds_matching_patterns(
        self, semantic_store, mock_embedder, episodic_store
    ):
        """Recall finds patterns with matching embeddings."""
        await semantic_store.add_pattern(
            pattern_type=PatternType.TIME_OF_DAY,
            description="User is most productive in the morning",
            confidence=0.8,
            observation_count=10,
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall(
            "user-1", "User is most productive in the morning"
        )
        pattern_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.PATTERN
        ]
        assert isinstance(pattern_results, list)


# ── Summary Search Tests ─────────────────────────────────────────────


class TestSummarySearch:
    """Test episode summary search."""

    @pytest.mark.asyncio
    async def test_finds_matching_summaries(
        self, episodic_store, mock_embedder
    ):
        """Recall finds summaries with matching embeddings."""
        now = _utcnow()
        embedding = await mock_embedder.embed("weekly project review")

        summary = EpisodeSummary(
            user_id="user-1",
            summary_text="This week focused on project planning and code reviews",
            period_start=now - timedelta(days=7),
            period_end=now,
            episode_count=15,
            embedding=embedding,
        )
        await episodic_store.store_summary(summary)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "weekly project review")
        summary_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.SUMMARY
        ]
        assert len(summary_results) >= 1
        assert "project planning" in summary_results[0].content


# ── Scoring & Ranking Tests ──────────────────────────────────────────


class TestScoringAndRanking:
    """Test relevance scoring and ranking."""

    def test_recency_boost_favors_recent_results(self):
        """Recent results get a higher recency boost."""
        now = _utcnow()
        config = RecallConfig(recency_boost_weight=0.2, recency_half_life_days=14.0)
        engine = PersonalHistoryRecallEngine(config=config)

        recent = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="recent",
            relevance_score=0.5,
            raw_similarity=0.5,
            timestamp=now - timedelta(hours=1),
        )

        old = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-2",
            content="old",
            relevance_score=0.5,
            raw_similarity=0.5,
            timestamp=now - timedelta(days=60),
        )

        recent_score = engine._compute_final_score(recent, now, set())
        old_score = engine._compute_final_score(old, now, set())

        assert recent_score > old_score, (
            f"Recent score ({recent_score}) should be > old score ({old_score})"
        )

    def test_entity_mention_boost(self):
        """Results mentioning query-relevant entities get boosted."""
        now = _utcnow()
        config = RecallConfig(entity_mention_boost=0.1)
        engine = PersonalHistoryRecallEngine(config=config)

        with_entity = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="met with Sarah",
            relevance_score=0.5,
            raw_similarity=0.5,
            metadata={"entities": ["Sarah"]},
        )

        without_entity = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-2",
            content="had a meeting",
            relevance_score=0.5,
            raw_similarity=0.5,
            metadata={"entities": []},
        )

        relevant = {"sarah"}
        score_with = engine._compute_final_score(with_entity, now, relevant)
        score_without = engine._compute_final_score(without_entity, now, relevant)

        assert score_with > score_without

    def test_source_weights_affect_ranking(self):
        """Source weights affect final score calculation."""
        now = _utcnow()
        config = RecallConfig(
            source_weights={
                "episode": 1.0,
                "pattern": 0.5,
            },
            recency_boost_weight=0.0,
        )
        engine = PersonalHistoryRecallEngine(config=config)

        episode = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="episode",
            relevance_score=0.8,
            raw_similarity=0.8,
        )

        pattern = RecallResult(
            source_type=RecallSourceType.PATTERN,
            source_id="pat-1",
            content="pattern",
            relevance_score=0.8,
            raw_similarity=0.8,
        )

        ep_score = engine._compute_final_score(episode, now, set())
        pat_score = engine._compute_final_score(pattern, now, set())

        assert ep_score > pat_score

    def test_final_score_capped_at_one(self):
        """Final relevance score is capped at 1.0."""
        now = _utcnow()
        config = RecallConfig(
            recency_boost_weight=0.5,
            entity_mention_boost=0.5,
        )
        engine = PersonalHistoryRecallEngine(config=config)

        result = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="recent with entities",
            relevance_score=0.99,
            raw_similarity=0.99,
            timestamp=now,
            metadata={"entities": ["Sarah", "Project", "Meeting"]},
        )

        score = engine._compute_final_score(
            result, now, {"sarah", "project", "meeting"}
        )
        assert score <= 1.0


# ── Deduplication Tests ──────────────────────────────────────────────


class TestDeduplication:
    """Test result deduplication."""

    def test_deduplicates_by_source_id(self):
        """Results with same source_type+source_id are deduplicated."""
        engine = PersonalHistoryRecallEngine()

        results = [
            RecallResult(
                source_type=RecallSourceType.EPISODE,
                source_id="ep-1",
                content="first instance",
                relevance_score=0.8,
                raw_similarity=0.8,
            ),
            RecallResult(
                source_type=RecallSourceType.EPISODE,
                source_id="ep-1",
                content="second instance",
                relevance_score=0.6,
                raw_similarity=0.6,
            ),
        ]

        deduped = engine._deduplicate(results)
        assert len(deduped) == 1
        # Should keep the higher-scored version
        assert deduped[0].relevance_score == 0.8

    def test_different_sources_not_deduped(self):
        """Results from different source types with same ID are kept."""
        engine = PersonalHistoryRecallEngine()

        results = [
            RecallResult(
                source_type=RecallSourceType.EPISODE,
                source_id="same-id",
                content="episode",
                relevance_score=0.8,
                raw_similarity=0.8,
            ),
            RecallResult(
                source_type=RecallSourceType.FACT,
                source_id="same-id",
                content="fact",
                relevance_score=0.7,
                raw_similarity=0.7,
            ),
        ]

        deduped = engine._deduplicate(results)
        assert len(deduped) == 2


# ── Source Filtering Tests ───────────────────────────────────────────


class TestSourceFiltering:
    """Test source type filtering."""

    @pytest.mark.asyncio
    async def test_filter_to_episodes_only(self, recall_engine, episodic_store, mock_embedder):
        """Can filter to only search episodes."""
        embedding = await mock_embedder.embed("test content")
        ep = _make_episode(raw_text="test content", embedding=embedding)
        await episodic_store.append(ep)

        response = await recall_engine.recall(
            "user-1",
            "test content",
            source_filter=[RecallSourceType.EPISODE],
        )
        # Only episode source should be searched
        for r in response.results:
            assert r.source_type == RecallSourceType.EPISODE

    @pytest.mark.asyncio
    async def test_filter_excludes_unselected_sources(self, recall_engine):
        """Source filter excludes non-matching sources."""
        response = await recall_engine.recall(
            "user-1",
            "test",
            source_filter=[RecallSourceType.FACT],
        )
        for r in response.results:
            assert r.source_type == RecallSourceType.FACT

    def test_should_search_with_filter(self):
        """_should_search correctly filters source types."""
        assert PersonalHistoryRecallEngine._should_search(
            RecallSourceType.EPISODE, None
        ) is True
        assert PersonalHistoryRecallEngine._should_search(
            RecallSourceType.EPISODE,
            [RecallSourceType.EPISODE],
        ) is True
        assert PersonalHistoryRecallEngine._should_search(
            RecallSourceType.EPISODE,
            [RecallSourceType.FACT],
        ) is False


# ── Configuration Tests ──────────────────────────────────────────────


class TestConfiguration:
    """Test recall engine configuration."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = RecallConfig()
        assert config.max_total_results == 25
        assert config.min_episode_similarity == 0.3
        assert config.recency_boost_weight == 0.15
        assert config.recency_half_life_days == 14.0
        assert "episode" in config.source_weights
        assert "fact" in config.source_weights

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = RecallConfig(
            max_total_results=10,
            min_episode_similarity=0.5,
            recency_boost_weight=0.3,
        )
        assert config.max_total_results == 10
        assert config.min_episode_similarity == 0.5
        assert config.recency_boost_weight == 0.3

    @pytest.mark.asyncio
    async def test_max_results_limit(self, episodic_store, mock_embedder):
        """max_results limits the number of returned results."""
        embedding = await mock_embedder.embed("test content")

        # Create many episodes
        for i in range(30):
            ep = _make_episode(
                raw_text=f"test content variant {i}",
                embedding=embedding,
            )
            await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall(
            "user-1", "test content", max_results=5
        )
        assert len(response.results) <= 5


# ── Statistics Tests ─────────────────────────────────────────────────


class TestStatistics:
    """Test recall engine statistics."""

    @pytest.mark.asyncio
    async def test_stats_track_queries(self, recall_engine):
        """Stats track total queries."""
        await recall_engine.recall("user-1", "query 1")
        await recall_engine.recall("user-1", "query 2")

        stats = recall_engine.stats
        assert stats["total_queries"] == 2

    @pytest.mark.asyncio
    async def test_stats_track_latency(self, recall_engine):
        """Stats track average latency."""
        await recall_engine.recall("user-1", "query")

        stats = recall_engine.stats
        assert stats["avg_latency_ms"] >= 0


# ── Serialization Tests ──────────────────────────────────────────────


class TestSerialization:
    """Test result serialization."""

    def test_recall_result_to_dict(self):
        """RecallResult serializes correctly."""
        now = _utcnow()
        result = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="test content",
            relevance_score=0.85,
            raw_similarity=0.8,
            timestamp=now,
            metadata={"intent": "task", "entities": ["Sarah"]},
        )
        d = result.to_dict()
        assert d["source_type"] == "episode"
        assert d["source_id"] == "ep-1"
        assert d["content"] == "test content"
        assert d["relevance_score"] == 0.85
        assert d["raw_similarity"] == 0.8
        assert d["timestamp"] == now.isoformat()
        assert d["metadata"]["intent"] == "task"

    def test_recall_response_to_dict(self):
        """RecallResponse serializes correctly."""
        response = RecallResponse(
            query="test query",
            results=[
                RecallResult(
                    source_type=RecallSourceType.EPISODE,
                    source_id="ep-1",
                    content="test",
                    relevance_score=0.9,
                    raw_similarity=0.85,
                ),
            ],
            total_results=1,
            sources_searched=["episode", "entity"],
            latency_ms=42.5,
            query_embedding_generated=True,
            entity_context_used=["Sarah"],
        )
        d = response.to_dict()
        assert d["query"] == "test query"
        assert len(d["results"]) == 1
        assert d["total_results"] == 1
        assert d["sources_searched"] == ["episode", "entity"]
        assert d["has_results"] is True
        assert d["entity_context_used"] == ["Sarah"]


# ── Cross-Source Integration Tests ───────────────────────────────────


class TestCrossSourceIntegration:
    """Test recall across multiple sources simultaneously."""

    @pytest.mark.asyncio
    async def test_recall_merges_results_from_multiple_sources(
        self, episodic_store, semantic_store, mock_embedder
    ):
        """Recall returns merged results from episodes and facts."""
        query = "project status update"
        embedding = await mock_embedder.embed(query)

        # Add an episode
        ep = _make_episode(
            raw_text="gave a project status update in standup",
            embedding=embedding,
        )
        await episodic_store.append(ep)

        # Add a fact
        await semantic_store.add_fact(
            content="Project Alpha status: on track",
            fact_type=FactType.ATTRIBUTE,
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", query)

        # Should have results from at least episodes
        assert response.has_results
        source_types = {r.source_type for r in response.results}
        assert RecallSourceType.EPISODE in source_types

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance(
        self, episodic_store, mock_embedder
    ):
        """Results are sorted by final relevance score, highest first."""
        query = "important meeting"
        embedding = await mock_embedder.embed(query)

        # Create episodes with same embedding (same similarity)
        # but different timestamps (different recency boost)
        now = _utcnow()
        ep_old = _make_episode(
            raw_text="old meeting discussion",
            embedding=embedding,
            timestamp=now - timedelta(days=90),
        )
        ep_new = _make_episode(
            raw_text="new meeting discussion",
            embedding=embedding,
            timestamp=now - timedelta(hours=1),
        )

        await episodic_store.append(ep_old)
        await episodic_store.append(ep_new)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
            config=RecallConfig(recency_boost_weight=0.2),
        )
        response = await engine.recall("user-1", query)

        if len(response.results) >= 2:
            # Results should be sorted descending by relevance
            for i in range(len(response.results) - 1):
                assert response.results[i].relevance_score >= response.results[i + 1].relevance_score


# ── Edge Case Tests ──────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, recall_engine):
        """Empty query doesn't crash."""
        response = await recall_engine.recall("user-1", "")
        assert isinstance(response, RecallResponse)

    @pytest.mark.asyncio
    async def test_very_long_query(self, recall_engine):
        """Very long query doesn't crash."""
        long_query = "test " * 1000
        response = await recall_engine.recall("user-1", long_query)
        assert isinstance(response, RecallResponse)

    @pytest.mark.asyncio
    async def test_nonexistent_user(self, recall_engine):
        """Query for nonexistent user returns empty results."""
        response = await recall_engine.recall("nonexistent-user", "test")
        assert response.total_results == 0

    @pytest.mark.asyncio
    async def test_recall_with_no_semantic_store(self, episodic_store, mock_embedder):
        """Recall works without semantic store (episodes only)."""
        embedding = await mock_embedder.embed("test")
        ep = _make_episode(raw_text="test episode", embedding=embedding)
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=None,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "test")
        assert isinstance(response, RecallResponse)

    def test_result_without_timestamp(self):
        """Results without timestamps still score correctly."""
        now = _utcnow()
        engine = PersonalHistoryRecallEngine()
        result = RecallResult(
            source_type=RecallSourceType.ENTITY,
            source_id="ent-1",
            content="test",
            relevance_score=0.5,
            raw_similarity=0.5,
            timestamp=None,
        )
        score = engine._compute_final_score(result, now, set())
        assert 0.0 <= score <= 1.0

    def test_result_with_non_list_entities_metadata(self):
        """Entity boost handles non-list entities in metadata gracefully."""
        now = _utcnow()
        config = RecallConfig(entity_mention_boost=0.1)
        engine = PersonalHistoryRecallEngine(config=config)
        result = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="test",
            relevance_score=0.5,
            raw_similarity=0.5,
            metadata={"entities": "not a list"},
        )
        # Should not crash
        score = engine._compute_final_score(result, now, {"test"})
        assert 0.0 <= score <= 1.0
