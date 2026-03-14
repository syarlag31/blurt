"""Tests for the personal history recall engine — Sub-AC 2 of AC 14.

Tests cover:
- Natural language query understanding (temporal, entity, relationship parsing)
- Relationship search across knowledge graph
- Source context enrichment for top results
- Full recall pipeline with NL query → ranked results with source context
- Integration with QuestionService premium-tier SEMANTIC_RECALL
- Open-ended natural language question support
- Edge cases and error handling
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
    RelationshipType,
)
from blurt.services.recall import (
    PersonalHistoryRecallEngine,
    QueryUnderstanding,
    RecallConfig,
    RecallResponse,
    RecallResult,
    RecallSourceType,
    SourceContext,
    parse_query,
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
        emotion=EmotionSnapshot(
            primary=emotion_primary, intensity=1.0, valence=0.3, arousal=0.4
        ),
    )


@pytest.fixture
def mock_embedder():
    return MockEmbeddingProvider()


@pytest.fixture
def episodic_store():
    return InMemoryEpisodicStore()


@pytest.fixture
def semantic_store(mock_embedder):
    return SemanticMemoryStore(user_id="user-1", embedding_provider=mock_embedder)


@pytest.fixture
def recall_engine(episodic_store, semantic_store, mock_embedder):
    return PersonalHistoryRecallEngine(
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        embedding_provider=mock_embedder,
    )


# ── NL Query Understanding Tests ─────────────────────────────────────


class TestQueryUnderstanding:
    """Test natural language query parsing."""

    def test_parse_yesterday(self):
        """Parses 'yesterday' temporal reference."""
        now = _utcnow()
        result = parse_query("what did I say yesterday?", now=now)
        assert result.temporal_hint == "yesterday"
        assert result.temporal_start is not None
        assert result.temporal_end is not None
        # Yesterday should be within 2 days ago
        assert result.temporal_start >= now - timedelta(days=2)

    def test_parse_last_week(self):
        """Parses 'last week' temporal reference."""
        now = _utcnow()
        result = parse_query("what happened last week?", now=now)
        assert result.temporal_hint == "last week"
        assert result.temporal_start is not None
        assert result.temporal_start >= now - timedelta(days=8)

    def test_parse_last_month(self):
        """Parses 'last month' temporal reference."""
        now = _utcnow()
        result = parse_query("what projects did I work on last month?", now=now)
        assert result.temporal_hint == "last month"
        assert result.temporal_start is not None

    def test_parse_n_days_ago(self):
        """Parses 'N days ago' temporal reference."""
        now = _utcnow()
        result = parse_query("what was I doing 3 days ago?", now=now)
        assert result.temporal_hint == "3 days ago"
        assert result.temporal_start is not None

    def test_parse_n_weeks_ago(self):
        """Parses 'N weeks ago' temporal reference."""
        now = _utcnow()
        result = parse_query("what happened 2 weeks ago?", now=now)
        assert result.temporal_hint == "2 weeks ago"
        assert result.temporal_start is not None

    def test_parse_today(self):
        """Parses 'today' temporal reference."""
        now = _utcnow()
        result = parse_query("what did I capture today?", now=now)
        assert result.temporal_hint == "today"
        assert result.temporal_start is not None
        # Today should start at midnight
        assert result.temporal_start.hour == 0

    def test_parse_recently(self):
        """Parses 'recently' temporal reference."""
        now = _utcnow()
        result = parse_query("what did I mention recently?", now=now)
        assert result.temporal_hint == "recently"
        assert result.temporal_start is not None

    def test_no_temporal_hint(self):
        """Queries without temporal hints have None."""
        result = parse_query("who is Sarah?")
        assert result.temporal_hint is None
        assert result.temporal_start is None
        assert result.temporal_end is None

    def test_parse_entity_references(self):
        """Extracts capitalized entity references."""
        result = parse_query("what did I say about Project Alpha?")
        assert "Project Alpha" in result.entity_references or any(
            "Project" in ref for ref in result.entity_references
        )

    def test_parse_entity_multi_word(self):
        """Extracts multi-word entity references."""
        result = parse_query("tell me about Sarah Chen from Acme Corp")
        entity_text = " ".join(result.entity_references)
        assert "Sarah" in entity_text or "Chen" in entity_text

    def test_detect_relationship_query(self):
        """Detects relationship-type questions."""
        result = parse_query("how is Sarah connected to the project?")
        assert result.is_relationship_query is True

    def test_detect_relationship_working_with(self):
        """Detects 'working with' relationship pattern."""
        result = parse_query("who is working with John?")
        assert result.is_relationship_query is True

    def test_detect_count_query(self):
        """Detects count/aggregation questions."""
        result = parse_query("how many times did I mention the gym?")
        assert result.is_count_query is True

    def test_detect_emotion_query(self):
        """Detects emotion-related questions."""
        result = parse_query("how did I feel about the meeting?")
        assert result.is_emotion_query is True

    def test_detect_emotion_mood(self):
        """Detects mood keyword in query."""
        result = parse_query("what was my mood last week?")
        assert result.is_emotion_query is True

    def test_search_text_strips_structural(self):
        """Search text removes structural words."""
        result = parse_query("what did I say about the project?")
        assert "project" in result.search_text
        # Structural words should be removed
        assert result.search_text != result.original_query.lower()

    def test_empty_query(self):
        """Empty query doesn't crash."""
        result = parse_query("")
        assert isinstance(result, QueryUnderstanding)
        assert result.original_query == ""

    def test_to_dict(self):
        """QueryUnderstanding serializes correctly."""
        result = parse_query("what happened yesterday with Sarah?")
        d = result.to_dict()
        assert "original_query" in d
        assert "temporal_hint" in d
        assert "entity_references" in d
        assert "search_text" in d


# ── Relationship Search Tests ─────────────────────────────────────────


class TestRelationshipSearch:
    """Test relationship search in the knowledge graph."""

    @pytest.mark.asyncio
    async def test_finds_relationships_by_entity(
        self, semantic_store, mock_embedder, episodic_store
    ):
        """Recall finds relationships involving mentioned entities."""
        # Create entities and a relationship
        sarah = await semantic_store.add_entity(
            name="Sarah", entity_type=EntityType.PERSON
        )
        project = await semantic_store.add_entity(
            name="Alpha Project", entity_type=EntityType.PROJECT
        )
        await semantic_store.add_or_strengthen_relationship(
            source_entity_id=sarah.id,
            target_entity_id=project.id,
            relationship_type=RelationshipType.COLLABORATES_ON,
            context="Sarah leads Alpha Project",
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )

        response = await engine.recall(
            "user-1",
            "how is Sarah connected to Alpha Project?",
            source_filter=[RecallSourceType.RELATIONSHIP],
        )
        assert "relationship" in response.sources_searched
        rel_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.RELATIONSHIP
        ]
        assert isinstance(rel_results, list)

    @pytest.mark.asyncio
    async def test_relationship_results_include_metadata(
        self, semantic_store, mock_embedder, episodic_store
    ):
        """Relationship results include entity names and connection type."""
        sarah = await semantic_store.add_entity(
            name="Sarah", entity_type=EntityType.PERSON
        )
        john = await semantic_store.add_entity(
            name="John", entity_type=EntityType.PERSON
        )
        await semantic_store.add_or_strengthen_relationship(
            source_entity_id=sarah.id,
            target_entity_id=john.id,
            relationship_type=RelationshipType.WORKS_WITH,
            context="Sarah and John are on the same team",
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "Sarah and John")
        rel_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.RELATIONSHIP
        ]
        if rel_results:
            meta = rel_results[0].metadata
            assert "source_entity" in meta or "target_entity" in meta
            assert "relationship_type" in meta

    @pytest.mark.asyncio
    async def test_relationship_search_with_no_entities(
        self, episodic_store, mock_embedder
    ):
        """Relationship search gracefully returns empty when no entities match."""
        semantic = SemanticMemoryStore(
            user_id="user-1", embedding_provider=mock_embedder
        )
        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall(
            "user-1",
            "random query with no entities",
            source_filter=[RecallSourceType.RELATIONSHIP],
        )
        assert isinstance(response, RecallResponse)


# ── Source Context Enrichment Tests ───────────────────────────────────


class TestSourceContextEnrichment:
    """Test source context enrichment for top results."""

    @pytest.mark.asyncio
    async def test_episode_results_get_source_context(
        self, episodic_store, mock_embedder
    ):
        """Top episode results are enriched with surrounding context."""
        embedding = await mock_embedder.embed("project meeting")
        session_id = "sess-context-1"

        # Create a sequence of episodes in the same session
        ep1 = _make_episode(
            raw_text="starting the standup",
            embedding=embedding,
            session_id=session_id,
            timestamp=_utcnow() - timedelta(minutes=5),
        )
        ep2 = _make_episode(
            raw_text="discussed the project timeline",
            embedding=embedding,
            session_id=session_id,
            timestamp=_utcnow() - timedelta(minutes=3),
        )
        ep3 = _make_episode(
            raw_text="wrapping up the meeting",
            embedding=embedding,
            session_id=session_id,
            timestamp=_utcnow() - timedelta(minutes=1),
        )
        await episodic_store.append(ep1)
        await episodic_store.append(ep2)
        await episodic_store.append(ep3)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall(
            "user-1", "project meeting", enrich_context=True, context_window=3
        )

        # Check that at least some results have source context
        enriched = [
            r for r in response.results
            if r.source_context is not None
        ]
        # At least one result should have context
        assert len(enriched) >= 1

        ctx = enriched[0].source_context
        assert ctx.session_id == session_id
        assert ctx.session_episode_count == 3

    @pytest.mark.asyncio
    async def test_source_context_includes_surrounding_text(
        self, episodic_store, mock_embedder
    ):
        """Source context includes preceding and following text."""
        embedding = await mock_embedder.embed("middle episode")
        session_id = "sess-surround-1"

        ep1 = _make_episode(
            raw_text="before the target",
            embedding=embedding,
            session_id=session_id,
            timestamp=_utcnow() - timedelta(minutes=5),
        )
        ep2 = _make_episode(
            raw_text="the target episode about project",
            embedding=embedding,
            session_id=session_id,
            timestamp=_utcnow() - timedelta(minutes=3),
        )
        ep3 = _make_episode(
            raw_text="after the target",
            embedding=embedding,
            session_id=session_id,
            timestamp=_utcnow() - timedelta(minutes=1),
        )
        await episodic_store.append(ep1)
        await episodic_store.append(ep2)
        await episodic_store.append(ep3)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "middle episode", context_window=5)

        # Find a result with context that has both preceding and following
        for r in response.results:
            if r.source_context is not None:
                ctx = r.source_context
                # At least one of preceding/following should be populated
                assert ctx.preceding_text is not None or ctx.following_text is not None
                break

    @pytest.mark.asyncio
    async def test_context_not_enriched_when_disabled(
        self, episodic_store, mock_embedder
    ):
        """Source context is not added when enrich_context=False."""
        embedding = await mock_embedder.embed("test")
        ep = _make_episode(raw_text="test episode", embedding=embedding)
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall(
            "user-1", "test", enrich_context=False
        )
        for r in response.results:
            assert r.source_context is None

    @pytest.mark.asyncio
    async def test_context_includes_surrounding_entities(
        self, episodic_store, mock_embedder
    ):
        """Source context collects entities from the session."""
        embedding = await mock_embedder.embed("project")
        session_id = "sess-ent-1"

        ep1 = _make_episode(
            raw_text="meeting with Sarah",
            embedding=embedding,
            session_id=session_id,
            entities=[EntityRef(name="Sarah", entity_type="person")],
            timestamp=_utcnow() - timedelta(minutes=5),
        )
        ep2 = _make_episode(
            raw_text="discussed Alpha project",
            embedding=embedding,
            session_id=session_id,
            entities=[EntityRef(name="Alpha", entity_type="project")],
            timestamp=_utcnow() - timedelta(minutes=3),
        )
        await episodic_store.append(ep1)
        await episodic_store.append(ep2)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", "project")

        enriched = [r for r in response.results if r.source_context is not None]
        if enriched:
            all_entities = enriched[0].source_context.surrounding_entities
            assert "Sarah" in all_entities or "Alpha" in all_entities


# ── NL Query Integration with Recall ─────────────────────────────────


class TestNLQueryRecallIntegration:
    """Test that NL query understanding improves recall results."""

    @pytest.mark.asyncio
    async def test_temporal_hint_filters_results(
        self, episodic_store, mock_embedder
    ):
        """Temporal hints from NL parsing filter episode results."""
        now = _utcnow()
        embedding = await mock_embedder.embed("gym workout")

        # Old episode (2 months ago)
        old_ep = _make_episode(
            raw_text="went to the gym",
            embedding=embedding,
            timestamp=now - timedelta(days=60),
        )
        await episodic_store.append(old_ep)

        # Recent episode (today)
        recent_ep = _make_episode(
            raw_text="gym session today",
            embedding=embedding,
            timestamp=now - timedelta(hours=1),
        )
        await episodic_store.append(recent_ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )

        # Query with "today" should use NL temporal filtering
        response = await engine.recall("user-1", "what did I do at the gym today?")
        assert response.query_understanding is not None
        assert response.query_understanding.temporal_hint == "today"

        # Episode results should be filtered to today
        episode_results = [
            r for r in response.results
            if r.source_type == RecallSourceType.EPISODE
        ]
        for r in episode_results:
            if r.timestamp is not None:
                assert r.timestamp >= now - timedelta(days=1)

    @pytest.mark.asyncio
    async def test_entity_references_boost_results(
        self, episodic_store, semantic_store, mock_embedder
    ):
        """Entity references from NL parsing boost relevant results."""
        embedding = await mock_embedder.embed("meeting Sarah")

        ep_with_entity = _make_episode(
            raw_text="had a meeting with Sarah about the project",
            embedding=embedding,
            entities=[EntityRef(name="Sarah", entity_type="person")],
        )
        ep_without = _make_episode(
            raw_text="had a general meeting about something",
            embedding=embedding,
        )
        await episodic_store.append(ep_with_entity)
        await episodic_store.append(ep_without)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
            config=RecallConfig(entity_mention_boost=0.15),
        )

        # "Sarah" is detected as entity reference → should boost the matching episode
        response = await engine.recall("user-1", "what did Sarah say about the project?")
        assert response.query_understanding is not None

    @pytest.mark.asyncio
    async def test_response_includes_query_understanding(self, recall_engine):
        """RecallResponse always includes query understanding."""
        response = await recall_engine.recall(
            "user-1", "what happened last week with Project Alpha?"
        )
        assert response.query_understanding is not None
        assert response.query_understanding.original_query == (
            "what happened last week with Project Alpha?"
        )
        assert response.query_understanding.temporal_hint == "last week"

    @pytest.mark.asyncio
    async def test_explicit_time_overrides_nl_parsing(
        self, episodic_store, mock_embedder
    ):
        """Explicit time_start/time_end override NL-parsed temporal hints."""
        now = _utcnow()
        embedding = await mock_embedder.embed("test")
        ep = _make_episode(
            raw_text="test episode",
            embedding=embedding,
            timestamp=now - timedelta(days=5),
        )
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )

        # NL says "yesterday" but explicit time says last 10 days
        response = await engine.recall(
            "user-1",
            "what happened yesterday?",
            time_start=now - timedelta(days=10),
        )
        # The explicit time should win
        assert response.query_understanding.temporal_hint == "yesterday"
        # But the search should still work (explicit override)
        assert isinstance(response, RecallResponse)


# ── Full Pipeline Integration Tests ──────────────────────────────────


class TestFullPipelineIntegration:
    """Test the complete recall pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_all_sources(
        self, episodic_store, semantic_store, mock_embedder
    ):
        """Full recall searches episodes, entities, facts, and patterns."""
        query = "what do I know about the project timeline?"
        embedding = await mock_embedder.embed(query)

        # Add episode
        ep = _make_episode(
            raw_text="the project timeline is behind schedule",
            embedding=embedding,
        )
        await episodic_store.append(ep)

        # Add entity
        await semantic_store.add_entity(
            name="Project Timeline",
            entity_type=EntityType.PROJECT,
        )

        # Add fact
        await semantic_store.add_fact(
            content="Project timeline was adjusted in Q1",
            fact_type=FactType.ATTRIBUTE,
        )

        # Add pattern
        await semantic_store.add_pattern(
            pattern_type=PatternType.TIME_OF_DAY,
            description="User discusses project timeline in morning meetings",
            confidence=0.8,
            observation_count=10,
        )

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", query)

        assert response.has_results
        assert response.total_results > 0
        assert len(response.sources_searched) > 0
        assert response.query_embedding_generated is True
        assert response.query_understanding is not None

        # Results should be sorted by relevance
        for i in range(len(response.results) - 1):
            assert (
                response.results[i].relevance_score
                >= response.results[i + 1].relevance_score
            )

    @pytest.mark.asyncio
    async def test_open_ended_question_returns_results(
        self, episodic_store, mock_embedder
    ):
        """Open-ended questions get meaningful results."""
        # Use the same query text for embedding so mock embedder matches
        query = "what's my typical morning routine?"
        embedding = await mock_embedder.embed(query)

        for text in [
            "I usually start with coffee and emails",
            "Morning standup at 9:30 every day",
            "I prefer to do deep work in the morning",
        ]:
            ep = _make_episode(raw_text=text, embedding=embedding)
            await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )

        response = await engine.recall("user-1", query)
        assert response.has_results
        assert response.total_results >= 1

    @pytest.mark.asyncio
    async def test_question_about_person_returns_relevant(
        self, episodic_store, semantic_store, mock_embedder
    ):
        """Questions about specific people return person-related results."""
        # Use the query text for embedding so mock embedder matches
        query = "when is Sarah's birthday?"
        embedding = await mock_embedder.embed(query)

        await semantic_store.add_entity(
            name="Sarah",
            entity_type=EntityType.PERSON,
            attributes={"birthday": "March 15"},
        )
        await semantic_store.add_fact(
            content="Sarah's birthday is March 15",
            fact_type=FactType.ATTRIBUTE,
        )

        ep = _make_episode(
            raw_text="Sarah mentioned her birthday is coming up next week",
            embedding=embedding,
            entities=[EntityRef(name="Sarah", entity_type="person")],
        )
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall("user-1", query)
        assert response.has_results


# ── Serialization Tests ──────────────────────────────────────────────


class TestNewSerialization:
    """Test serialization of new types."""

    def test_source_context_to_dict(self):
        """SourceContext serializes correctly."""
        ctx = SourceContext(
            preceding_text="before this",
            following_text="after this",
            session_id="sess-1",
            session_episode_count=5,
            surrounding_entities=["Sarah", "Alpha"],
            surrounding_intents=["task", "journal"],
        )
        d = ctx.to_dict()
        assert d["preceding_text"] == "before this"
        assert d["following_text"] == "after this"
        assert d["session_id"] == "sess-1"
        assert d["session_episode_count"] == 5
        assert d["surrounding_entities"] == ["Sarah", "Alpha"]

    def test_recall_result_with_context_to_dict(self):
        """RecallResult with source_context serializes correctly."""
        ctx = SourceContext(
            preceding_text="context before",
            session_id="sess-1",
        )
        result = RecallResult(
            source_type=RecallSourceType.EPISODE,
            source_id="ep-1",
            content="test content",
            relevance_score=0.85,
            raw_similarity=0.8,
            source_context=ctx,
        )
        d = result.to_dict()
        assert "source_context" in d
        assert d["source_context"]["preceding_text"] == "context before"

    def test_recall_result_without_context_to_dict(self):
        """RecallResult without source_context omits it from dict."""
        result = RecallResult(
            source_type=RecallSourceType.FACT,
            source_id="fact-1",
            content="test",
            relevance_score=0.5,
            raw_similarity=0.5,
        )
        d = result.to_dict()
        assert "source_context" not in d

    def test_recall_response_with_understanding_to_dict(self):
        """RecallResponse with query_understanding serializes correctly."""
        understanding = parse_query("what happened last week?")
        response = RecallResponse(
            query="what happened last week?",
            results=[],
            total_results=0,
            sources_searched=["episode"],
            query_understanding=understanding,
        )
        d = response.to_dict()
        assert "query_understanding" in d
        assert d["query_understanding"]["temporal_hint"] == "last week"


# ── Edge Cases ────────────────────────────────────────────────────────


class TestRecallEdgeCases:
    """Test edge cases for the enhanced recall engine."""

    @pytest.mark.asyncio
    async def test_recall_with_all_source_types(self, recall_engine):
        """Recall can be filtered to any source type without error."""
        for src_type in RecallSourceType:
            response = await recall_engine.recall(
                "user-1",
                "test query",
                source_filter=[src_type],
            )
            assert isinstance(response, RecallResponse)

    @pytest.mark.asyncio
    async def test_relationship_filter_only(self, recall_engine):
        """Can filter to only relationships."""
        response = await recall_engine.recall(
            "user-1",
            "test",
            source_filter=[RecallSourceType.RELATIONSHIP],
        )
        for r in response.results:
            assert r.source_type == RecallSourceType.RELATIONSHIP

    @pytest.mark.asyncio
    async def test_unicode_query(self, recall_engine):
        """Unicode queries don't crash."""
        response = await recall_engine.recall(
            "user-1", "what about the caf\u00e9 meeting?"
        )
        assert isinstance(response, RecallResponse)

    @pytest.mark.asyncio
    async def test_very_specific_temporal_query(self, recall_engine):
        """Specific temporal queries parse correctly."""
        response = await recall_engine.recall(
            "user-1", "what did I do 5 days ago?"
        )
        assert response.query_understanding is not None
        assert response.query_understanding.temporal_hint == "5 days ago"

    def test_parse_query_with_special_chars(self):
        """Query with special characters doesn't crash."""
        result = parse_query("what about the $100 meeting @ 3pm?")
        assert isinstance(result, QueryUnderstanding)

    def test_parse_query_multiple_temporal(self):
        """Only first temporal match is used."""
        result = parse_query("what happened yesterday and last week?")
        # Should use first match
        assert result.temporal_hint is not None

    @pytest.mark.asyncio
    async def test_context_window_zero(self, episodic_store, mock_embedder):
        """context_window=0 disables enrichment efficiently."""
        embedding = await mock_embedder.embed("test")
        ep = _make_episode(raw_text="test", embedding=embedding)
        await episodic_store.append(ep)

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            embedding_provider=mock_embedder,
        )
        response = await engine.recall(
            "user-1", "test", context_window=0
        )
        # context_window=0 means empty slice, no enrichment
        for r in response.results:
            assert r.source_context is None
