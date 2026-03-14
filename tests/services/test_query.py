"""Tests for the structured query parser and executor (AC 14, Sub-AC 1).

Tests cover:
- Query parser: all 7 query types from natural language
- Entity type filter extraction
- Date range parsing (relative references)
- Topic extraction
- Query executor: all query type handlers
- Edge cases: empty queries, ambiguous questions, no results
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from blurt.memory.episodic import (
    EntityRef,
    Episode,
    EpisodeContext,
    InMemoryEpisodicStore,
)
from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    FactType,
    RelationshipEdge,
    RelationshipType,
    SemanticSearchResult,
)
from blurt.services.query import (
    DateRangeFilter,
    DateReference,
    QueryExecutor,
    QueryParser,
    QueryResult,
    QueryType,
    ResponseFormatter,
    StructuredQuery,
)


# ── Fixtures ──────────────────────────────────────────────────────────

# Fixed time for deterministic tests
FIXED_NOW = datetime(2026, 3, 13, 14, 0, 0, tzinfo=timezone.utc)  # Friday


class FakeGraphStore:
    """Minimal in-memory graph store for testing the query executor."""

    def __init__(self, user_id: str = "user-1"):
        self.user_id = user_id
        self._entities: dict[str, EntityNode] = {}
        self._facts: dict[str, Fact] = {}
        self._relationships: dict[str, RelationshipEdge] = {}

    def add_test_entity(self, entity: EntityNode) -> None:
        self._entities[entity.id] = entity

    def add_test_fact(self, fact: Fact) -> None:
        self._facts[fact.id] = fact

    def add_test_relationship(self, rel: RelationshipEdge) -> None:
        self._relationships[rel.id] = rel

    async def find_entity_by_name(self, name: str) -> EntityNode | None:
        normalized = name.lower().strip()
        for e in self._entities.values():
            if e.normalized_name == normalized:
                return e
            if normalized in [a.lower() for a in e.aliases]:
                return e
        return None

    async def get_entity(self, entity_id: str) -> EntityNode | None:
        return self._entities.get(entity_id)

    async def get_all_entities(
        self, entity_type: EntityType | None = None
    ) -> list[EntityNode]:
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return entities

    async def get_entity_facts(
        self, entity_id: str, active_only: bool = True
    ) -> list[Fact]:
        return [
            f for f in self._facts.values()
            if f.subject_entity_id == entity_id
            and (not active_only or f.is_active)
        ]

    async def get_all_facts(
        self, fact_type: FactType | None = None, active_only: bool = True
    ) -> list[Fact]:
        facts = list(self._facts.values())
        if fact_type:
            facts = [f for f in facts if f.fact_type == fact_type]
        if active_only:
            facts = [f for f in facts if f.is_active]
        return facts

    async def get_entity_relationships(
        self, entity_id: str
    ) -> list[RelationshipEdge]:
        return [
            r for r in self._relationships.values()
            if r.source_entity_id == entity_id
            or r.target_entity_id == entity_id
        ]

    async def search(
        self,
        query: str,
        top_k: int = 10,
        item_types: list[str] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]:
        # Simple keyword match for testing
        results = []
        for entity in self._entities.values():
            if query.lower() in entity.name.lower():
                results.append(SemanticSearchResult(
                    item_type="entity",
                    item_id=entity.id,
                    content=entity.name,
                    similarity_score=0.9,
                ))
        for fact in self._facts.values():
            if query.lower() in fact.content.lower():
                results.append(SemanticSearchResult(
                    item_type="fact",
                    item_id=fact.id,
                    content=fact.content,
                    similarity_score=0.8,
                ))
        return results[:top_k]


def _make_entity(
    name: str,
    entity_type: EntityType = EntityType.PERSON,
    user_id: str = "user-1",
    mention_count: int = 1,
    last_seen: datetime | None = None,
    first_seen: datetime | None = None,
) -> EntityNode:
    return EntityNode(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=name,
        entity_type=entity_type,
        mention_count=mention_count,
        last_seen=last_seen or FIXED_NOW,
        first_seen=first_seen or FIXED_NOW - timedelta(days=30),
    )


def _make_fact(
    content: str,
    subject_entity_id: str,
    fact_type: FactType = FactType.ATTRIBUTE,
    user_id: str = "user-1",
) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        user_id=user_id,
        fact_type=fact_type,
        subject_entity_id=subject_entity_id,
        content=content,
    )


def _make_relationship(
    source_id: str,
    target_id: str,
    rel_type: RelationshipType = RelationshipType.WORKS_WITH,
    user_id: str = "user-1",
) -> RelationshipEdge:
    return RelationshipEdge(
        id=str(uuid.uuid4()),
        user_id=user_id,
        source_entity_id=source_id,
        target_entity_id=target_id,
        relationship_type=rel_type,
    )


def _make_episode(
    raw_text: str,
    user_id: str = "user-1",
    intent: str = "task",
    timestamp: datetime | None = None,
    entities: list[EntityRef] | None = None,
) -> Episode:
    return Episode(
        id=str(uuid.uuid4()),
        user_id=user_id,
        raw_text=raw_text,
        intent=intent,
        timestamp=timestamp or FIXED_NOW,
        entities=entities or [],
        context=EpisodeContext(session_id="test-session"),
    )


@pytest.fixture
def parser() -> QueryParser:
    return QueryParser(now=FIXED_NOW)


@pytest.fixture
def graph_store() -> FakeGraphStore:
    store = FakeGraphStore()

    # Populate with test data
    sarah = _make_entity("Sarah", EntityType.PERSON, mention_count=10)
    jake = _make_entity("Jake", EntityType.PERSON, mention_count=5)
    blurt_project = _make_entity("Blurt", EntityType.PROJECT, mention_count=8)
    acme = _make_entity("Acme Corp", EntityType.ORGANIZATION, mention_count=3)
    office = _make_entity("Downtown Office", EntityType.PLACE, mention_count=2)

    store.add_test_entity(sarah)
    store.add_test_entity(jake)
    store.add_test_entity(blurt_project)
    store.add_test_entity(acme)
    store.add_test_entity(office)

    # Facts
    store.add_test_fact(_make_fact("Sarah is my manager", sarah.id))
    store.add_test_fact(_make_fact(
        "Sarah's birthday is March 15",
        sarah.id,
        FactType.ATTRIBUTE,
    ))
    store.add_test_fact(_make_fact(
        "Jake prefers async communication",
        jake.id,
        FactType.PREFERENCE,
    ))

    # Relationships
    store.add_test_relationship(_make_relationship(
        sarah.id, blurt_project.id, RelationshipType.MANAGES,
    ))
    store.add_test_relationship(_make_relationship(
        sarah.id, jake.id, RelationshipType.WORKS_WITH,
    ))

    return store


@pytest.fixture
async def episodic_store() -> InMemoryEpisodicStore:
    store = InMemoryEpisodicStore()

    # Populate with test episodes
    episodes = [
        _make_episode(
            "Need to finish the quarterly report",
            intent="task",
            timestamp=FIXED_NOW - timedelta(hours=2),
            entities=[EntityRef(name="quarterly report", entity_type="project")],
        ),
        _make_episode(
            "Meeting with Sarah about Blurt at 3pm",
            intent="event",
            timestamp=FIXED_NOW - timedelta(days=1),
            entities=[
                EntityRef(name="Sarah", entity_type="person"),
                EntityRef(name="Blurt", entity_type="project"),
            ],
        ),
        _make_episode(
            "Had a great brainstorming session with Jake",
            intent="journal",
            timestamp=FIXED_NOW - timedelta(days=3),
            entities=[EntityRef(name="Jake", entity_type="person")],
        ),
        _make_episode(
            "Remind me to call the dentist",
            intent="reminder",
            timestamp=FIXED_NOW - timedelta(days=5),
        ),
        _make_episode(
            "What if we added voice to the app?",
            intent="idea",
            timestamp=FIXED_NOW - timedelta(days=8),
        ),
    ]

    for ep in episodes:
        await store.append(ep)

    return store


@pytest.fixture
def executor(
    graph_store: FakeGraphStore,
    episodic_store: InMemoryEpisodicStore,
) -> QueryExecutor:
    return QueryExecutor(
        graph_store=graph_store,  # type: ignore[arg-type]
        episodic_store=episodic_store,
        user_id="user-1",
    )


# ── Parser Tests ──────────────────────────────────────────────────────


class TestQueryParser:
    """Tests for the QueryParser."""

    def test_empty_query(self, parser: QueryParser):
        result = parser.parse("")
        assert result.query_type == QueryType.SEMANTIC
        assert result.raw_text == ""

    def test_entity_lookup_who_is(self, parser: QueryParser):
        result = parser.parse("who is Sarah?")
        assert result.query_type == QueryType.ENTITY_LOOKUP
        assert "Sarah" in result.entity_names

    def test_entity_lookup_what_is(self, parser: QueryParser):
        result = parser.parse("what is the Blurt project?")
        assert result.query_type == QueryType.ENTITY_LOOKUP
        assert "the Blurt project" in result.entity_names

    def test_entity_lookup_tell_me_about(self, parser: QueryParser):
        result = parser.parse("tell me about Jake")
        assert result.query_type == QueryType.ENTITY_LOOKUP
        assert "Jake" in result.entity_names

    def test_entity_list_people(self, parser: QueryParser):
        result = parser.parse("what people do I know?")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.PERSON

    def test_entity_list_projects(self, parser: QueryParser):
        result = parser.parse("show me all projects")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.PROJECT

    def test_entity_list_organizations(self, parser: QueryParser):
        result = parser.parse("which companies do I work with?")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.ORGANIZATION

    def test_count_query_people(self, parser: QueryParser):
        result = parser.parse("how many people do I know?")
        assert result.query_type == QueryType.COUNT
        assert result.entity_type_filter == EntityType.PERSON

    def test_count_query_tasks(self, parser: QueryParser):
        result = parser.parse("how many tasks do I have this week?")
        assert result.query_type == QueryType.COUNT
        assert result.intent_filter == "task"
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.THIS_WEEK

    def test_relationship_query_connected(self, parser: QueryParser):
        result = parser.parse("how is Sarah connected to the Blurt project?")
        assert result.query_type == QueryType.RELATIONSHIP
        assert len(result.entity_names) == 2
        assert "Sarah" in result.entity_names
        assert "the Blurt project" in result.entity_names

    def test_relationship_query_between(self, parser: QueryParser):
        result = parser.parse("relationship between Jake and Sarah")
        assert result.query_type == QueryType.RELATIONSHIP
        assert len(result.entity_names) == 2

    def test_timeline_query_what_happened(self, parser: QueryParser):
        result = parser.parse("what happened last week?")
        assert result.query_type == QueryType.TIMELINE
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.LAST_WEEK

    def test_timeline_query_what_did_i_do(self, parser: QueryParser):
        result = parser.parse("what did I do yesterday?")
        assert result.query_type == QueryType.TIMELINE
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.YESTERDAY

    def test_timeline_defaults_to_this_week(self, parser: QueryParser):
        result = parser.parse("what happened?")
        assert result.query_type == QueryType.TIMELINE
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.THIS_WEEK

    def test_fact_recall_when_is(self, parser: QueryParser):
        result = parser.parse("when is Sarah's birthday?")
        assert result.query_type == QueryType.FACT_RECALL
        assert "Sarah" in result.entity_names

    def test_fact_recall_what_did_i_say(self, parser: QueryParser):
        result = parser.parse("what did I say about the project?")
        assert result.query_type == QueryType.FACT_RECALL

    def test_fact_recall_did_i_ever(self, parser: QueryParser):
        result = parser.parse("did I ever finish that book?")
        assert result.query_type == QueryType.FACT_RECALL

    def test_semantic_fallback(self, parser: QueryParser):
        result = parser.parse("something random with no pattern")
        assert result.query_type == QueryType.SEMANTIC
        assert len(result.topics) > 0

    def test_topics_extracted(self, parser: QueryParser):
        result = parser.parse("something about machine learning and AI")
        topics = result.topics
        assert "machine" in topics
        assert "learning" in topics


class TestDateRangeFilter:
    """Tests for date range parsing."""

    def test_today(self, parser: QueryParser):
        result = parser.parse("what did I do today?")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.TODAY
        assert result.date_range.start is not None
        assert result.date_range.start.hour == 0
        assert result.date_range.end is not None

    def test_yesterday(self, parser: QueryParser):
        result = parser.parse("what happened yesterday?")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.YESTERDAY
        expected_start = FIXED_NOW.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        assert result.date_range.start == expected_start

    def test_this_week(self, parser: QueryParser):
        result = parser.parse("show me this week's tasks")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.THIS_WEEK
        # 2026-03-13 is a Friday, so Monday is March 9
        assert result.date_range.start is not None
        assert result.date_range.start.day == 9

    def test_last_week(self, parser: QueryParser):
        result = parser.parse("what happened last week?")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.LAST_WEEK

    def test_this_month(self, parser: QueryParser):
        result = parser.parse("what did I do this month?")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.THIS_MONTH
        assert result.date_range.start is not None
        assert result.date_range.start.day == 1

    def test_last_month(self, parser: QueryParser):
        result = parser.parse("what happened last month?")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.LAST_MONTH
        assert result.date_range.start is not None
        assert result.date_range.start.month == 2

    def test_no_date_reference(self, parser: QueryParser):
        result = parser.parse("who is Sarah?")
        assert result.date_range is None

    def test_date_range_from_reference_custom(self):
        dr = DateRangeFilter.from_reference(DateReference.CUSTOM, FIXED_NOW)
        assert dr.reference == DateReference.CUSTOM
        assert dr.start is None
        assert dr.end is None

    def test_last_month_january_wraps_to_december(self):
        jan_now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        dr = DateRangeFilter.from_reference(DateReference.LAST_MONTH, jan_now)
        assert dr.start is not None
        assert dr.start.month == 12
        assert dr.start.year == 2025


class TestEntityTypeExtraction:
    """Tests for entity type filter extraction."""

    def test_person_from_who(self, parser: QueryParser):
        result = parser.parse("who do I work with?")
        assert result.entity_type_filter == EntityType.PERSON

    def test_place_from_where(self, parser: QueryParser):
        result = parser.parse("where do I usually work?")
        assert result.entity_type_filter == EntityType.PLACE

    def test_project_keyword(self, parser: QueryParser):
        result = parser.parse("list all projects")
        assert result.entity_type_filter == EntityType.PROJECT

    def test_organization_keyword(self, parser: QueryParser):
        result = parser.parse("what companies do I interact with?")
        assert result.entity_type_filter == EntityType.ORGANIZATION

    def test_no_entity_type(self, parser: QueryParser):
        result = parser.parse("what did I say about coffee?")
        assert result.entity_type_filter is None


class TestIntentFilter:
    """Tests for intent filter extraction."""

    def test_task_intent(self, parser: QueryParser):
        result = parser.parse("how many tasks do I have?")
        assert result.intent_filter == "task"

    def test_event_intent(self, parser: QueryParser):
        result = parser.parse("show me my events this week")
        assert result.intent_filter == "event"

    def test_meeting_maps_to_event(self, parser: QueryParser):
        result = parser.parse("what meetings do I have?")
        assert result.intent_filter == "event"

    def test_idea_intent(self, parser: QueryParser):
        result = parser.parse("show me my recent ideas")
        assert result.intent_filter == "idea"

    def test_no_intent(self, parser: QueryParser):
        result = parser.parse("who is Sarah?")
        assert result.intent_filter is None


class TestStructuredQuery:
    """Tests for StructuredQuery properties and serialization."""

    def test_has_entity_filter_with_names(self):
        q = StructuredQuery(entity_names=["Sarah"])
        assert q.has_entity_filter is True

    def test_has_entity_filter_with_type(self):
        q = StructuredQuery(entity_type_filter=EntityType.PERSON)
        assert q.has_entity_filter is True

    def test_has_entity_filter_empty(self):
        q = StructuredQuery()
        assert q.has_entity_filter is False

    def test_has_date_filter(self):
        q = StructuredQuery(date_range=DateRangeFilter(reference=DateReference.TODAY))
        assert q.has_date_filter is True

    def test_has_topic_filter(self):
        q = StructuredQuery(topics=["machine", "learning"])
        assert q.has_topic_filter is True

    def test_to_dict(self):
        q = StructuredQuery(
            query_type=QueryType.ENTITY_LOOKUP,
            raw_text="who is Sarah?",
            entity_names=["Sarah"],
        )
        d = q.to_dict()
        assert d["query_type"] == "entity_lookup"
        assert d["entity_names"] == ["Sarah"]
        assert d["raw_text"] == "who is Sarah?"

    def test_to_dict_with_date_range(self):
        dr = DateRangeFilter.from_reference(DateReference.TODAY, FIXED_NOW)
        q = StructuredQuery(date_range=dr)
        d = q.to_dict()
        assert "date_range" in d
        assert d["date_range"]["reference"] == "today"


# ── Executor Tests ────────────────────────────────────────────────────


class TestQueryExecutor:
    """Tests for the QueryExecutor."""

    @pytest.mark.asyncio
    async def test_entity_lookup(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("who is Sarah?")
        result = await executor.execute(query)

        assert result.has_results
        assert len(result.entities) >= 1
        assert any(e.name == "Sarah" for e in result.entities)
        # Should also have facts about Sarah
        assert len(result.facts) >= 1

    @pytest.mark.asyncio
    async def test_entity_lookup_not_found(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("who is Nonexistent Person?")
        result = await executor.execute(query)

        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_entity_list(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("what people do I know?")
        result = await executor.execute(query)

        assert result.has_results
        assert len(result.entities) >= 2  # Sarah and Jake
        # Should be sorted by mention count
        if len(result.entities) >= 2:
            assert result.entities[0].mention_count >= result.entities[1].mention_count

    @pytest.mark.asyncio
    async def test_entity_list_projects(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("list all projects")
        result = await executor.execute(query)

        assert result.has_results
        assert all(e.entity_type == EntityType.PROJECT for e in result.entities)

    @pytest.mark.asyncio
    async def test_fact_recall(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("when is Sarah's birthday?")
        result = await executor.execute(query)

        assert result.has_results
        assert any("birthday" in f.content.lower() for f in result.facts)

    @pytest.mark.asyncio
    async def test_timeline_query(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("what happened this week?")
        result = await executor.execute(query)

        # Should return episodes from this week
        assert isinstance(result.episodes, list)
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_relationship_query(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("how is Sarah connected to Blurt?")
        result = await executor.execute(query)

        assert len(result.entities) == 2
        assert len(result.relationships) >= 1

    @pytest.mark.asyncio
    async def test_relationship_not_enough_entities(
        self, executor: QueryExecutor
    ):
        query = StructuredQuery(
            query_type=QueryType.RELATIONSHIP,
            entity_names=["Sarah"],
            raw_text="how is Sarah connected to?",
        )
        result = await executor.execute(query)
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_count_query_entities(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("how many people do I know?")
        result = await executor.execute(query)

        assert result.has_results
        assert len(result.entities) >= 2  # Sarah and Jake

    @pytest.mark.asyncio
    async def test_count_query_by_intent(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("how many tasks do I have?")
        result = await executor.execute(query)

        # Should query episodic memory by intent
        assert isinstance(result.episodes, list)

    @pytest.mark.asyncio
    async def test_semantic_fallback(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("something about the quarterly report")
        result = await executor.execute(query)

        # Semantic search should return results
        assert isinstance(result.semantic_results, list)

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("who is Sarah?")
        result = await executor.execute(query)

        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_total_results_computed(self, executor: QueryExecutor, parser: QueryParser):
        query = parser.parse("who is Sarah?")
        result = await executor.execute(query)

        expected_total = (
            len(result.entities)
            + len(result.facts)
            + len(result.episodes)
            + len(result.relationships)
            + len(result.semantic_results)
        )
        assert result.total_results == expected_total


class TestQueryResult:
    """Tests for QueryResult serialization."""

    def test_to_dict(self):
        query = StructuredQuery(raw_text="test")
        result = QueryResult(
            query=query,
            total_results=0,
            execution_time_ms=1.5,
        )
        d = result.to_dict()
        assert d["total_results"] == 0
        assert d["execution_time_ms"] == 1.5
        assert "query" in d

    def test_has_results_false(self):
        result = QueryResult(query=StructuredQuery(), total_results=0)
        assert result.has_results is False

    def test_has_results_true(self):
        result = QueryResult(query=StructuredQuery(), total_results=5)
        assert result.has_results is True


# ── Integration-style tests ──────────────────────────────────────────


class TestParserExecutorIntegration:
    """End-to-end tests: parse then execute."""

    @pytest.mark.asyncio
    async def test_lookup_with_relationships(
        self, executor: QueryExecutor, parser: QueryParser
    ):
        """Entity lookup should return entity + facts + relationships."""
        query = parser.parse("tell me about Sarah")
        result = await executor.execute(query)

        assert len(result.entities) >= 1
        sarah = next(e for e in result.entities if e.name == "Sarah")
        assert sarah.entity_type == EntityType.PERSON
        assert len(result.facts) >= 1
        assert len(result.relationships) >= 1

    @pytest.mark.asyncio
    async def test_date_scoped_entity_list(
        self, executor: QueryExecutor, parser: QueryParser
    ):
        """Entity list with date range should filter by last_seen."""
        query = parser.parse("what people do I know this week?")
        result = await executor.execute(query)

        # All returned entities should have last_seen within this week
        if result.entities and query.date_range and query.date_range.start:
            for e in result.entities:
                assert e.last_seen >= query.date_range.start

    @pytest.mark.asyncio
    async def test_semantic_with_date_filter(
        self, executor: QueryExecutor, parser: QueryParser
    ):
        """Semantic queries with date references should also filter episodes."""
        query = parser.parse("anything interesting today?")
        result = await executor.execute(query)

        assert query.date_range is not None
        assert query.date_range.reference == DateReference.TODAY
        # Executor should use the date range
        assert isinstance(result.episodes, list)


# ── Tomorrow date reference tests ─────────────────────────────────────


class TestTomorrowDateReference:
    """Tests for the TOMORROW date reference."""

    def test_tomorrow_parsing(self, parser: QueryParser):
        result = parser.parse("what meetings do I have tomorrow?")
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.TOMORROW

    def test_tomorrow_date_range(self):
        dr = DateRangeFilter.from_reference(DateReference.TOMORROW, FIXED_NOW)
        assert dr.start is not None
        assert dr.end is not None
        # Tomorrow starts at midnight of the next day
        today_start = FIXED_NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        expected_start = today_start + timedelta(days=1)
        assert dr.start == expected_start
        assert dr.end == expected_start + timedelta(days=1)

    def test_tomorrow_is_full_day(self):
        dr = DateRangeFilter.from_reference(DateReference.TOMORROW, FIXED_NOW)
        assert dr.start is not None
        assert dr.end is not None
        assert (dr.end - dr.start).total_seconds() == 86400


# ── Entity-topic query pattern tests ──────────────────────────────────


class TestEntityTopicQueries:
    """Tests for 'Who did I talk to about X?' patterns."""

    def test_who_did_i_talk_to_about(self, parser: QueryParser):
        result = parser.parse("who did I talk to about the budget?")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.PERSON
        assert len(result.topics) > 0
        assert "budget" in result.topics

    def test_who_did_i_speak_with_about(self, parser: QueryParser):
        result = parser.parse("who did I speak with about project alpha?")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.PERSON
        assert any("project" in t or "alpha" in t for t in result.topics)

    def test_who_did_i_meet_with_about(self, parser: QueryParser):
        result = parser.parse("who did I meet with about hiring?")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.PERSON
        assert "hiring" in result.topics

    def test_who_mentioned(self, parser: QueryParser):
        result = parser.parse("who mentioned the deadline?")
        assert result.query_type == QueryType.ENTITY_LIST
        assert result.entity_type_filter == EntityType.PERSON
        assert "deadline" in result.topics


# ── Intent-temporal query pattern tests ───────────────────────────────


class TestIntentTemporalQueries:
    """Tests for 'What meetings do I have tomorrow?' patterns."""

    def test_what_meetings_tomorrow(self, parser: QueryParser):
        result = parser.parse("what meetings do I have tomorrow?")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "event"
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.TOMORROW

    def test_what_tasks_today(self, parser: QueryParser):
        result = parser.parse("what tasks do I have today?")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "task"
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.TODAY

    def test_what_events_this_week(self, parser: QueryParser):
        result = parser.parse("what events do I have this week?")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "event"
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.THIS_WEEK

    def test_do_i_have_any_meetings_tomorrow(self, parser: QueryParser):
        result = parser.parse("do I have any meetings tomorrow?")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "event"
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.TOMORROW

    def test_show_me_my_reminders(self, parser: QueryParser):
        result = parser.parse("show me my reminders")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "reminder"

    def test_list_my_tasks(self, parser: QueryParser):
        result = parser.parse("list my tasks")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "task"

    def test_intent_temporal_defaults_to_today(self, parser: QueryParser):
        """When no date is specified, defaults to today."""
        result = parser.parse("show me my meetings")
        assert result.query_type == QueryType.TIMELINE
        assert result.intent_filter == "event"
        assert result.date_range is not None
        assert result.date_range.reference == DateReference.TODAY


# ── Response Formatter tests ──────────────────────────────────────────


class TestResponseFormatter:
    """Tests for the ResponseFormatter."""

    def test_format_empty_result(self):
        query = StructuredQuery(raw_text="who is Nobody?", query_type=QueryType.ENTITY_LOOKUP)
        result = QueryResult(query=query, total_results=0)
        response = ResponseFormatter.format(result)

        assert response["answer_summary"] != ""
        assert response["metadata"]["has_results"] is False
        assert response["metadata"]["total_results"] == 0
        assert response["items"] == []

    def test_format_entity_lookup(self):
        entity = _make_entity("Sarah", EntityType.PERSON, mention_count=10)
        fact = _make_fact("Sarah is my manager", entity.id)
        query = StructuredQuery(
            raw_text="who is Sarah?",
            query_type=QueryType.ENTITY_LOOKUP,
            entity_names=["Sarah"],
        )
        result = QueryResult(
            query=query,
            entities=[entity],
            facts=[fact],
            total_results=2,
        )
        response = ResponseFormatter.format(result)

        assert "Sarah" in response["answer_summary"]
        assert "person" in response["answer_summary"]
        assert len(response["items"]) == 2
        assert response["items"][0]["type"] == "entity"
        assert response["items"][1]["type"] == "fact"

    def test_format_entity_list(self):
        entities = [
            _make_entity("Sarah", EntityType.PERSON, mention_count=10),
            _make_entity("Jake", EntityType.PERSON, mention_count=5),
        ]
        query = StructuredQuery(
            raw_text="who do I know?",
            query_type=QueryType.ENTITY_LIST,
            entity_type_filter=EntityType.PERSON,
        )
        result = QueryResult(
            query=query,
            entities=entities,
            total_results=2,
        )
        response = ResponseFormatter.format(result)

        assert "2" in response["answer_summary"]
        assert "person" in response["answer_summary"]
        assert "Sarah" in response["answer_summary"]
        assert "Jake" in response["answer_summary"]

    def test_format_fact_recall(self):
        entity = _make_entity("Sarah", EntityType.PERSON)
        fact = _make_fact("Sarah's birthday is March 15", entity.id)
        query = StructuredQuery(
            raw_text="when is Sarah's birthday?",
            query_type=QueryType.FACT_RECALL,
            entity_names=["Sarah"],
        )
        result = QueryResult(
            query=query,
            entities=[entity],
            facts=[fact],
            total_results=2,
        )
        response = ResponseFormatter.format(result)

        assert "birthday" in response["answer_summary"].lower()

    def test_format_timeline_with_episodes(self):
        episodes = [
            _make_episode("Meeting with Sarah", intent="event", timestamp=FIXED_NOW),
            _make_episode("Finish quarterly report", intent="task", timestamp=FIXED_NOW),
        ]
        query = StructuredQuery(
            raw_text="what happened today?",
            query_type=QueryType.TIMELINE,
            date_range=DateRangeFilter.from_reference(DateReference.TODAY, FIXED_NOW),
        )
        result = QueryResult(
            query=query,
            episodes=episodes,
            total_results=2,
        )
        response = ResponseFormatter.format(result)

        assert "2 items" in response["answer_summary"]
        assert "today" in response["answer_summary"]
        assert len(response["items"]) == 2
        assert response["items"][0]["type"] == "episode"

    def test_format_timeline_empty(self):
        query = StructuredQuery(
            raw_text="what happened tomorrow?",
            query_type=QueryType.TIMELINE,
            date_range=DateRangeFilter.from_reference(DateReference.TOMORROW, FIXED_NOW),
        )
        result = QueryResult(query=query, total_results=0)
        response = ResponseFormatter.format(result)

        assert "tomorrow" in response["answer_summary"]

    def test_format_relationship(self):
        entity_a = _make_entity("Sarah", EntityType.PERSON)
        entity_b = _make_entity("Blurt", EntityType.PROJECT)
        rel = _make_relationship(entity_a.id, entity_b.id, RelationshipType.MANAGES)
        query = StructuredQuery(
            raw_text="how is Sarah connected to Blurt?",
            query_type=QueryType.RELATIONSHIP,
            entity_names=["Sarah", "Blurt"],
        )
        result = QueryResult(
            query=query,
            entities=[entity_a, entity_b],
            relationships=[rel],
            total_results=3,
        )
        response = ResponseFormatter.format(result)

        assert "Sarah" in response["answer_summary"]
        assert "Blurt" in response["answer_summary"]
        assert "connected" in response["answer_summary"]

    def test_format_count_entities(self):
        entities = [
            _make_entity("Sarah", EntityType.PERSON),
            _make_entity("Jake", EntityType.PERSON),
            _make_entity("Lisa", EntityType.PERSON),
        ]
        query = StructuredQuery(
            raw_text="how many people do I know?",
            query_type=QueryType.COUNT,
            entity_type_filter=EntityType.PERSON,
        )
        result = QueryResult(
            query=query,
            entities=entities,
            total_results=3,
        )
        response = ResponseFormatter.format(result)

        assert "3" in response["answer_summary"]
        assert "person" in response["answer_summary"]

    def test_format_count_episodes(self):
        episodes = [
            _make_episode("Do the thing", intent="task"),
        ]
        query = StructuredQuery(
            raw_text="how many tasks do I have?",
            query_type=QueryType.COUNT,
            intent_filter="task",
        )
        result = QueryResult(
            query=query,
            episodes=episodes,
            total_results=1,
        )
        response = ResponseFormatter.format(result)

        assert "1" in response["answer_summary"]
        assert "task" in response["answer_summary"]

    def test_format_semantic_results(self):
        query = StructuredQuery(
            raw_text="coffee habits",
            query_type=QueryType.SEMANTIC,
        )
        result = QueryResult(
            query=query,
            semantic_results=[
                SemanticSearchResult(
                    item_type="fact",
                    item_id="f1",
                    content="I prefer oat milk lattes",
                    similarity_score=0.85,
                ),
            ],
            total_results=1,
        )
        response = ResponseFormatter.format(result)

        assert "1 related item" in response["answer_summary"]
        assert response["items"][0]["type"] == "semantic_match"

    def test_format_metadata(self):
        query = StructuredQuery(raw_text="test", query_type=QueryType.SEMANTIC)
        result = QueryResult(
            query=query,
            total_results=0,
            execution_time_ms=12.345,
        )
        response = ResponseFormatter.format(result)

        assert response["metadata"]["execution_time_ms"] == 12.3
        assert response["query_info"]["query_type"] == "semantic"
        assert response["query_info"]["raw_text"] == "test"

    def test_format_no_results_entity(self):
        query = StructuredQuery(
            raw_text="who is Bob?",
            query_type=QueryType.ENTITY_LOOKUP,
            entity_names=["Bob"],
        )
        result = QueryResult(query=query, total_results=0)
        response = ResponseFormatter.format(result)

        assert "Bob" in response["answer_summary"]

    def test_format_no_results_intent(self):
        query = StructuredQuery(
            raw_text="what meetings do I have tomorrow?",
            query_type=QueryType.TIMELINE,
            intent_filter="event",
            date_range=DateRangeFilter.from_reference(DateReference.TOMORROW, FIXED_NOW),
        )
        result = QueryResult(query=query, total_results=0)
        response = ResponseFormatter.format(result)

        assert "event" in response["answer_summary"]
        assert "tomorrow" in response["answer_summary"]


class TestResponseFormatterAntiShame:
    """Ensure response formatter never produces shame language."""

    SHAME_WORDS = [
        "overdue", "behind", "missed", "failed", "lazy",
        "you should", "hurry", "penalty", "streak", "guilt",
        "forgot", "neglected", "shame",
    ]

    def _check_no_shame(self, text: str) -> None:
        lower = text.lower()
        for word in self.SHAME_WORDS:
            assert word not in lower, f"Shame word '{word}' found in: {text}"

    def test_no_shame_in_empty_results(self):
        for qt in QueryType:
            query = StructuredQuery(raw_text="test", query_type=qt)
            result = QueryResult(query=query, total_results=0)
            response = ResponseFormatter.format(result)
            self._check_no_shame(response["answer_summary"])

    def test_no_shame_in_timeline_summaries(self):
        episodes = [
            _make_episode("Something happened", intent="task"),
        ]
        for ref in DateReference:
            dr = DateRangeFilter.from_reference(ref, FIXED_NOW)
            query = StructuredQuery(
                raw_text="test",
                query_type=QueryType.TIMELINE,
                date_range=dr,
            )
            result = QueryResult(query=query, episodes=episodes, total_results=1)
            response = ResponseFormatter.format(result)
            self._check_no_shame(response["answer_summary"])

    def test_no_shame_in_count_zero(self):
        query = StructuredQuery(
            raw_text="how many tasks?",
            query_type=QueryType.COUNT,
            intent_filter="task",
        )
        result = QueryResult(query=query, total_results=0)
        response = ResponseFormatter.format(result)
        self._check_no_shame(response["answer_summary"])


# ── Full pipeline integration tests ──────────────────────────────────


class TestFullPipelineIntegration:
    """End-to-end: parse → execute → format."""

    @pytest.mark.asyncio
    async def test_meetings_tomorrow(self, executor: QueryExecutor, parser: QueryParser):
        """'What meetings do I have tomorrow?' should parse, execute, and format."""
        query = parser.parse("what meetings do I have tomorrow?")

        assert query.query_type == QueryType.TIMELINE
        assert query.intent_filter == "event"
        assert query.date_range is not None
        assert query.date_range.reference == DateReference.TOMORROW

        result = await executor.execute(query)
        response = ResponseFormatter.format(result)

        assert "answer_summary" in response
        assert "items" in response
        assert response["metadata"]["has_results"] is not None

    @pytest.mark.asyncio
    async def test_who_talked_about_topic(self, executor: QueryExecutor, parser: QueryParser):
        """'Who did I talk to about X?' should parse and execute."""
        query = parser.parse("who did I talk to about the quarterly report?")

        assert query.query_type == QueryType.ENTITY_LIST
        assert query.entity_type_filter == EntityType.PERSON
        assert len(query.topics) > 0

        result = await executor.execute(query)
        response = ResponseFormatter.format(result)

        assert "answer_summary" in response
        assert response["query_info"]["query_type"] == "entity_list"

    @pytest.mark.asyncio
    async def test_entity_lookup_full_pipeline(
        self, executor: QueryExecutor, parser: QueryParser
    ):
        """Entity lookup: parse → execute → format with facts + relationships."""
        query = parser.parse("tell me about Sarah")
        result = await executor.execute(query)
        response = ResponseFormatter.format(result)

        assert response["metadata"]["has_results"] is True
        assert "Sarah" in response["answer_summary"]
        # Should contain entity and fact items
        item_types = {item["type"] for item in response["items"]}
        assert "entity" in item_types
        assert "fact" in item_types

    @pytest.mark.asyncio
    async def test_count_query_full_pipeline(
        self, executor: QueryExecutor, parser: QueryParser
    ):
        """Count query: parse → execute → format."""
        query = parser.parse("how many people do I know?")
        result = await executor.execute(query)
        response = ResponseFormatter.format(result)

        assert response["metadata"]["has_results"] is True
        assert "person" in response["answer_summary"]

    @pytest.mark.asyncio
    async def test_relationship_full_pipeline(
        self, executor: QueryExecutor, parser: QueryParser
    ):
        """Relationship query: parse → execute → format."""
        query = parser.parse("how is Sarah connected to Blurt?")
        result = await executor.execute(query)
        response = ResponseFormatter.format(result)

        assert response["metadata"]["has_results"] is True
        assert "Sarah" in response["answer_summary"]
        assert "Blurt" in response["answer_summary"]
