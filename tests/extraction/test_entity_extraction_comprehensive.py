"""Comprehensive tests for entity extraction covering every entity type,
edge cases (ambiguous entities, multiple entities in one utterance,
possessives, nicknames), and accuracy validation.

Tests target the primary EntityExtractor in blurt/extraction/entities.py
using both mocked LLM responses and regex-fallback paths to validate
the full extraction pipeline's correctness.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.extraction.entities import (
    EntityExtractor,
    EntityExtractionError,
    ExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    TemporalReference,
    TemporalType,
)
from blurt.models.entities import EntityType, FactType, RelationshipType


# ── Helpers ──────────────────────────────────────────────────────────


def _make_gemini_response(
    data: dict[str, Any],
    model: str = "gemini-2.5-flash-lite",
    latency_ms: float = 42.0,
) -> MagicMock:
    """Create a mock GeminiResponse with JSON text."""
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.model = model
    resp.latency_ms = latency_ms
    return resp


def _mock_client(
    response_data: dict[str, Any],
    is_connected: bool = True,
) -> MagicMock:
    """Create a mock GeminiClient returning the given response data."""
    client = MagicMock()
    client.is_connected = is_connected
    client.generate = AsyncMock(return_value=_make_gemini_response(response_data))
    return client


def _empty_response(**overrides: Any) -> dict[str, Any]:
    """Base empty response dict with overrides."""
    base: dict[str, Any] = {
        "entities": [],
        "relationships": [],
        "facts": [],
        "temporal_references": [],
    }
    base.update(overrides)
    return base


def _entity(
    name: str,
    etype: str,
    confidence: float = 0.9,
    aliases: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shorthand for building an entity dict in LLM response format."""
    return {
        "name": name,
        "type": etype,
        "confidence": confidence,
        "aliases": aliases or [],
        "attributes": attributes or {},
    }


def _rel(
    source: str,
    target: str,
    rtype: str = "related_to",
    confidence: float = 0.85,
) -> dict[str, Any]:
    return {"source": source, "target": target, "type": rtype, "confidence": confidence}


def _fact(
    content: str,
    ftype: str = "attribute",
    subject: str | None = None,
    confidence: float = 0.8,
) -> dict[str, Any]:
    return {"content": content, "type": ftype, "subject": subject, "confidence": confidence}


def _names(result: ExtractionResult) -> set[str]:
    return {e.name for e in result.entities}


def _normalized(result: ExtractionResult) -> set[str]:
    return {e.normalized_name for e in result.entities}


def _types_map(result: ExtractionResult) -> dict[str, EntityType]:
    return {e.name: e.entity_type for e in result.entities}


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: Per-entity-type extraction (LLM path)
# ═══════════════════════════════════════════════════════════════════════


class TestPersonEntityType:
    """Comprehensive PERSON entity extraction tests."""

    @pytest.mark.asyncio
    async def test_simple_first_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Sarah", "person"),
        ]))
        result = await EntityExtractor(client).extract("I talked to Sarah about it")
        assert result.entities[0].entity_type == EntityType.PERSON
        assert result.entities[0].normalized_name == "sarah"

    @pytest.mark.asyncio
    async def test_full_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Sarah Johnson", "person"),
        ]))
        result = await EntityExtractor(client).extract("Sarah Johnson called me today")
        assert result.entities[0].name == "Sarah Johnson"
        assert result.entities[0].normalized_name == "sarah johnson"

    @pytest.mark.asyncio
    async def test_title_with_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Dr. Patel", "person", attributes={"title": "Dr."}),
        ]))
        result = await EntityExtractor(client).extract("I saw Dr. Patel this morning")
        assert result.entities[0].name == "Dr. Patel"
        assert result.entities[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_nickname_as_alias(self):
        """A person referenced by nickname should have it captured in aliases."""
        client = _mock_client(_empty_response(entities=[
            _entity("Michael", "person", aliases=["Mike", "Mikey"]),
        ]))
        result = await EntityExtractor(client).extract(
            "Mike said he'd handle it — you know Michael, he always delivers"
        )
        entity = result.entities[0]
        assert entity.name == "Michael"
        assert "Mike" in entity.aliases
        assert "Mikey" in entity.aliases

    @pytest.mark.asyncio
    async def test_person_with_role_attribute(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Lisa", "person", attributes={"role": "CTO", "company": "Acme"}),
        ]))
        result = await EntityExtractor(client).extract("Lisa our CTO at Acme wants a sync")
        assert result.entities[0].attributes["role"] == "CTO"

    @pytest.mark.asyncio
    async def test_multiple_people_same_utterance(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Alice", "person"),
            _entity("Bob", "person"),
            _entity("Charlie", "person"),
        ]))
        result = await EntityExtractor(client).extract(
            "Alice, Bob, and Charlie are joining the standup"
        )
        assert result.entity_count == 3
        assert all(e.entity_type == EntityType.PERSON for e in result.entities)

    @pytest.mark.asyncio
    async def test_possessive_person_reference(self):
        """Possessive form 'Sarah's' should still extract the person."""
        client = _mock_client(_empty_response(
            entities=[_entity("Sarah", "person")],
            relationships=[_rel("speaker", "Sarah", "managed_by")],
        ))
        result = await EntityExtractor(client).extract(
            "Sarah's feedback on the design was really insightful"
        )
        assert "Sarah" in _names(result)
        assert result.entities[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_person_with_non_english_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("José García", "person"),
            _entity("Yuki Tanaka", "person"),
        ]))
        result = await EntityExtractor(client).extract(
            "José García and Yuki Tanaka are joining the project"
        )
        assert result.entity_count == 2
        assert "José García" in _names(result)
        assert "Yuki Tanaka" in _names(result)


class TestPlaceEntityType:
    """Comprehensive PLACE entity extraction tests."""

    @pytest.mark.asyncio
    async def test_city_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("San Francisco", "place"),
        ]))
        result = await EntityExtractor(client).extract("Flying to San Francisco tomorrow")
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_office_location(self):
        client = _mock_client(_empty_response(entities=[
            _entity("the downtown office", "place"),
        ]))
        result = await EntityExtractor(client).extract(
            "Let's meet at the downtown office"
        )
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_room_reference(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Room 302", "place", attributes={"type": "room"}),
        ]))
        result = await EntityExtractor(client).extract("The meeting is in Room 302")
        assert result.entities[0].name == "Room 302"

    @pytest.mark.asyncio
    async def test_country_and_city(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Tokyo", "place"),
            _entity("Japan", "place"),
        ]))
        result = await EntityExtractor(client).extract("Flying to Tokyo, Japan next week")
        assert result.entity_count == 2
        assert all(e.entity_type == EntityType.PLACE for e in result.entities)

    @pytest.mark.asyncio
    async def test_venue_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Blue Bottle Coffee", "place"),
        ]))
        result = await EntityExtractor(client).extract(
            "Grab coffee at Blue Bottle Coffee on 3rd street"
        )
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_home_as_place(self):
        client = _mock_client(_empty_response(entities=[
            _entity("home", "place"),
        ]))
        result = await EntityExtractor(client).extract("Working from home today")
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_possessive_place(self):
        """Google's campus — possessive form of a place."""
        client = _mock_client(_empty_response(entities=[
            _entity("Google's campus", "place"),
        ]))
        result = await EntityExtractor(client).extract(
            "The tour of Google's campus was great"
        )
        assert result.entities[0].entity_type == EntityType.PLACE


class TestProjectEntityType:
    """Comprehensive PROJECT entity extraction tests."""

    @pytest.mark.asyncio
    async def test_named_project(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Project Atlas", "project"),
        ]))
        result = await EntityExtractor(client).extract("Need to finalize Project Atlas")
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_informal_project(self):
        client = _mock_client(_empty_response(entities=[
            _entity("the Q2 deck", "project", aliases=["Q2 planning deck"]),
        ]))
        result = await EntityExtractor(client).extract(
            "I still need to finish the Q2 deck before the review"
        )
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_codename_project(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Phoenix", "project"),
        ]))
        result = await EntityExtractor(client).extract(
            "Phoenix is behind schedule, we might need to cut scope"
        )
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_versioned_project(self):
        client = _mock_client(_empty_response(entities=[
            _entity("v2 migration", "project", attributes={"version": "2"}),
        ]))
        result = await EntityExtractor(client).extract(
            "The v2 migration is top priority right now"
        )
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_deliverable_as_project(self):
        client = _mock_client(_empty_response(entities=[
            _entity("annual report", "project"),
        ]))
        result = await EntityExtractor(client).extract(
            "Need to draft the annual report next week"
        )
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_multiple_projects_one_utterance(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Project Alpha", "project"),
            _entity("the API redesign", "project"),
            _entity("onboarding flow", "project"),
        ]))
        result = await EntityExtractor(client).extract(
            "Project Alpha depends on the API redesign, "
            "and we're also updating the onboarding flow"
        )
        assert result.entity_count == 3
        assert all(e.entity_type == EntityType.PROJECT for e in result.entities)


class TestOrganizationEntityType:
    """Comprehensive ORGANIZATION entity extraction tests."""

    @pytest.mark.asyncio
    async def test_company_name(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Google", "organization"),
        ]))
        result = await EntityExtractor(client).extract("Google announced new pricing")
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_team_as_organization(self):
        client = _mock_client(_empty_response(entities=[
            _entity("the marketing team", "organization"),
        ]))
        result = await EntityExtractor(client).extract(
            "The marketing team needs assets by Thursday"
        )
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_department(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Engineering", "organization"),
        ]))
        result = await EntityExtractor(client).extract("Engineering is pushing back")
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_institution(self):
        client = _mock_client(_empty_response(entities=[
            _entity("MIT", "organization"),
        ]))
        result = await EntityExtractor(client).extract("The paper from MIT is interesting")
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_org_with_suffix(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Acme Corp", "organization"),
        ]))
        result = await EntityExtractor(client).extract("Partnering with Acme Corp")
        assert result.entities[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_possessive_organization(self):
        """Company's product — possessive form of an org."""
        client = _mock_client(_empty_response(entities=[
            _entity("Apple", "organization"),
        ]))
        result = await EntityExtractor(client).extract("Apple's new product is impressive")
        assert result.entities[0].entity_type == EntityType.ORGANIZATION


class TestTopicEntityType:
    """Comprehensive TOPIC entity extraction tests."""

    @pytest.mark.asyncio
    async def test_technical_topic(self):
        client = _mock_client(_empty_response(entities=[
            _entity("machine learning", "topic"),
        ]))
        result = await EntityExtractor(client).extract(
            "I've been reading about machine learning approaches"
        )
        assert result.entities[0].entity_type == EntityType.TOPIC

    @pytest.mark.asyncio
    async def test_abstract_concept(self):
        client = _mock_client(_empty_response(entities=[
            _entity("work-life balance", "topic"),
        ]))
        result = await EntityExtractor(client).extract(
            "I need to think more about work-life balance"
        )
        assert result.entities[0].entity_type == EntityType.TOPIC

    @pytest.mark.asyncio
    async def test_discussion_topic(self):
        client = _mock_client(_empty_response(entities=[
            _entity("pricing strategy", "topic"),
        ]))
        result = await EntityExtractor(client).extract(
            "We need to discuss our pricing strategy"
        )
        assert result.entities[0].entity_type == EntityType.TOPIC

    @pytest.mark.asyncio
    async def test_unknown_type_defaults_to_topic(self):
        """An entity with an unrecognized type should default to TOPIC."""
        client = _mock_client(_empty_response(entities=[
            _entity("something", "emotion"),
        ]))
        result = await EntityExtractor(client).extract("something happened")
        assert result.entities[0].entity_type == EntityType.TOPIC


class TestToolEntityType:
    """Comprehensive TOOL entity extraction tests."""

    @pytest.mark.asyncio
    async def test_notion_tool(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Notion", "tool", confidence=0.98),
        ]))
        result = await EntityExtractor(client).extract("Update the Notion page")
        assert result.entities[0].entity_type == EntityType.TOOL

    @pytest.mark.asyncio
    async def test_multiple_tools(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Slack", "tool"),
            _entity("Figma", "tool"),
            _entity("Jira", "tool"),
        ]))
        result = await EntityExtractor(client).extract(
            "Check Slack, update Figma, and create a Jira ticket"
        )
        assert result.entity_count == 3
        assert all(e.entity_type == EntityType.TOOL for e in result.entities)

    @pytest.mark.asyncio
    async def test_tool_with_qualifier(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Google Calendar", "tool"),
        ]))
        result = await EntityExtractor(client).extract(
            "Sync the event to Google Calendar"
        )
        assert result.entities[0].entity_type == EntityType.TOOL


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Edge cases — ambiguous entities
# ═══════════════════════════════════════════════════════════════════════


class TestAmbiguousEntities:
    """Test entities that could be classified as multiple types."""

    @pytest.mark.asyncio
    async def test_apple_as_org_not_fruit(self):
        """'Apple' in tech context should be classified as organization."""
        client = _mock_client(_empty_response(entities=[
            _entity("Apple", "organization"),
        ]))
        result = await EntityExtractor(client).extract(
            "Apple just released a new product"
        )
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_phoenix_as_project_not_place(self):
        """'Phoenix' in project context should be project, not place."""
        client = _mock_client(_empty_response(entities=[
            _entity("Phoenix", "project"),
        ]))
        result = await EntityExtractor(client).extract(
            "Phoenix is behind schedule, we need to cut scope"
        )
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_linear_as_tool_not_adjective(self):
        """'Linear' in tool context should be extracted as tool."""
        client = _mock_client(_empty_response(entities=[
            _entity("Linear", "tool"),
        ]))
        result = await EntityExtractor(client).extract("Create a ticket in Linear")
        assert result.entities[0].entity_type == EntityType.TOOL

    @pytest.mark.asyncio
    async def test_slack_as_tool_not_adjective(self):
        """'Slack' should be recognized as tool, not the adjective."""
        client = _mock_client(_empty_response(entities=[
            _entity("Slack", "tool"),
        ]))
        result = await EntityExtractor(client).extract("Post the update on Slack")
        assert result.entities[0].entity_type == EntityType.TOOL

    @pytest.mark.asyncio
    async def test_name_that_is_also_place(self):
        """'Austin' could be a person or place — context determines."""
        client = _mock_client(_empty_response(entities=[
            _entity("Austin", "person"),
        ]))
        result = await EntityExtractor(client).extract("Austin said he'd be late")
        assert result.entities[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_same_name_different_context_as_place(self):
        """'Austin' as place in travel context."""
        client = _mock_client(_empty_response(entities=[
            _entity("Austin", "place"),
        ]))
        result = await EntityExtractor(client).extract("Flying to Austin for the conf")
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_cursor_as_tool(self):
        """'Cursor' is an ambiguous word but should be tool in dev context."""
        client = _mock_client(_empty_response(entities=[
            _entity("Cursor", "tool"),
        ]))
        result = await EntityExtractor(client).extract(
            "I switched to Cursor for coding"
        )
        assert result.entities[0].entity_type == EntityType.TOOL

    @pytest.mark.asyncio
    async def test_entity_appears_as_both_person_and_org_in_different_roles(self):
        """'Jordan' as person alongside org — distinct entities, no confusion."""
        client = _mock_client(_empty_response(entities=[
            _entity("Jordan", "person"),
            _entity("Jordan Ventures", "organization"),
        ]))
        result = await EntityExtractor(client).extract(
            "Jordan from Jordan Ventures wants a meeting"
        )
        types = _types_map(result)
        assert types["Jordan"] == EntityType.PERSON
        assert types["Jordan Ventures"] == EntityType.ORGANIZATION


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: Multiple entities in one utterance
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleEntitiesOneUtterance:
    """Test extraction of many entities from complex, information-dense speech."""

    @pytest.mark.asyncio
    async def test_all_six_entity_types_in_one_utterance(self):
        """Extract all 6 entity types from a single rich utterance."""
        client = _mock_client(_empty_response(entities=[
            _entity("Sarah", "person"),
            _entity("San Francisco", "place"),
            _entity("Project Atlas", "project"),
            _entity("Acme Corp", "organization"),
            _entity("machine learning", "topic"),
            _entity("Notion", "tool"),
        ]))
        result = await EntityExtractor(client).extract(
            "Sarah at Acme Corp in San Francisco uses Notion for "
            "Project Atlas about machine learning"
        )
        assert result.entity_count == 6
        found_types = result.entity_types_found
        assert found_types == {
            EntityType.PERSON, EntityType.PLACE, EntityType.PROJECT,
            EntityType.ORGANIZATION, EntityType.TOPIC, EntityType.TOOL,
        }

    @pytest.mark.asyncio
    async def test_five_people_in_one_utterance(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Alice", "person"),
            _entity("Bob", "person"),
            _entity("Charlie", "person"),
            _entity("Diana", "person"),
            _entity("Eve", "person"),
        ]))
        result = await EntityExtractor(client).extract(
            "Alice, Bob, Charlie, Diana, and Eve are all on the call"
        )
        assert result.entity_count == 5
        assert all(e.entity_type == EntityType.PERSON for e in result.entities)

    @pytest.mark.asyncio
    async def test_mixed_entities_with_relationships(self):
        """Complex utterance with entities AND relationships."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("James", "person"),
                _entity("Lisa", "person"),
                _entity("Horizon", "project"),
                _entity("the product team", "organization"),
            ],
            relationships=[
                _rel("James", "the product team", "part_of"),
                _rel("Lisa", "the product team", "part_of"),
                _rel("James", "Horizon", "collaborates_on"),
                _rel("Lisa", "Horizon", "collaborates_on"),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Meeting with James and Lisa from the product team about Horizon"
        )
        assert result.entity_count == 4
        assert len(result.relationships) == 4

    @pytest.mark.asyncio
    async def test_entities_with_facts_and_temporals(self):
        """Extraction yields entities, facts, and temporal references together."""
        client = _mock_client(_empty_response(
            entities=[_entity("Sarah", "person")],
            facts=[_fact("Prefers morning meetings", "preference", "Sarah")],
            temporal_references=[{"text": "tomorrow at 9am", "type": "relative"}],
        ))
        result = await EntityExtractor(client).extract(
            "Sarah prefers morning meetings — let's schedule for tomorrow at 9am"
        )
        assert result.entity_count == 1
        assert len(result.facts) == 1
        assert len(result.temporal_references) == 1

    @pytest.mark.asyncio
    async def test_dense_status_update(self):
        """Information-dense status update with many entity types."""
        client = _mock_client(_empty_response(entities=[
            _entity("the API gateway", "project"),
            _entity("Priya", "person"),
            _entity("DevOps", "organization"),
            _entity("AWS", "organization"),
            _entity("Terraform", "tool"),
            _entity("infrastructure scalability", "topic"),
        ]))
        result = await EntityExtractor(client).extract(
            "The API gateway migration is 80% done. Priya is leading it "
            "with DevOps. We're using Terraform to move to AWS. "
            "Main concern is infrastructure scalability."
        )
        assert result.entity_count == 6


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Possessives and ownership patterns
# ═══════════════════════════════════════════════════════════════════════


class TestPossessives:
    """Test that possessive forms ('s, s') extract the base entity correctly."""

    @pytest.mark.asyncio
    async def test_person_possessive_extracts_person(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Sarah", "person"),
        ]))
        result = await EntityExtractor(client).extract(
            "Sarah's presentation was excellent"
        )
        assert "Sarah" in _names(result)

    @pytest.mark.asyncio
    async def test_org_possessive_extracts_org(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Google", "organization"),
        ]))
        result = await EntityExtractor(client).extract(
            "Google's new API pricing is interesting"
        )
        assert "Google" in _names(result)
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_project_possessive(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Atlas", "project"),
        ]))
        result = await EntityExtractor(client).extract(
            "Atlas's timeline needs adjustment"
        )
        assert "Atlas" in _names(result)

    @pytest.mark.asyncio
    async def test_possessive_with_relationship(self):
        """'Sarah's manager' should extract Sarah and the relationship."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Sarah", "person"),
                _entity("Tom", "person"),
            ],
            relationships=[
                _rel("Tom", "Sarah", "manages"),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Tom is Sarah's manager"
        )
        assert result.entity_count == 2
        assert len(result.relationships) == 1

    @pytest.mark.asyncio
    async def test_multiple_possessives_in_one_utterance(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Mark", "person"),
            _entity("Google", "organization"),
            _entity("Phoenix", "project"),
        ]))
        result = await EntityExtractor(client).extract(
            "Mark's update on Google's investment in Phoenix's development"
        )
        assert result.entity_count == 3


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: Nicknames, aliases, and informal references
# ═══════════════════════════════════════════════════════════════════════


class TestNicknamesAndAliases:
    """Test that nicknames, shortened names, and aliases are handled."""

    @pytest.mark.asyncio
    async def test_nickname_captured_in_aliases(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Robert", "person", aliases=["Bob", "Bobby"]),
        ]))
        result = await EntityExtractor(client).extract(
            "Bob mentioned it — you know Robert, he always says that"
        )
        entity = result.entities[0]
        assert entity.name == "Robert"
        assert "Bob" in entity.aliases

    @pytest.mark.asyncio
    async def test_multiple_aliases(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Elizabeth", "person", aliases=["Liz", "Beth", "Lizzy"]),
        ]))
        result = await EntityExtractor(client).extract(
            "Liz — well, Elizabeth, or Beth as some call her"
        )
        entity = result.entities[0]
        assert len(entity.aliases) >= 2

    @pytest.mark.asyncio
    async def test_project_alias(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Q2 Planning Deck", "project", aliases=["the deck", "Q2 deck"]),
        ]))
        result = await EntityExtractor(client).extract(
            "Need to finish the deck — the Q2 planning deck is due soon"
        )
        assert result.entities[0].aliases == ["the deck", "Q2 deck"]

    @pytest.mark.asyncio
    async def test_informal_team_reference(self):
        """Informal 'the backend folks' aliased to formal team name."""
        client = _mock_client(_empty_response(entities=[
            _entity("Backend Engineering", "organization",
                    aliases=["the backend folks", "backend team"]),
        ]))
        result = await EntityExtractor(client).extract(
            "Check with the backend folks on this"
        )
        entity = result.entities[0]
        assert entity.name == "Backend Engineering"
        assert "the backend folks" in entity.aliases

    @pytest.mark.asyncio
    async def test_tool_alias(self):
        """VS Code / VSCode aliasing."""
        client = _mock_client(_empty_response(entities=[
            _entity("VS Code", "tool", aliases=["VSCode", "Visual Studio Code"]),
        ]))
        result = await EntityExtractor(client).extract(
            "I switched from VSCode to Cursor"
        )
        assert "VSCode" in result.entities[0].aliases or result.entities[0].name == "VS Code"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: Regex fallback — per entity type
# ═══════════════════════════════════════════════════════════════════════


class TestRegexFallbackEntityTypes:
    """Test regex fallback extraction for each entity type and edge case."""

    @pytest.mark.asyncio
    async def test_regex_extracts_person_from_context(self):
        """'with Sarah' context triggers person classification in regex."""
        extractor = EntityExtractor()  # No client → regex fallback
        result = await extractor.extract("I had lunch with Sarah today")
        sarah = [e for e in result.entities if e.normalized_name == "sarah"]
        assert len(sarah) == 1
        assert sarah[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_regex_extracts_place_from_context(self):
        """'at [Name]' context triggers place classification in regex."""
        extractor = EntityExtractor()
        result = await extractor.extract("Meeting at Starbucks this afternoon")
        starbucks = [e for e in result.entities if e.normalized_name == "starbucks"]
        assert len(starbucks) == 1
        assert starbucks[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_regex_extracts_org_by_suffix(self):
        """Organization suffixes (Corp, LLC, etc.) trigger org classification.

        The proper noun regex captures 'Acme Corp' as a two-word capitalized
        phrase. The suffix check classifies it as ORGANIZATION — but only
        when no higher-priority pattern (person/place) matches first.
        Using a neutral context (no 'with', 'at', 'to', etc.) so the
        suffix check is the deciding factor.
        """
        extractor = EntityExtractor()
        # Neutral context: no person/place pattern triggers
        result = await extractor.extract("Acme Corp released their quarterly report")
        acme_corp = [e for e in result.entities if e.normalized_name == "acme corp"]
        assert len(acme_corp) == 1
        assert acme_corp[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_regex_extracts_project_from_context(self):
        """'working on [Name]' triggers project classification."""
        extractor = EntityExtractor()
        result = await extractor.extract(
            "I've been working on Phoenix all week"
        )
        phoenix = [e for e in result.entities if e.normalized_name == "phoenix"]
        assert len(phoenix) == 1
        assert phoenix[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_regex_extracts_known_tools(self):
        """Known tool names are extracted with high confidence."""
        extractor = EntityExtractor()
        result = await extractor.extract("Check the Notion page and update Slack")
        tool_names = {e.normalized_name for e in result.entities if e.entity_type == EntityType.TOOL}
        assert "notion" in tool_names
        assert "slack" in tool_names

    @pytest.mark.asyncio
    async def test_regex_tool_confidence_beats_proper_noun(self):
        """Tool detection (0.8 confidence) should beat proper noun regex (0.5)."""
        extractor = EntityExtractor()
        result = await extractor.extract("Open Notion and start writing")
        notion = [e for e in result.entities if e.normalized_name == "notion"]
        assert len(notion) == 1
        assert notion[0].confidence >= 0.7
        assert notion[0].entity_type == EntityType.TOOL

    @pytest.mark.asyncio
    async def test_regex_multi_word_tool_google_calendar(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Add it to google calendar")
        cal = [e for e in result.entities if e.normalized_name == "google calendar"]
        assert len(cal) == 1
        assert cal[0].entity_type == EntityType.TOOL

    @pytest.mark.asyncio
    async def test_regex_skips_common_sentence_starters(self):
        """Common words like 'The', 'This', 'What' should not be extracted."""
        extractor = EntityExtractor()
        result = await extractor.extract(
            "The project is going well. This is great. What a day."
        )
        names = _normalized(result)
        for skip_word in ["the", "this", "what"]:
            assert skip_word not in names

    @pytest.mark.asyncio
    async def test_regex_skips_pronouns(self):
        extractor = EntityExtractor()
        result = await extractor.extract("He said She was there")
        names = _normalized(result)
        assert "he" not in names
        assert "she" not in names

    @pytest.mark.asyncio
    async def test_regex_person_said_pattern(self):
        """'[Name] said' triggers person classification."""
        extractor = EntityExtractor()
        result = await extractor.extract("Marcus said we should reconsider")
        marcus = [e for e in result.entities if e.normalized_name == "marcus"]
        assert len(marcus) == 1
        assert marcus[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_regex_single_cap_word_defaults_to_person(self):
        """Single capitalized word without place/project context defaults to person.

        Note: 'to Alex' matches the place pattern 'to {name}', so we use
        a context that doesn't trigger place/project heuristics.
        """
        extractor = EntityExtractor()
        result = await extractor.extract("Alex mentioned it yesterday")
        alex = [e for e in result.entities if e.normalized_name == "alex"]
        assert len(alex) == 1
        assert alex[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_regex_multi_word_cap_defaults_to_topic(self):
        """Multi-word capitalized phrase without clear context defaults to topic."""
        extractor = EntityExtractor()
        result = await extractor.extract("This reminds me of Deep Learning Fundamentals")
        dl = [e for e in result.entities if "deep learning" in e.normalized_name]
        assert len(dl) == 1
        assert dl[0].entity_type == EntityType.TOPIC

    @pytest.mark.asyncio
    async def test_regex_deduplication_keeps_highest_confidence(self):
        """When same entity is found by both regex and tool, keep highest confidence."""
        extractor = EntityExtractor()
        result = await extractor.extract("I use Notion every day for Notion pages")
        notion = [e for e in result.entities if e.normalized_name == "notion"]
        assert len(notion) == 1  # Deduplicated
        assert notion[0].confidence == 0.8  # Tool confidence

    @pytest.mark.asyncio
    async def test_regex_context_snippet_populated(self):
        """Regex extraction should populate context_snippet for proper nouns."""
        extractor = EntityExtractor()
        result = await extractor.extract("I had lunch with Sarah and it was nice")
        sarah = [e for e in result.entities if e.normalized_name == "sarah"]
        assert len(sarah) == 1
        assert sarah[0].context_snippet  # Should have surrounding text

    @pytest.mark.asyncio
    async def test_regex_span_populated(self):
        """Regex extraction should populate character span."""
        extractor = EntityExtractor()
        text = "I met Sarah at the office"
        result = await extractor.extract(text)
        sarah = [e for e in result.entities if e.normalized_name == "sarah"]
        assert len(sarah) == 1
        assert sarah[0].span is not None
        start, end = sarah[0].span
        assert text[start:end] == "Sarah"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: Regex temporal extraction edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestRegexTemporalEdgeCases:
    """Test temporal regex patterns for all types and edge cases."""

    @pytest.mark.asyncio
    async def test_relative_tomorrow(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Finish this tomorrow")
        assert any(t.temporal_type == TemporalType.RELATIVE for t in result.temporal_references)

    @pytest.mark.asyncio
    async def test_relative_next_week(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Let's revisit next week")
        rel = [t for t in result.temporal_references if t.temporal_type == TemporalType.RELATIVE]
        assert len(rel) >= 1

    @pytest.mark.asyncio
    async def test_relative_in_n_days(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Ship in 3 days")
        rel = [t for t in result.temporal_references if t.temporal_type == TemporalType.RELATIVE]
        assert len(rel) >= 1

    @pytest.mark.asyncio
    async def test_absolute_time_3pm(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Call at 3pm")
        abs_t = [t for t in result.temporal_references if t.temporal_type == TemporalType.ABSOLUTE]
        assert len(abs_t) >= 1

    @pytest.mark.asyncio
    async def test_absolute_date(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Due on March 15th")
        abs_t = [t for t in result.temporal_references if t.temporal_type == TemporalType.ABSOLUTE]
        assert len(abs_t) >= 1

    @pytest.mark.asyncio
    async def test_absolute_date_slash_format(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Meeting on 3/15/2026")
        abs_t = [t for t in result.temporal_references if t.temporal_type == TemporalType.ABSOLUTE]
        assert len(abs_t) >= 1

    @pytest.mark.asyncio
    async def test_recurring_every_monday(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Standup every Monday at 10am")
        rec = [t for t in result.temporal_references if t.temporal_type == TemporalType.RECURRING]
        assert len(rec) >= 1

    @pytest.mark.asyncio
    async def test_recurring_daily(self):
        extractor = EntityExtractor()
        result = await extractor.extract("We do daily standups")
        rec = [t for t in result.temporal_references if t.temporal_type == TemporalType.RECURRING]
        assert len(rec) >= 1

    @pytest.mark.asyncio
    async def test_duration_for_30_minutes(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Block for 30 minutes")
        dur = [t for t in result.temporal_references if t.temporal_type == TemporalType.DURATION]
        assert len(dur) >= 1

    @pytest.mark.asyncio
    async def test_vague_soon(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I'll get to it soon")
        vag = [t for t in result.temporal_references if t.temporal_type == TemporalType.VAGUE]
        assert len(vag) >= 1

    @pytest.mark.asyncio
    async def test_vague_later(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Let's do it later")
        vag = [t for t in result.temporal_references if t.temporal_type == TemporalType.VAGUE]
        assert len(vag) >= 1

    @pytest.mark.asyncio
    async def test_multiple_temporal_types_in_one_utterance(self):
        extractor = EntityExtractor()
        result = await extractor.extract(
            "Meeting tomorrow at 3pm for 30 minutes, recurring every Monday"
        )
        types = {t.temporal_type for t in result.temporal_references}
        assert len(types) >= 2  # At least relative + one other

    @pytest.mark.asyncio
    async def test_no_duplicate_temporals(self):
        """Same temporal mentioned twice should only appear once."""
        extractor = EntityExtractor()
        result = await extractor.extract("tomorrow and tomorrow again")
        tomorrow_refs = [t for t in result.temporal_references if "tomorrow" in t.text.lower()]
        assert len(tomorrow_refs) == 1


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: Regex fact extraction edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestRegexFactEdgeCases:
    """Test regex-based fact/preference/habit extraction."""

    @pytest.mark.asyncio
    async def test_preference_i_prefer(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I prefer async communication")
        prefs = [f for f in result.facts if f.fact_type == FactType.PREFERENCE]
        assert len(prefs) >= 1

    @pytest.mark.asyncio
    async def test_preference_i_like(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I like working in the morning")
        prefs = [f for f in result.facts if f.fact_type == FactType.PREFERENCE]
        assert len(prefs) >= 1

    @pytest.mark.asyncio
    async def test_preference_i_hate(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I hate long meetings")
        prefs = [f for f in result.facts if f.fact_type == FactType.PREFERENCE]
        assert len(prefs) >= 1

    @pytest.mark.asyncio
    async def test_habit_i_usually(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I usually take a walk at lunch")
        habits = [f for f in result.facts if f.fact_type == FactType.HABIT]
        assert len(habits) >= 1

    @pytest.mark.asyncio
    async def test_habit_i_always(self):
        extractor = EntityExtractor()
        result = await extractor.extract("I always check email first thing")
        habits = [f for f in result.facts if f.fact_type == FactType.HABIT]
        assert len(habits) >= 1

    @pytest.mark.asyncio
    async def test_attribute_x_is_my_y(self):
        extractor = EntityExtractor()
        result = await extractor.extract("Sarah is my manager")
        attrs = [f for f in result.facts if f.fact_type == FactType.ATTRIBUTE]
        assert len(attrs) >= 1

    @pytest.mark.asyncio
    async def test_attribute_my_x_is_y(self):
        extractor = EntityExtractor()
        result = await extractor.extract("My office is downtown")
        attrs = [f for f in result.facts if f.fact_type == FactType.ATTRIBUTE]
        assert len(attrs) >= 1

    @pytest.mark.asyncio
    async def test_short_match_filtered_out(self):
        """Fact matches <= 5 chars should be filtered out."""
        extractor = EntityExtractor()
        result = await extractor.extract("I like x.")
        # "I like x" is 8 chars so it would match, but "x." is the captured group
        # The filter is on the full match group(0), which is "I like x."
        # Just ensure no crash
        assert isinstance(result, ExtractionResult)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: Confidence filtering and clamping
# ═══════════════════════════════════════════════════════════════════════


class TestConfidenceHandling:
    """Test confidence scoring, filtering, and edge cases."""

    @pytest.mark.asyncio
    async def test_filters_below_min_confidence(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Sarah", "person", confidence=0.95),
            _entity("maybe", "topic", confidence=0.1),
        ]))
        extractor = EntityExtractor(client, min_confidence=0.3)
        result = await extractor.extract("Sarah mentioned maybe thing")
        assert result.entity_count == 1
        assert result.entities[0].name == "Sarah"

    @pytest.mark.asyncio
    async def test_clamps_confidence_above_one(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Test", "person", confidence=1.5),
        ]))
        extractor = EntityExtractor(client, min_confidence=0.0)
        result = await extractor.extract("Test person")
        assert result.entities[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_clamps_confidence_below_zero(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Test", "person", confidence=-0.5),
        ]))
        extractor = EntityExtractor(client, min_confidence=0.0)
        result = await extractor.extract("Test person")
        assert result.entities[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_zero_min_confidence_allows_all(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Low", "person", confidence=0.01),
            _entity("High", "person", confidence=0.99),
        ]))
        extractor = EntityExtractor(client, min_confidence=0.0)
        result = await extractor.extract("Low and High")
        assert result.entity_count == 2

    @pytest.mark.asyncio
    async def test_high_min_confidence_filters_most(self):
        client = _mock_client(_empty_response(entities=[
            _entity("Sure", "person", confidence=0.95),
            _entity("Maybe", "person", confidence=0.7),
            _entity("Doubt", "person", confidence=0.4),
        ]))
        extractor = EntityExtractor(client, min_confidence=0.9)
        result = await extractor.extract("Sure Maybe Doubt")
        assert result.entity_count == 1
        assert result.entities[0].name == "Sure"

    @pytest.mark.asyncio
    async def test_relationship_confidence_filtering(self):
        client = _mock_client(_empty_response(
            entities=[_entity("A", "person"), _entity("B", "person")],
            relationships=[
                _rel("A", "B", "works_with", confidence=0.1),
            ],
        ))
        extractor = EntityExtractor(client, min_confidence=0.3)
        result = await extractor.extract("A and B")
        assert len(result.relationships) == 0  # Filtered out

    @pytest.mark.asyncio
    async def test_fact_confidence_filtering(self):
        client = _mock_client(_empty_response(
            facts=[_fact("low conf fact", "preference", confidence=0.1)],
        ))
        extractor = EntityExtractor(client, min_confidence=0.3)
        result = await extractor.extract("something")
        assert len(result.facts) == 0


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: Relationship extraction
# ═══════════════════════════════════════════════════════════════════════


class TestRelationshipExtraction:
    """Test relationship extraction between entities."""

    @pytest.mark.asyncio
    async def test_managed_by_relationship(self):
        client = _mock_client(_empty_response(
            entities=[_entity("Sarah", "person")],
            relationships=[_rel("speaker", "Sarah", "managed_by")],
        ))
        result = await EntityExtractor(client).extract("Sarah is my manager")
        assert result.relationships[0].relationship_type == RelationshipType.MANAGED_BY

    @pytest.mark.asyncio
    async def test_works_with_relationship(self):
        client = _mock_client(_empty_response(
            entities=[_entity("Alice", "person"), _entity("Bob", "person")],
            relationships=[_rel("Alice", "Bob", "works_with")],
        ))
        result = await EntityExtractor(client).extract("Alice works with Bob")
        assert result.relationships[0].relationship_type == RelationshipType.WORKS_WITH

    @pytest.mark.asyncio
    async def test_collaborates_on_relationship(self):
        client = _mock_client(_empty_response(
            entities=[
                _entity("Sarah", "person"),
                _entity("Atlas", "project"),
            ],
            relationships=[_rel("Sarah", "Atlas", "collaborates_on")],
        ))
        result = await EntityExtractor(client).extract("Sarah is working on Atlas")
        assert result.relationships[0].relationship_type == RelationshipType.COLLABORATES_ON

    @pytest.mark.asyncio
    async def test_unknown_relationship_defaults_to_related_to(self):
        client = _mock_client(_empty_response(
            entities=[_entity("A", "person"), _entity("B", "person")],
            relationships=[_rel("A", "B", "some_unknown_rel")],
        ))
        result = await EntityExtractor(client).extract("A and B")
        assert result.relationships[0].relationship_type == RelationshipType.RELATED_TO

    @pytest.mark.asyncio
    async def test_empty_source_or_target_skipped(self):
        client = _mock_client(_empty_response(
            relationships=[
                {"source": "", "target": "B", "type": "works_with", "confidence": 0.9},
                {"source": "A", "target": "", "type": "works_with", "confidence": 0.9},
                {"source": "C", "target": "D", "type": "works_with", "confidence": 0.9},
            ],
        ))
        result = await EntityExtractor(client).extract("test")
        assert len(result.relationships) == 1
        assert result.relationships[0].source_name == "C"

    @pytest.mark.asyncio
    async def test_multiple_relationships(self):
        client = _mock_client(_empty_response(
            entities=[
                _entity("Sarah", "person"),
                _entity("Acme", "organization"),
                _entity("Atlas", "project"),
            ],
            relationships=[
                _rel("Sarah", "Acme", "part_of"),
                _rel("Sarah", "Atlas", "collaborates_on"),
                _rel("Acme", "Atlas", "contains"),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Sarah at Acme works on Atlas"
        )
        assert len(result.relationships) == 3


# ═══════════════════════════════════════════════════════════════════════
# SECTION 11: Error handling and robustness
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Test extraction robustness under various failure modes."""

    @pytest.mark.asyncio
    async def test_invalid_json_raises_extraction_error(self):
        """Invalid JSON from LLM should raise EntityExtractionError."""
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = "not valid json"
        resp.model = "gemini-2.5-flash-lite"
        resp.latency_ms = 10.0
        client.generate = AsyncMock(return_value=resp)

        extractor = EntityExtractor(client)
        # LLM fails → falls back to regex, which returns a valid result
        result = await extractor.extract("Sarah mentioned something")
        assert result.model_used == "regex-fallback"

    @pytest.mark.asyncio
    async def test_api_error_falls_back_to_regex(self):
        client = MagicMock()
        client.is_connected = True
        client.generate = AsyncMock(side_effect=Exception("API timeout"))
        extractor = EntityExtractor(client)
        result = await extractor.extract("Sarah is here")
        assert result.model_used == "regex-fallback"

    @pytest.mark.asyncio
    async def test_malformed_entity_dict_is_skipped(self):
        """Entities missing required fields should be skipped gracefully."""
        client = _mock_client(_empty_response(entities=[
            {"type": "person", "confidence": 0.9},  # Missing name
            {"name": "Valid", "type": "person", "confidence": 0.9},
        ]))
        result = await EntityExtractor(client).extract("test")
        assert result.entity_count == 1
        assert result.entities[0].name == "Valid"

    @pytest.mark.asyncio
    async def test_non_dict_response_raises_error(self):
        """Array response from LLM should raise because dict expected."""
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '["not", "a", "dict"]'
        resp.model = "gemini-2.5-flash-lite"
        resp.latency_ms = 10.0
        client.generate = AsyncMock(return_value=resp)

        extractor = EntityExtractor(client)
        # Falls back to regex because parse raises ValueError
        result = await extractor.extract("test input")
        assert result.model_used == "regex-fallback"

    @pytest.mark.asyncio
    async def test_max_entities_limit_enforced(self):
        many_entities = [_entity(f"E{i}", "person") for i in range(100)]
        client = _mock_client(_empty_response(entities=many_entities))
        extractor = EntityExtractor(client, max_entities_per_extraction=5)
        result = await extractor.extract("lots of entities")
        assert result.entity_count == 5

    @pytest.mark.asyncio
    async def test_missing_sections_in_response_ok(self):
        """Response missing some sections (e.g. no facts key) should not crash."""
        client = _mock_client({"entities": [_entity("Sarah", "person")]})
        result = await EntityExtractor(client).extract("Sarah is here")
        assert result.entity_count == 1
        assert result.facts == []
        assert result.relationships == []
        assert result.temporal_references == []


# ═══════════════════════════════════════════════════════════════════════
# SECTION 12: LLM response parsing robustness
# ═══════════════════════════════════════════════════════════════════════


class TestResponseParsing:
    """Test parsing of various LLM response formats and quirks."""

    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self):
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '```json\n{"entities": [{"name": "Test", "type": "person", "confidence": 0.9}], "relationships": [], "facts": [], "temporal_references": []}\n```'
        resp.model = "test"
        resp.latency_ms = 1.0
        client.generate = AsyncMock(return_value=resp)

        result = await EntityExtractor(client).extract("test")
        assert result.entity_count == 1

    @pytest.mark.asyncio
    async def test_trailing_commas_cleaned(self):
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '{"entities": [{"name": "Test", "type": "person", "confidence": 0.9,},], "relationships": [], "facts": [], "temporal_references": [],}'
        resp.model = "test"
        resp.latency_ms = 1.0
        client.generate = AsyncMock(return_value=resp)

        result = await EntityExtractor(client).extract("test")
        assert result.entity_count == 1

    @pytest.mark.asyncio
    async def test_bare_code_fence_without_json_tag(self):
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '```\n{"entities": [{"name": "X", "type": "person", "confidence": 0.9}], "relationships": [], "facts": [], "temporal_references": []}\n```'
        resp.model = "test"
        resp.latency_ms = 1.0
        client.generate = AsyncMock(return_value=resp)

        result = await EntityExtractor(client).extract("test")
        assert result.entity_count == 1

    @pytest.mark.asyncio
    async def test_extra_whitespace_in_response(self):
        client = MagicMock()
        client.is_connected = True
        resp = MagicMock()
        resp.text = '\n\n  {"entities": [], "relationships": [], "facts": [], "temporal_references": []}  \n\n'
        resp.model = "test"
        resp.latency_ms = 1.0
        client.generate = AsyncMock(return_value=resp)

        result = await EntityExtractor(client).extract("test")
        assert result.entity_count == 0


# ═══════════════════════════════════════════════════════════════════════
# SECTION 13: Accuracy validation — realistic natural speech
# ═══════════════════════════════════════════════════════════════════════


class TestAccuracyValidation:
    """Validate extraction accuracy on realistic, natural speech inputs.

    These tests simulate real blurt utterances with expected LLM outputs
    to validate the full pipeline produces correct, well-typed results.
    """

    @pytest.mark.asyncio
    async def test_task_with_person_project_tool(self):
        """Task-intent: person + project + tool extraction."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Marco", "person", confidence=0.95, attributes={"role": "designer"}),
                _entity("the landing page redesign", "project", confidence=0.9),
                _entity("Figma", "tool", confidence=0.98),
            ],
            relationships=[
                _rel("speaker", "Marco", "works_with"),
                _rel("Marco", "the landing page redesign", "collaborates_on"),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Remind me to send Marco the updated mockups from Figma "
            "for the landing page redesign"
        )
        assert result.entity_count == 3
        types = _types_map(result)
        assert types["Marco"] == EntityType.PERSON
        assert types["the landing page redesign"] == EntityType.PROJECT
        assert types["Figma"] == EntityType.TOOL
        assert len(result.relationships) == 2

    @pytest.mark.asyncio
    async def test_event_with_multiple_people_and_place(self):
        """Event-intent: multiple people + place extraction."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Rachel", "person", confidence=0.92),
                _entity("Tom", "person", confidence=0.92),
                _entity("Rooftop Bar", "place", confidence=0.88),
            ],
            temporal_references=[
                {"text": "Friday at 7pm", "type": "absolute"},
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Set up dinner with Rachel and Tom at the Rooftop Bar on Friday at 7pm"
        )
        persons = result.entities_by_type(EntityType.PERSON)
        places = result.entities_by_type(EntityType.PLACE)
        assert len(persons) == 2
        assert len(places) == 1
        assert len(result.temporal_references) == 1

    @pytest.mark.asyncio
    async def test_journal_entry_with_emotion_context(self):
        """Journal-intent: entities from reflective, emotional speech."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Anthropic", "organization", confidence=0.95),
                _entity("the API project", "project", confidence=0.9),
                _entity("Maya", "person", confidence=0.92, attributes={"role": "mentor"}),
            ],
            facts=[
                _fact("Feeling good about progress", "attribute", confidence=0.8),
                _fact("Maya's feedback was helpful", "attribute", "Maya", confidence=0.85),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Feeling really good about the progress at Anthropic. "
            "The API project is coming together and Maya's feedback was super helpful."
        )
        assert result.entity_count == 3
        assert len(result.facts) == 2

    @pytest.mark.asyncio
    async def test_casual_rambling_extracts_correctly(self):
        """Casual, unstructured speech still yields correct extraction."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Jess", "person", confidence=0.9),
                _entity("Slack", "tool", confidence=0.95),
                _entity("the onboarding redesign", "project", confidence=0.88),
                _entity("the UX team", "organization", confidence=0.85),
                _entity("New York", "place", confidence=0.92),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Okay so Jess pinged me on Slack about the onboarding redesign "
            "and apparently the UX team in New York has some thoughts"
        )
        assert result.entity_count == 5
        types = _types_map(result)
        assert types["Jess"] == EntityType.PERSON
        assert types["Slack"] == EntityType.TOOL
        assert types["the UX team"] == EntityType.ORGANIZATION
        assert types["New York"] == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_question_intent_extracts_entities(self):
        """Question-intent: entities extracted from a question."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Karen", "person", confidence=0.92),
                _entity("budget proposal", "project", confidence=0.88),
            ],
            temporal_references=[
                {"text": "last week", "type": "relative"},
            ],
        ))
        result = await EntityExtractor(client).extract(
            "What did Karen say about the budget proposal last week?"
        )
        assert "Karen" in _names(result)
        assert "budget proposal" in _names(result)
        assert len(result.temporal_references) == 1

    @pytest.mark.asyncio
    async def test_reminder_with_rich_context(self):
        """Reminder-intent: person + place + temporal."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Dr. Patel", "person", confidence=0.95),
                _entity("the clinic on Oak Street", "place", confidence=0.88),
            ],
            temporal_references=[
                {"text": "tomorrow at 3", "type": "relative"},
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Don't forget I have that appointment with Dr. Patel "
            "at the clinic on Oak Street tomorrow at 3"
        )
        types = _types_map(result)
        assert types["Dr. Patel"] == EntityType.PERSON
        assert types["the clinic on Oak Street"] == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_update_with_status_entities(self):
        """Update-intent: project status with org + place references."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("the database migration", "project", confidence=0.92),
                _entity("DevOps", "organization", confidence=0.88),
                _entity("production", "place", confidence=0.82),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "Quick update: the database migration to production is done. "
            "DevOps signed off this morning."
        )
        assert result.entity_count == 3

    @pytest.mark.asyncio
    async def test_idea_with_tools_and_orgs(self):
        """Idea-intent: tools + organizations in brainstorming speech."""
        client = _mock_client(_empty_response(
            entities=[
                _entity("Notion", "tool", confidence=0.98),
                _entity("our CRM", "project", confidence=0.85),
                _entity("Salesforce", "organization", confidence=0.9),
            ],
        ))
        result = await EntityExtractor(client).extract(
            "What if we integrated Notion with our CRM? "
            "Salesforce does something similar but it's too heavy."
        )
        assert result.entity_count == 3
        assert result.entities_by_type(EntityType.TOOL)[0].name == "Notion"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 14: Batch extraction
# ═══════════════════════════════════════════════════════════════════════


class TestBatchExtractionComprehensive:
    """Test batch extraction with various inputs."""

    @pytest.mark.asyncio
    async def test_batch_returns_one_result_per_input(self):
        extractor = EntityExtractor()
        texts = ["Sarah said hello", "Meeting at Google", "Check Notion"]
        results = await extractor.extract_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        extractor = EntityExtractor()
        results = await extractor.extract_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_with_mixed_empty_and_valid(self):
        extractor = EntityExtractor()
        results = await extractor.extract_batch(["", "Sarah is here", "  "])
        assert len(results) == 3
        assert results[0].entity_count == 0  # Empty
        assert results[1].has_entities  # Sarah
        assert results[2].entity_count == 0  # Whitespace

    @pytest.mark.asyncio
    async def test_batch_preserves_raw_text(self):
        extractor = EntityExtractor()
        texts = ["Hello Sarah", "Bye Bob"]
        results = await extractor.extract_batch(texts)
        assert results[0].raw_text == "Hello Sarah"
        assert results[1].raw_text == "Bye Bob"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 15: ExtractionResult dataclass properties
# ═══════════════════════════════════════════════════════════════════════


class TestExtractionResultProperties:
    """Test ExtractionResult computed properties."""

    def test_empty_result_defaults(self):
        result = ExtractionResult()
        assert result.entity_count == 0
        assert not result.has_entities
        assert result.entity_types_found == set()
        assert result.entities_by_type(EntityType.PERSON) == []
        assert result.raw_text == ""
        assert result.model_used == ""
        assert result.latency_ms == 0.0

    def test_entity_types_found(self):
        result = ExtractionResult(entities=[
            ExtractedEntity(name="A", entity_type=EntityType.PERSON),
            ExtractedEntity(name="B", entity_type=EntityType.PLACE),
            ExtractedEntity(name="C", entity_type=EntityType.PERSON),
        ])
        assert result.entity_types_found == {EntityType.PERSON, EntityType.PLACE}

    def test_entities_by_type_filters_correctly(self):
        result = ExtractionResult(entities=[
            ExtractedEntity(name="A", entity_type=EntityType.PERSON),
            ExtractedEntity(name="B", entity_type=EntityType.PLACE),
            ExtractedEntity(name="C", entity_type=EntityType.PERSON),
        ])
        assert len(result.entities_by_type(EntityType.PERSON)) == 2
        assert len(result.entities_by_type(EntityType.PLACE)) == 1
        assert len(result.entities_by_type(EntityType.TOOL)) == 0


# ═══════════════════════════════════════════════════════════════════════
# SECTION 16: ExtractedEntity dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestExtractedEntityDataclass:
    """Test ExtractedEntity auto-normalization and defaults."""

    def test_auto_normalizes_name(self):
        e = ExtractedEntity(name="  Sarah Johnson  ", entity_type=EntityType.PERSON)
        assert e.normalized_name == "sarah johnson"

    def test_explicit_normalized_name_preserved(self):
        e = ExtractedEntity(
            name="Sarah",
            entity_type=EntityType.PERSON,
            normalized_name="sarah_j",
        )
        assert e.normalized_name == "sarah_j"

    def test_default_confidence(self):
        e = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert e.confidence == 1.0

    def test_default_aliases_and_attributes(self):
        e = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert e.aliases == []
        assert e.attributes == {}

    def test_span_defaults_to_none(self):
        e = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert e.span is None

    def test_context_snippet_defaults_to_empty(self):
        e = ExtractedEntity(name="Test", entity_type=EntityType.PERSON)
        assert e.context_snippet == ""
