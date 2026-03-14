"""Unit and integration tests for entity extraction accuracy.

Validates that the EntityExtractor correctly identifies and classifies
entities across all four entity types (person, place, project, organization)
from diverse natural speech inputs.

Tests cover:
- Each entity type individually with varied phrasing
- Multiple entities in a single utterance
- Ambiguous and indirect references
- Edge cases (empty input, no entities, malformed responses)
- Metadata extraction (roles, relationships, temporal)
- Response parsing robustness (JSON variants, markdown wrapping)
- Integration flow: transcript → extraction → structured entities
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from blurt.clients.gemini import GeminiResponse, ModelTier
from blurt.core.entity_extractor import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    EntityExtractor,
    ExtractionResult,
)
from blurt.core.models import Entity, EntityType


# ── Fixtures & Helpers ───────────────────────────────────────────────


def _make_gemini_response(entities_json: list[dict[str, Any]], latency_ms: float = 50.0) -> GeminiResponse:
    """Build a mock GeminiResponse with a JSON entity list."""
    return GeminiResponse(
        text=json.dumps(entities_json),
        raw={"candidates": [{"content": {"parts": [{"text": json.dumps(entities_json)}]}}]},
        model="gemini-2.5-flash-lite",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=latency_ms,
    )


def _make_gemini_response_raw(text: str, latency_ms: float = 50.0) -> GeminiResponse:
    """Build a mock GeminiResponse with raw text (for edge cases)."""
    return GeminiResponse(
        text=text,
        raw={},
        model="gemini-2.5-flash-lite",
        usage={},
        latency_ms=latency_ms,
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock GeminiClient."""
    client = AsyncMock()
    client.generate = AsyncMock()
    return client


@pytest.fixture
def extractor(mock_client: AsyncMock) -> EntityExtractor:
    """Create an EntityExtractor with mocked Gemini client."""
    return EntityExtractor(mock_client)


def _entity_names(result: ExtractionResult) -> set[str]:
    """Extract entity names from a result as a set."""
    return {e.name for e in result.entities}


def _entity_types(result: ExtractionResult) -> dict[str, EntityType]:
    """Map entity name → type for easy assertions."""
    return {e.name: e.entity_type for e in result.entities}


# ── PERSON Entity Tests ──────────────────────────────────────────────


class TestPersonExtraction:
    """Test extraction of PERSON entities from diverse speech patterns."""

    @pytest.mark.asyncio
    async def test_simple_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a simple first name."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Sarah", "entity_type": "person", "metadata": {}}
        ])

        result = await extractor.extract("I need to talk to Sarah about this")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].name == "Sarah"
        assert result.entities[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_full_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a full name with first and last."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "John Mitchell", "entity_type": "person", "metadata": {}}
        ])

        result = await extractor.extract("John Mitchell called me this morning")

        assert result.success
        assert result.entities[0].name == "John Mitchell"
        assert result.entities[0].entity_type == EntityType.PERSON

    @pytest.mark.asyncio
    async def test_title_and_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a person with a professional title."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Dr. Chen", "entity_type": "person", "metadata": {"title": "Dr."}}
        ])

        result = await extractor.extract("I have an appointment with Dr. Chen next week")

        assert result.success
        assert result.entities[0].name == "Dr. Chen"

    @pytest.mark.asyncio
    async def test_role_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a person referred to by role/relationship."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "my manager", "entity_type": "person", "metadata": {"role": "manager", "relationship": "reports_to"}}
        ])

        result = await extractor.extract("My manager wants the report by Friday")

        assert result.success
        entity = result.entities[0]
        assert entity.entity_type == EntityType.PERSON
        assert entity.metadata.get("role") == "manager"

    @pytest.mark.asyncio
    async def test_multiple_people(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract multiple people from one utterance."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Alice", "entity_type": "person", "metadata": {}},
            {"name": "Bob", "entity_type": "person", "metadata": {}},
            {"name": "Carlos", "entity_type": "person", "metadata": {}},
        ])

        result = await extractor.extract(
            "I need Alice, Bob, and Carlos on the call tomorrow"
        )

        assert result.success
        assert result.entity_count == 3
        persons = result.entities_by_type(EntityType.PERSON)
        assert len(persons) == 3
        assert _entity_names(result) == {"Alice", "Bob", "Carlos"}

    @pytest.mark.asyncio
    async def test_informal_nickname(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract person mentioned by casual nickname."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Mike", "entity_type": "person", "metadata": {"informal": True}}
        ])

        result = await extractor.extract("gotta ping Mike about the deploy")

        assert result.success
        assert result.entities[0].name == "Mike"
        assert result.entities[0].entity_type == EntityType.PERSON


# ── PLACE Entity Tests ───────────────────────────────────────────────


class TestPlaceExtraction:
    """Test extraction of PLACE entities from diverse speech patterns."""

    @pytest.mark.asyncio
    async def test_city_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a city name."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Berlin", "entity_type": "place", "metadata": {"type": "city"}}
        ])

        result = await extractor.extract("The conference is in Berlin next month")

        assert result.success
        assert result.entities[0].name == "Berlin"
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_office_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract an informal place reference."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "the downtown office", "entity_type": "place", "metadata": {}}
        ])

        result = await extractor.extract("Let's meet at the downtown office")

        assert result.success
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_room_number(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a specific room reference."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Room 302", "entity_type": "place", "metadata": {"type": "room"}}
        ])

        result = await extractor.extract("The meeting is in Room 302")

        assert result.success
        assert result.entities[0].name == "Room 302"

    @pytest.mark.asyncio
    async def test_venue_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a named venue."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Blue Bottle Coffee", "entity_type": "place", "metadata": {"type": "cafe"}}
        ])

        result = await extractor.extract("Grab coffee at Blue Bottle Coffee on 3rd street")

        assert result.success
        assert result.entities[0].name == "Blue Bottle Coffee"
        assert result.entities[0].entity_type == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_country_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract country-level place references."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Japan", "entity_type": "place", "metadata": {"type": "country"}},
            {"name": "Tokyo", "entity_type": "place", "metadata": {"type": "city"}},
        ])

        result = await extractor.extract("Flying to Tokyo, Japan for the summit")

        assert result.success
        places = result.entities_by_type(EntityType.PLACE)
        assert len(places) == 2
        assert _entity_names(result) == {"Japan", "Tokyo"}

    @pytest.mark.asyncio
    async def test_home_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract casual 'home' as a place."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "home", "entity_type": "place", "metadata": {"type": "residence"}}
        ])

        result = await extractor.extract("I'll be working from home tomorrow")

        assert result.success
        assert result.entities[0].entity_type == EntityType.PLACE


# ── PROJECT Entity Tests ─────────────────────────────────────────────


class TestProjectExtraction:
    """Test extraction of PROJECT entities from diverse speech patterns."""

    @pytest.mark.asyncio
    async def test_named_project(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a properly named project."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Project Atlas", "entity_type": "project", "metadata": {}}
        ])

        result = await extractor.extract("We need to finalize Project Atlas this sprint")

        assert result.success
        assert result.entities[0].name == "Project Atlas"
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_informal_project_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract an informally referenced project."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "the Q2 deck", "entity_type": "project", "metadata": {"temporal": "Q2"}}
        ])

        result = await extractor.extract("I still need to finish the Q2 deck before the review")

        assert result.success
        assert result.entities[0].name == "the Q2 deck"
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_codenamed_project(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a project known by a codename."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Phoenix", "entity_type": "project", "metadata": {"codename": True}}
        ])

        result = await extractor.extract("Phoenix is behind schedule, we might need to cut scope")

        assert result.success
        assert result.entities[0].name == "Phoenix"
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_versioned_project(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a project with version reference."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "v2 migration", "entity_type": "project", "metadata": {"version": "2"}}
        ])

        result = await extractor.extract("The v2 migration is the top priority right now")

        assert result.success
        assert result.entities[0].entity_type == EntityType.PROJECT

    @pytest.mark.asyncio
    async def test_deliverable_as_project(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a deliverable/artifact as a project entity."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "annual report", "entity_type": "project", "metadata": {"type": "deliverable"}}
        ])

        result = await extractor.extract("Need to start drafting the annual report next week")

        assert result.success
        assert result.entities[0].entity_type == EntityType.PROJECT


# ── ORGANIZATION Entity Tests ────────────────────────────────────────


class TestOrganizationExtraction:
    """Test extraction of ORGANIZATION entities from diverse speech patterns."""

    @pytest.mark.asyncio
    async def test_company_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a well-known company name."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Google", "entity_type": "organization", "metadata": {"type": "company"}}
        ])

        result = await extractor.extract("Google just announced their new API pricing")

        assert result.success
        assert result.entities[0].name == "Google"
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_team_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a team reference as organization."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "the marketing team", "entity_type": "organization", "metadata": {"type": "team"}}
        ])

        result = await extractor.extract("The marketing team needs our assets by Thursday")

        assert result.success
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_department_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a department reference."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Engineering", "entity_type": "organization", "metadata": {"type": "department"}}
        ])

        result = await extractor.extract("Engineering is pushing back on the timeline")

        assert result.success
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_institution(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract an educational institution."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "MIT", "entity_type": "organization", "metadata": {"type": "university"}}
        ])

        result = await extractor.extract("The paper from MIT has some interesting findings")

        assert result.success
        assert result.entities[0].name == "MIT"
        assert result.entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_startup_name(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract a startup company name."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Acme Corp", "entity_type": "organization", "metadata": {"type": "company"}}
        ])

        result = await extractor.extract("We're partnering with Acme Corp for the launch")

        assert result.success
        assert result.entities[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_client_reference(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Extract an organization referred to as a client."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Meridian Health", "entity_type": "organization", "metadata": {"relationship": "client"}}
        ])

        result = await extractor.extract("Meridian Health wants to reschedule the demo")

        assert result.success
        assert result.entities[0].entity_type == EntityType.ORGANIZATION


# ── Multi-Entity and Mixed-Type Tests ────────────────────────────────


class TestMultiEntityExtraction:
    """Test extraction of multiple entities and mixed types from complex speech."""

    @pytest.mark.asyncio
    async def test_all_four_types_in_one_utterance(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract all four entity types from a single utterance."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Sarah", "entity_type": "person", "metadata": {}},
            {"name": "Project Atlas", "entity_type": "project", "metadata": {}},
            {"name": "Google", "entity_type": "organization", "metadata": {}},
            {"name": "San Francisco", "entity_type": "place", "metadata": {}},
        ])

        result = await extractor.extract(
            "Sarah is flying to San Francisco to present Project Atlas to Google"
        )

        assert result.success
        assert result.entity_count == 4
        types = _entity_types(result)
        assert types["Sarah"] == EntityType.PERSON
        assert types["Project Atlas"] == EntityType.PROJECT
        assert types["Google"] == EntityType.ORGANIZATION
        assert types["San Francisco"] == EntityType.PLACE

    @pytest.mark.asyncio
    async def test_complex_meeting_scenario(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract entities from a complex meeting description."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "James", "entity_type": "person", "metadata": {}},
            {"name": "Lisa", "entity_type": "person", "metadata": {}},
            {"name": "the product team", "entity_type": "organization", "metadata": {"type": "team"}},
            {"name": "Horizon", "entity_type": "project", "metadata": {}},
            {"name": "Building C conference room", "entity_type": "place", "metadata": {}},
        ])

        result = await extractor.extract(
            "Meeting with James and Lisa from the product team "
            "about Horizon in the Building C conference room"
        )

        assert result.success
        assert result.entity_count == 5
        assert len(result.entities_by_type(EntityType.PERSON)) == 2
        assert len(result.entities_by_type(EntityType.ORGANIZATION)) == 1
        assert len(result.entities_by_type(EntityType.PROJECT)) == 1
        assert len(result.entities_by_type(EntityType.PLACE)) == 1

    @pytest.mark.asyncio
    async def test_casual_rambling_speech(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract entities from casual, unstructured speech."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Dave", "entity_type": "person", "metadata": {}},
            {"name": "the onboarding flow", "entity_type": "project", "metadata": {}},
            {"name": "Figma", "entity_type": "organization", "metadata": {"type": "tool"}},
        ])

        result = await extractor.extract(
            "so yeah I was talking to Dave and he thinks the onboarding flow "
            "needs more work we should look at the Figma again"
        )

        assert result.success
        assert result.entity_count == 3

    @pytest.mark.asyncio
    async def test_status_update_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract entities from a project status update."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "API gateway", "entity_type": "project", "metadata": {}},
            {"name": "Priya", "entity_type": "person", "metadata": {"role": "lead"}},
            {"name": "DevOps", "entity_type": "organization", "metadata": {"type": "team"}},
            {"name": "AWS", "entity_type": "organization", "metadata": {"type": "cloud_provider"}},
        ])

        result = await extractor.extract(
            "The API gateway migration is 80 percent done. "
            "Priya is leading it with DevOps. "
            "We're moving everything to AWS by end of quarter."
        )

        assert result.success
        assert result.entity_count == 4
        names = _entity_names(result)
        assert "Priya" in names
        assert "AWS" in names

    @pytest.mark.asyncio
    async def test_journaling_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract entities from a journal-style personal reflection."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Central Park", "entity_type": "place", "metadata": {}},
            {"name": "Emma", "entity_type": "person", "metadata": {"relationship": "friend"}},
        ])

        result = await extractor.extract(
            "Had a great walk in Central Park this morning with Emma. "
            "Feeling really energized and creative today."
        )

        assert result.success
        assert result.entity_count == 2
        types = _entity_types(result)
        assert types["Central Park"] == EntityType.PLACE
        assert types["Emma"] == EntityType.PERSON


# ── Edge Cases and Error Handling ────────────────────────────────────


class TestEdgeCases:
    """Test edge cases, error handling, and robustness."""

    @pytest.mark.asyncio
    async def test_empty_input(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Empty string returns empty result without calling the model."""
        result = await extractor.extract("")

        assert result.success
        assert result.entity_count == 0
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_input(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Whitespace-only input returns empty result."""
        result = await extractor.extract("   \n\t  ")

        assert result.success
        assert result.entity_count == 0
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_entities_in_text(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Text with no entities returns empty list."""
        mock_client.generate.return_value = _make_gemini_response([])

        result = await extractor.extract("I feel tired today and want to take a nap")

        assert result.success
        assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_model_returns_invalid_json(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Handle model returning invalid JSON gracefully."""
        mock_client.generate.return_value = _make_gemini_response_raw("not valid json at all")

        result = await extractor.extract("Tell Sarah about the project")

        assert not result.success
        assert result.error is not None
        assert "JSON" in result.error or "json" in result.error.lower()

    @pytest.mark.asyncio
    async def test_model_returns_wrapped_json(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Handle model wrapping JSON in markdown code fences."""
        wrapped = '```json\n[{"name": "Sarah", "entity_type": "person", "metadata": {}}]\n```'
        mock_client.generate.return_value = _make_gemini_response_raw(wrapped)

        result = await extractor.extract("Talk to Sarah")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].name == "Sarah"

    @pytest.mark.asyncio
    async def test_model_returns_object_wrapper(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Handle model returning {"entities": [...]} instead of bare array."""
        obj_wrapped = json.dumps({
            "entities": [
                {"name": "Berlin", "entity_type": "place", "metadata": {}}
            ]
        })
        mock_client.generate.return_value = _make_gemini_response_raw(obj_wrapped)

        result = await extractor.extract("Going to Berlin")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].name == "Berlin"

    @pytest.mark.asyncio
    async def test_unknown_entity_type_is_skipped(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Unknown entity types are silently skipped."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Sarah", "entity_type": "person", "metadata": {}},
            {"name": "happiness", "entity_type": "emotion", "metadata": {}},
        ])

        result = await extractor.extract("Sarah makes me happy")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].name == "Sarah"

    @pytest.mark.asyncio
    async def test_entity_without_name_is_skipped(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Entities with empty or missing names are skipped."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "", "entity_type": "person", "metadata": {}},
            {"name": "Alice", "entity_type": "person", "metadata": {}},
        ])

        result = await extractor.extract("Alice said something")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_non_dict_items_in_array_skipped(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Non-dict items in the response array are skipped."""
        mock_client.generate.return_value = _make_gemini_response_raw(
            json.dumps([
                "invalid_string_item",
                {"name": "Bob", "entity_type": "person", "metadata": {}},
                42,
            ])
        )

        result = await extractor.extract("Bob is here")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].name == "Bob"

    @pytest.mark.asyncio
    async def test_invalid_metadata_type_handled(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Non-dict metadata is replaced with empty dict."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Sarah", "entity_type": "person", "metadata": "not_a_dict"}
        ])

        result = await extractor.extract("Talk to Sarah")

        assert result.success
        assert result.entity_count == 1
        assert result.entities[0].metadata == {}

    @pytest.mark.asyncio
    async def test_gemini_api_error(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """API errors are caught and returned as failed result."""
        mock_client.generate.side_effect = Exception("API connection timeout")

        result = await extractor.extract("Tell Sarah about the project")

        assert not result.success
        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.entity_count == 0

    @pytest.mark.asyncio
    async def test_model_returns_non_list_non_dict(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Handle model returning something that's not a list or dict."""
        mock_client.generate.return_value = _make_gemini_response_raw('"just a string"')

        result = await extractor.extract("Something")

        assert result.success
        assert result.entity_count == 0


# ── ExtractionResult Tests ───────────────────────────────────────────


class TestExtractionResult:
    """Test ExtractionResult dataclass properties and methods."""

    def test_success_property(self) -> None:
        result = ExtractionResult(raw_text="hello")
        assert result.success

        result_err = ExtractionResult(raw_text="hello", error="failed")
        assert not result_err.success

    def test_entity_count(self) -> None:
        result = ExtractionResult(
            entities=[
                Entity(name="A", entity_type=EntityType.PERSON),
                Entity(name="B", entity_type=EntityType.PLACE),
            ]
        )
        assert result.entity_count == 2

    def test_entities_by_type(self) -> None:
        result = ExtractionResult(
            entities=[
                Entity(name="Sarah", entity_type=EntityType.PERSON),
                Entity(name="Bob", entity_type=EntityType.PERSON),
                Entity(name="Berlin", entity_type=EntityType.PLACE),
                Entity(name="Atlas", entity_type=EntityType.PROJECT),
            ]
        )

        assert len(result.entities_by_type(EntityType.PERSON)) == 2
        assert len(result.entities_by_type(EntityType.PLACE)) == 1
        assert len(result.entities_by_type(EntityType.PROJECT)) == 1
        assert len(result.entities_by_type(EntityType.ORGANIZATION)) == 0


# ── Extractor Configuration Tests ────────────────────────────────────


class TestExtractorConfig:
    """Test EntityExtractor configuration and behavior."""

    @pytest.mark.asyncio
    async def test_uses_fast_tier(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Verify extraction uses FAST (Flash-Lite) tier."""
        mock_client.generate.return_value = _make_gemini_response([])

        await extractor.extract("test input")

        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["tier"] == ModelTier.FAST

    @pytest.mark.asyncio
    async def test_uses_json_response_mime(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Verify extraction requests JSON response format."""
        mock_client.generate.return_value = _make_gemini_response([])

        await extractor.extract("test input")

        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["response_mime_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_uses_system_prompt(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Verify extraction sends the system prompt."""
        mock_client.generate.return_value = _make_gemini_response([])

        await extractor.extract("test input")

        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["system_instruction"] == ENTITY_EXTRACTION_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self, mock_client: AsyncMock) -> None:
        """Custom system prompt is used."""
        custom_prompt = "Extract only people."
        extractor = EntityExtractor(mock_client, system_prompt=custom_prompt)
        mock_client.generate.return_value = _make_gemini_response([])

        await extractor.extract("test")

        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["system_instruction"] == custom_prompt

    @pytest.mark.asyncio
    async def test_low_temperature(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Verify extraction uses low temperature for deterministic output."""
        mock_client.generate.return_value = _make_gemini_response([])

        await extractor.extract("test input")

        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["temperature"] <= 0.1

    @pytest.mark.asyncio
    async def test_stats_tracking(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Stats track extraction count and errors."""
        mock_client.generate.return_value = _make_gemini_response([])
        await extractor.extract("first")

        mock_client.generate.side_effect = Exception("fail")
        await extractor.extract("second")

        stats = extractor.stats
        assert stats["extraction_count"] == 2
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_latency_captured(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Latency from model response is captured in result."""
        mock_client.generate.return_value = _make_gemini_response([], latency_ms=123.4)

        result = await extractor.extract("test")

        assert result.latency_ms == 123.4

    @pytest.mark.asyncio
    async def test_raw_response_captured(self, extractor: EntityExtractor, mock_client: AsyncMock) -> None:
        """Raw model response text is captured for debugging."""
        mock_client.generate.return_value = _make_gemini_response(
            [{"name": "X", "entity_type": "person", "metadata": {}}]
        )

        result = await extractor.extract("test")

        assert result.model_response != ""
        assert "X" in result.model_response


# ── Integration Tests ────────────────────────────────────────────────


class TestEntityExtractionIntegration:
    """Integration tests validating the full extraction pipeline.

    These test realistic speech inputs and verify the complete flow:
    transcript → EntityExtractor → ExtractionResult → Entity objects.
    """

    @pytest.mark.asyncio
    async def test_pipeline_task_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Task-intent speech with person, project, org entities."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Marco", "entity_type": "person", "metadata": {"role": "designer"}},
            {"name": "the landing page redesign", "entity_type": "project", "metadata": {}},
            {"name": "Stripe", "entity_type": "organization", "metadata": {"relationship": "partner"}},
        ])

        result = await extractor.extract(
            "Remind me to send Marco the updated mockups for the landing page redesign "
            "and include the Stripe integration notes"
        )

        assert result.success
        assert result.entity_count == 3
        # Verify each entity has a valid UUID id
        for entity in result.entities:
            assert entity.id is not None
            uuid.UUID(entity.id)  # Validates UUID format

    @pytest.mark.asyncio
    async def test_pipeline_event_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Event-intent speech with person and place entities."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Rachel", "entity_type": "person", "metadata": {}},
            {"name": "Tom", "entity_type": "person", "metadata": {}},
            {"name": "Rooftop Bar", "entity_type": "place", "metadata": {"type": "venue"}},
        ])

        result = await extractor.extract(
            "Set up dinner with Rachel and Tom at the Rooftop Bar on Friday at 7pm"
        )

        assert result.success
        persons = result.entities_by_type(EntityType.PERSON)
        places = result.entities_by_type(EntityType.PLACE)
        assert len(persons) == 2
        assert len(places) == 1

    @pytest.mark.asyncio
    async def test_pipeline_idea_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Idea-intent speech with org and project entities."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Notion", "entity_type": "organization", "metadata": {"type": "tool"}},
            {"name": "our CRM", "entity_type": "project", "metadata": {}},
            {"name": "Salesforce", "entity_type": "organization", "metadata": {"type": "competitor"}},
        ])

        result = await extractor.extract(
            "What if we integrated Notion with our CRM? "
            "Salesforce does something similar but it's way too heavy"
        )

        assert result.success
        orgs = result.entities_by_type(EntityType.ORGANIZATION)
        assert len(orgs) == 2

    @pytest.mark.asyncio
    async def test_pipeline_question_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Question-intent speech extracting relevant entities."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Karen", "entity_type": "person", "metadata": {}},
            {"name": "budget proposal", "entity_type": "project", "metadata": {}},
        ])

        result = await extractor.extract(
            "What did Karen say about the budget proposal last week?"
        )

        assert result.success
        assert "Karen" in _entity_names(result)
        assert "budget proposal" in _entity_names(result)

    @pytest.mark.asyncio
    async def test_pipeline_journal_entry(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Journal-intent speech with personal/emotional context entities."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Anthropic", "entity_type": "organization", "metadata": {}},
            {"name": "the API project", "entity_type": "project", "metadata": {}},
            {"name": "Maya", "entity_type": "person", "metadata": {"relationship": "mentor"}},
        ])

        result = await extractor.extract(
            "Feeling really good about the progress at Anthropic. "
            "The API project is coming together and Maya's feedback was super helpful."
        )

        assert result.success
        assert result.entity_count == 3

    @pytest.mark.asyncio
    async def test_pipeline_update_with_entities(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Update-intent speech with status references."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "the database migration", "entity_type": "project", "metadata": {"status": "complete"}},
            {"name": "DevOps", "entity_type": "organization", "metadata": {"type": "team"}},
            {"name": "production", "entity_type": "place", "metadata": {"type": "environment"}},
        ])

        result = await extractor.extract(
            "Quick update: the database migration to production is done. "
            "DevOps signed off this morning."
        )

        assert result.success
        assert result.entity_count == 3

    @pytest.mark.asyncio
    async def test_pipeline_long_rambling_input(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Long, rambling natural speech with many entity mentions."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Jess", "entity_type": "person", "metadata": {}},
            {"name": "the onboarding redesign", "entity_type": "project", "metadata": {}},
            {"name": "Slack", "entity_type": "organization", "metadata": {"type": "tool"}},
            {"name": "the UX team", "entity_type": "organization", "metadata": {"type": "team"}},
            {"name": "New York", "entity_type": "place", "metadata": {"type": "city"}},
            {"name": "the Q3 roadmap", "entity_type": "project", "metadata": {}},
        ])

        result = await extractor.extract(
            "Okay so Jess pinged me on Slack about the onboarding redesign and "
            "apparently the UX team in New York has some thoughts. "
            "We should probably loop them in on the Q3 roadmap too while we're at it."
        )

        assert result.success
        assert result.entity_count == 6
        assert len(result.entities_by_type(EntityType.PERSON)) == 1
        assert len(result.entities_by_type(EntityType.PROJECT)) == 2
        assert len(result.entities_by_type(EntityType.ORGANIZATION)) == 2
        assert len(result.entities_by_type(EntityType.PLACE)) == 1

    @pytest.mark.asyncio
    async def test_pipeline_entities_have_unique_ids(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Each extracted entity gets a unique UUID."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "A", "entity_type": "person", "metadata": {}},
            {"name": "B", "entity_type": "person", "metadata": {}},
            {"name": "C", "entity_type": "place", "metadata": {}},
        ])

        result = await extractor.extract("A and B met at C")

        ids = [e.id for e in result.entities]
        assert len(set(ids)) == 3  # All unique

    @pytest.mark.asyncio
    async def test_pipeline_entity_type_distribution(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Verify entities_by_type returns correct distribution."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "P1", "entity_type": "person", "metadata": {}},
            {"name": "P2", "entity_type": "person", "metadata": {}},
            {"name": "L1", "entity_type": "place", "metadata": {}},
            {"name": "PR1", "entity_type": "project", "metadata": {}},
            {"name": "PR2", "entity_type": "project", "metadata": {}},
            {"name": "PR3", "entity_type": "project", "metadata": {}},
            {"name": "O1", "entity_type": "organization", "metadata": {}},
        ])

        result = await extractor.extract("complex input")

        assert len(result.entities_by_type(EntityType.PERSON)) == 2
        assert len(result.entities_by_type(EntityType.PLACE)) == 1
        assert len(result.entities_by_type(EntityType.PROJECT)) == 3
        assert len(result.entities_by_type(EntityType.ORGANIZATION)) == 1


# ── Diverse Natural Speech Patterns ──────────────────────────────────


class TestDiverseSpeechPatterns:
    """Test extraction from varied speech patterns, tones, and registers."""

    @pytest.mark.asyncio
    async def test_very_casual_speech(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract from very casual, spoken-word input."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Jake", "entity_type": "person", "metadata": {}},
            {"name": "that Kubernetes thing", "entity_type": "project", "metadata": {}},
        ])

        result = await extractor.extract(
            "ugh Jake keeps bugging me about that Kubernetes thing like dude I'm on it"
        )

        assert result.success
        assert result.entity_count == 2

    @pytest.mark.asyncio
    async def test_formal_business_speech(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract from formal business language."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Mr. Thompson", "entity_type": "person", "metadata": {"title": "Mr."}},
            {"name": "Henderson & Associates", "entity_type": "organization", "metadata": {}},
            {"name": "the Q4 financial review", "entity_type": "project", "metadata": {}},
            {"name": "our Chicago headquarters", "entity_type": "place", "metadata": {}},
        ])

        result = await extractor.extract(
            "Please schedule a call with Mr. Thompson from Henderson and Associates "
            "regarding the Q4 financial review at our Chicago headquarters"
        )

        assert result.success
        assert result.entity_count == 4

    @pytest.mark.asyncio
    async def test_interrupted_speech(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract from speech with false starts and corrections."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Anna", "entity_type": "person", "metadata": {}},
            {"name": "the sprint demo", "entity_type": "project", "metadata": {}},
        ])

        result = await extractor.extract(
            "I need to um wait no I should talk to Anna about um the sprint demo"
        )

        assert result.success
        assert result.entity_count == 2

    @pytest.mark.asyncio
    async def test_multilingual_names(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract entities with non-English names."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Yuki Tanaka", "entity_type": "person", "metadata": {}},
            {"name": "München", "entity_type": "place", "metadata": {"type": "city"}},
            {"name": "Müller GmbH", "entity_type": "organization", "metadata": {}},
        ])

        result = await extractor.extract(
            "Yuki Tanaka from Müller GmbH is visiting from München next week"
        )

        assert result.success
        assert result.entity_count == 3
        assert "Yuki Tanaka" in _entity_names(result)

    @pytest.mark.asyncio
    async def test_stream_of_consciousness(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract from stream-of-consciousness thinking aloud."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "the mobile app", "entity_type": "project", "metadata": {}},
            {"name": "React Native", "entity_type": "organization", "metadata": {"type": "framework"}},
            {"name": "Ben", "entity_type": "person", "metadata": {}},
        ])

        result = await extractor.extract(
            "thinking about the mobile app and whether we should go React Native "
            "or native native... Ben had some thoughts about this I should ask him"
        )

        assert result.success
        assert result.entity_count == 3

    @pytest.mark.asyncio
    async def test_negative_sentiment_speech(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract entities even from frustrated or negative speech."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "the CI pipeline", "entity_type": "project", "metadata": {}},
            {"name": "Jenkins", "entity_type": "organization", "metadata": {"type": "tool"}},
        ])

        result = await extractor.extract(
            "The CI pipeline is completely broken again. "
            "Jenkins keeps timing out and nobody seems to care."
        )

        assert result.success
        assert result.entity_count == 2

    @pytest.mark.asyncio
    async def test_reminder_with_context(
        self, extractor: EntityExtractor, mock_client: AsyncMock
    ) -> None:
        """Extract from a reminder-style blurt with rich context."""
        mock_client.generate.return_value = _make_gemini_response([
            {"name": "Dr. Patel", "entity_type": "person", "metadata": {"title": "Dr."}},
            {"name": "the clinic on Oak Street", "entity_type": "place", "metadata": {}},
        ])

        result = await extractor.extract(
            "Don't forget I have that appointment with Dr. Patel "
            "at the clinic on Oak Street tomorrow at 3"
        )

        assert result.success
        assert result.entity_count == 2
        types = _entity_types(result)
        assert types["Dr. Patel"] == EntityType.PERSON
        assert types["the clinic on Oak Street"] == EntityType.PLACE
