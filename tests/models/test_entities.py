"""Tests for entity data models and typed attribute schemas."""

from __future__ import annotations

from datetime import date


from blurt.models.entities import (
    ENTITY_ATTRIBUTE_SCHEMAS,
    EntityNode,
    EntityType,
    Fact,
    FactType,
    LearnedPattern,
    OrganizationAttributes,
    PatternType,
    PersonAttributes,
    PlaceAttributes,
    ProjectAttributes,
    ProjectStatus,
    RelationshipEdge,
    RelationshipType,
    SemanticSearchResult,
    is_valid_relationship,
    parse_typed_attributes,
    typed_attributes_to_dict,
)


# ── EntityType & EntityNode ──────────────────────────────────────────


class TestEntityType:
    def test_all_core_types_exist(self):
        assert EntityType.PERSON.value == "person"
        assert EntityType.PLACE.value == "place"
        assert EntityType.PROJECT.value == "project"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.TOPIC.value == "topic"
        assert EntityType.TOOL.value == "tool"

    def test_enum_from_string(self):
        assert EntityType("person") is EntityType.PERSON
        assert EntityType("project") is EntityType.PROJECT


class TestEntityNode:
    def test_create_minimal(self):
        node = EntityNode(user_id="u1", name="Alice", entity_type=EntityType.PERSON)
        assert node.name == "Alice"
        assert node.normalized_name == "alice"
        assert node.entity_type == EntityType.PERSON
        assert node.user_id == "u1"
        assert node.id  # auto-generated UUID

    def test_normalized_name_auto_set(self):
        node = EntityNode(user_id="u1", name="  Acme Corp  ", entity_type=EntityType.ORGANIZATION)
        assert node.normalized_name == "acme corp"

    def test_normalized_name_preserved_if_set(self):
        node = EntityNode(
            user_id="u1", name="Alice", normalized_name="custom", entity_type=EntityType.PERSON
        )
        assert node.normalized_name == "custom"

    def test_attributes_dict(self):
        node = EntityNode(
            user_id="u1",
            name="Bob",
            entity_type=EntityType.PERSON,
            attributes={"role": "manager", "team": "platform"},
        )
        assert node.attributes["role"] == "manager"

    def test_aliases(self):
        node = EntityNode(
            user_id="u1",
            name="Google",
            entity_type=EntityType.ORGANIZATION,
            aliases=["alphabet", "goog"],
        )
        assert "alphabet" in node.aliases

    def test_embedding_optional(self):
        node = EntityNode(user_id="u1", name="X", entity_type=EntityType.TOPIC)
        assert node.embedding is None


# ── Relationship Types ───────────────────────────────────────────────


class TestRelationshipType:
    def test_person_specific_types(self):
        assert RelationshipType.REPORTS_TO.value == "reports_to"
        assert RelationshipType.FRIEND_OF.value == "friend_of"
        assert RelationshipType.FAMILY_OF.value == "family_of"

    def test_org_specific_types(self):
        assert RelationshipType.MEMBER_OF.value == "member_of"
        assert RelationshipType.EMPLOYED_BY.value == "employed_by"
        assert RelationshipType.FOUNDED.value == "founded"

    def test_project_specific_types(self):
        assert RelationshipType.OWNS.value == "owns"
        assert RelationshipType.CONTRIBUTES_TO.value == "contributes_to"
        assert RelationshipType.DEPENDS_ON.value == "depends_on"
        assert RelationshipType.BLOCKED_BY.value == "blocked_by"

    def test_location_types(self):
        assert RelationshipType.BASED_IN.value == "based_in"
        assert RelationshipType.LOCATED_AT.value == "located_at"


class TestRelationshipEdge:
    def test_create(self):
        edge = RelationshipEdge(
            user_id="u1",
            source_entity_id="e1",
            target_entity_id="e2",
            relationship_type=RelationshipType.WORKS_WITH,
        )
        assert edge.strength == 1.0
        assert edge.co_mention_count == 1

    def test_strength_bounds(self):
        edge = RelationshipEdge(
            user_id="u1",
            source_entity_id="e1",
            target_entity_id="e2",
            relationship_type=RelationshipType.KNOWS,
            strength=100.0,
        )
        assert edge.strength == 100.0


# ── PersonAttributes ─────────────────────────────────────────────────


class TestPersonAttributes:
    def test_default_schema_tag(self):
        attrs = PersonAttributes()
        assert attrs.entity_schema == "person"

    def test_all_fields_optional(self):
        attrs = PersonAttributes()
        assert attrs.first_name is None
        assert attrs.role is None
        assert attrs.email is None

    def test_populated(self):
        attrs = PersonAttributes(
            first_name="Sarah",
            last_name="Chen",
            role="engineering manager",
            relationship_to_user="manager",
            timezone="America/Los_Angeles",
            communication_style="direct",
            interaction_frequency="daily",
        )
        assert attrs.first_name == "Sarah"
        assert attrs.role == "engineering manager"
        assert attrs.interaction_frequency == "daily"

    def test_extra_captures_overflow(self):
        attrs = PersonAttributes(extra={"custom_field": "value"})
        assert attrs.extra["custom_field"] == "value"

    def test_birthday(self):
        attrs = PersonAttributes(birthday=date(1990, 5, 15))
        assert attrs.birthday == date(1990, 5, 15)


# ── PlaceAttributes ──────────────────────────────────────────────────


class TestPlaceAttributes:
    def test_default_schema_tag(self):
        attrs = PlaceAttributes()
        assert attrs.entity_schema == "place"

    def test_location_fields(self):
        attrs = PlaceAttributes(
            address="123 Main St",
            city="San Francisco",
            state="CA",
            country="US",
            coordinates=(37.7749, -122.4194),
        )
        assert attrs.city == "San Francisco"
        assert attrs.coordinates == (37.7749, -122.4194)

    def test_place_type(self):
        attrs = PlaceAttributes(
            place_type="office",
            category="work",
            relationship_to_user="workplace",
        )
        assert attrs.place_type == "office"

    def test_associated_activities(self):
        attrs = PlaceAttributes(
            associated_activities=["standup", "lunch", "deep work"],
        )
        assert len(attrs.associated_activities) == 3


# ── ProjectAttributes ────────────────────────────────────────────────


class TestProjectAttributes:
    def test_default_status_active(self):
        attrs = ProjectAttributes()
        assert attrs.status == ProjectStatus.ACTIVE

    def test_shame_free_language(self):
        # "target_date" not "deadline" — shame-free design
        attrs = ProjectAttributes(target_date=date(2026, 6, 1))
        assert attrs.target_date == date(2026, 6, 1)
        assert not hasattr(attrs, "deadline")
        assert not hasattr(attrs, "overdue")

    def test_project_lifecycle(self):
        for status in ProjectStatus:
            attrs = ProjectAttributes(status=status)
            assert attrs.status == status

    def test_stakeholders(self):
        attrs = ProjectAttributes(
            owner="Alice",
            stakeholders=["Bob", "Charlie"],
            team="Platform",
        )
        assert len(attrs.stakeholders) == 2

    def test_external_links(self):
        attrs = ProjectAttributes(
            notion_page_id="abc123",
            calendar_event_ids=["ev1", "ev2"],
        )
        assert attrs.notion_page_id == "abc123"

    def test_user_energy_and_momentum(self):
        attrs = ProjectAttributes(
            user_energy_association="energizing",
            momentum="gaining",
        )
        assert attrs.user_energy_association == "energizing"


# ── OrganizationAttributes ───────────────────────────────────────────


class TestOrganizationAttributes:
    def test_default_schema_tag(self):
        attrs = OrganizationAttributes()
        assert attrs.entity_schema == "organization"

    def test_full_org(self):
        attrs = OrganizationAttributes(
            full_name="Acme Corporation",
            org_type="company",
            industry="technology",
            relationship_to_user="employer",
            user_role="senior engineer",
            key_contacts=["Sarah", "Bob"],
            headquarters="San Francisco",
        )
        assert attrs.org_type == "company"
        assert attrs.user_role == "senior engineer"
        assert len(attrs.key_contacts) == 2


# ── parse_typed_attributes ───────────────────────────────────────────


class TestParseTypedAttributes:
    def test_parse_person(self):
        raw = {"first_name": "Alice", "role": "designer", "unknown_field": 42}
        result = parse_typed_attributes(EntityType.PERSON, raw)
        assert isinstance(result, PersonAttributes)
        assert result.first_name == "Alice"
        assert result.role == "designer"
        assert result.extra["unknown_field"] == 42

    def test_parse_place(self):
        raw = {"city": "NYC", "place_type": "office"}
        result = parse_typed_attributes(EntityType.PLACE, raw)
        assert isinstance(result, PlaceAttributes)
        assert result.city == "NYC"

    def test_parse_project(self):
        raw = {"status": "on_hold", "owner": "Bob"}
        result = parse_typed_attributes(EntityType.PROJECT, raw)
        assert isinstance(result, ProjectAttributes)
        assert result.status == ProjectStatus.ON_HOLD

    def test_parse_organization(self):
        raw = {"org_type": "team", "team_size": 8}
        result = parse_typed_attributes(EntityType.ORGANIZATION, raw)
        assert isinstance(result, OrganizationAttributes)
        assert result.team_size == 8

    def test_returns_none_for_topic(self):
        assert parse_typed_attributes(EntityType.TOPIC, {"key": "val"}) is None

    def test_returns_none_for_tool(self):
        assert parse_typed_attributes(EntityType.TOOL, {}) is None

    def test_empty_dict(self):
        result = parse_typed_attributes(EntityType.PERSON, {})
        assert isinstance(result, PersonAttributes)
        assert result.first_name is None


# ── typed_attributes_to_dict ─────────────────────────────────────────


class TestTypedAttributesToDict:
    def test_roundtrip_person(self):
        original = {"first_name": "Alice", "role": "designer", "custom": 42}
        parsed = parse_typed_attributes(EntityType.PERSON, original)
        assert parsed is not None
        result = typed_attributes_to_dict(parsed)
        assert result["first_name"] == "Alice"
        assert result["role"] == "designer"
        assert result["custom"] == 42
        # None fields excluded
        assert "last_name" not in result

    def test_roundtrip_project(self):
        original = {"status": "active", "owner": "Bob", "tags": ["urgent"]}
        parsed = parse_typed_attributes(EntityType.PROJECT, original)
        assert parsed is not None
        result = typed_attributes_to_dict(parsed)
        assert result["status"] == "active"
        assert result["tags"] == ["urgent"]


# ── Relationship Validation ──────────────────────────────────────────


class TestRelationshipValidation:
    def test_valid_person_person(self):
        assert is_valid_relationship(
            EntityType.PERSON, EntityType.PERSON, RelationshipType.WORKS_WITH
        )

    def test_valid_person_org(self):
        assert is_valid_relationship(
            EntityType.PERSON, EntityType.ORGANIZATION, RelationshipType.EMPLOYED_BY
        )

    def test_valid_person_project(self):
        assert is_valid_relationship(
            EntityType.PERSON, EntityType.PROJECT, RelationshipType.COLLABORATES_ON
        )

    def test_valid_project_project(self):
        assert is_valid_relationship(
            EntityType.PROJECT, EntityType.PROJECT, RelationshipType.DEPENDS_ON
        )

    def test_valid_org_place(self):
        assert is_valid_relationship(
            EntityType.ORGANIZATION, EntityType.PLACE, RelationshipType.BASED_IN
        )

    def test_invalid_place_manages_person(self):
        assert not is_valid_relationship(
            EntityType.PLACE, EntityType.PERSON, RelationshipType.MANAGES
        )

    def test_mentioned_with_universal(self):
        # MENTIONED_WITH is valid for any pair
        for et1 in EntityType:
            for et2 in EntityType:
                assert is_valid_relationship(et1, et2, RelationshipType.MENTIONED_WITH)

    def test_related_to_universal(self):
        for et1 in EntityType:
            for et2 in EntityType:
                assert is_valid_relationship(et1, et2, RelationshipType.RELATED_TO)


# ── Schema Registry ─────────────────────────────────────────────────


class TestSchemaRegistry:
    def test_four_core_types_registered(self):
        assert EntityType.PERSON in ENTITY_ATTRIBUTE_SCHEMAS
        assert EntityType.PLACE in ENTITY_ATTRIBUTE_SCHEMAS
        assert EntityType.PROJECT in ENTITY_ATTRIBUTE_SCHEMAS
        assert EntityType.ORGANIZATION in ENTITY_ATTRIBUTE_SCHEMAS

    def test_topic_not_registered(self):
        assert EntityType.TOPIC not in ENTITY_ATTRIBUTE_SCHEMAS

    def test_tool_not_registered(self):
        assert EntityType.TOOL not in ENTITY_ATTRIBUTE_SCHEMAS


# ── Existing models still work ───────────────────────────────────────


class TestExistingModelsUnchanged:
    """Ensure backward compatibility with models created by previous ACs."""

    def test_fact_model(self):
        fact = Fact(user_id="u1", fact_type=FactType.ATTRIBUTE, content="Alice is my manager")
        assert fact.confidence == 1.0
        assert fact.is_active

    def test_learned_pattern(self):
        pattern = LearnedPattern(
            user_id="u1",
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="User prefers deep work in the morning",
        )
        assert pattern.confidence == 0.5

    def test_semantic_search_result(self):
        result = SemanticSearchResult(
            item_type="entity",
            item_id="e1",
            content="person: Alice",
            similarity_score=0.95,
        )
        assert result.similarity_score == 0.95


# ── Core Entity model alignment ────────────────────────────────────


class TestCoreEntityAlignment:
    """Verify that core/models.Entity aligns with models/entities.EntityNode."""

    def test_core_entity_type_has_all_types(self):
        """core/models.EntityType must include TOPIC and TOOL."""
        from blurt.core.models import EntityType as CoreEntityType

        assert CoreEntityType.PERSON.value == "person"
        assert CoreEntityType.PLACE.value == "place"
        assert CoreEntityType.PROJECT.value == "project"
        assert CoreEntityType.ORGANIZATION.value == "organization"
        assert CoreEntityType.TOPIC.value == "topic"
        assert CoreEntityType.TOOL.value == "tool"

    def test_core_entity_type_values_match_graph_entity_type(self):
        """All values in core EntityType exist in graph EntityType."""
        from blurt.core.models import EntityType as CoreEntityType

        for core_et in CoreEntityType:
            graph_et = EntityType(core_et.value)
            assert graph_et.value == core_et.value

    def test_core_entity_has_confidence(self):
        from blurt.core.models import Entity as CoreEntity, EntityType as CoreET

        entity = CoreEntity(name="Alice", entity_type=CoreET.PERSON)
        assert entity.confidence == 1.0

    def test_core_entity_has_aliases(self):
        from blurt.core.models import Entity as CoreEntity, EntityType as CoreET

        entity = CoreEntity(
            name="Google", entity_type=CoreET.ORGANIZATION, aliases=["Alphabet"]
        )
        assert "Alphabet" in entity.aliases

    def test_core_entity_has_source_blurt_id(self):
        from blurt.core.models import Entity as CoreEntity, EntityType as CoreET

        entity = CoreEntity(
            name="Atlas",
            entity_type=CoreET.PROJECT,
            source_blurt_id="blurt-123",
        )
        assert entity.source_blurt_id == "blurt-123"

    def test_to_entity_node_kwargs(self):
        """Entity.to_entity_node_kwargs produces valid EntityNode constructor args."""
        from blurt.core.models import Entity as CoreEntity, EntityType as CoreET

        entity = CoreEntity(
            name="Sarah Chen",
            entity_type=CoreET.PERSON,
            metadata={"role": "manager", "team": "platform"},
            aliases=["Sarah"],
        )
        kwargs = entity.to_entity_node_kwargs(user_id="u1")

        assert kwargs["user_id"] == "u1"
        assert kwargs["name"] == "Sarah Chen"
        assert kwargs["entity_type"] == EntityType.PERSON
        assert kwargs["aliases"] == ["Sarah"]
        assert kwargs["attributes"]["role"] == "manager"

        # Must be valid for EntityNode construction
        node = EntityNode(**kwargs)
        assert node.name == "Sarah Chen"
        assert node.normalized_name == "sarah chen"
        assert node.entity_type == EntityType.PERSON

    def test_to_entity_node_kwargs_with_typed_attributes(self):
        """Metadata from Entity can be parsed into typed PersonAttributes."""
        from blurt.core.models import Entity as CoreEntity, EntityType as CoreET

        entity = CoreEntity(
            name="Bob",
            entity_type=CoreET.PERSON,
            metadata={"first_name": "Bob", "role": "designer", "custom_key": "val"},
        )
        kwargs = entity.to_entity_node_kwargs(user_id="u1")
        node = EntityNode(**kwargs)

        # The raw attributes can be parsed into typed PersonAttributes
        typed = parse_typed_attributes(node.entity_type, node.attributes)
        assert isinstance(typed, PersonAttributes)
        assert typed.first_name == "Bob"
        assert typed.role == "designer"
        assert typed.extra["custom_key"] == "val"
