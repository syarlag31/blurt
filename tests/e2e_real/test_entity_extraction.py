"""Real E2E test: entity extraction for person and project entities.

Validates that the real Gemini API correctly extracts ``person`` and
``project`` entities from the sentence:

    "Tell Sarah the launch project is delayed"

Expected entities:
- **Sarah** → ``person``
- **launch project** (or similar) → ``project``
"""

from __future__ import annotations

import pytest

from blurt.core.entity_extractor import EntityExtractor, ExtractionResult
from blurt.core.models import EntityType


@pytest.mark.real_e2e
@pytest.mark.asyncio
async def test_extract_person_and_project_entities(
    entity_extractor: EntityExtractor,
) -> None:
    """Gemini should extract a person and a project from the input text."""
    text = "Tell Sarah the launch project is delayed"

    result: ExtractionResult = await entity_extractor.extract(text)

    # Extraction should succeed without errors.
    assert result.success, f"Entity extraction failed: {result.error}"
    assert result.entity_count >= 2, (
        f"Expected at least 2 entities, got {result.entity_count}: "
        f"{[(e.name, e.entity_type.value) for e in result.entities]}"
    )

    # Collect extracted types and lowered names for flexible assertions.
    names_lower = [e.name.lower() for e in result.entities]
    types_found = {e.entity_type for e in result.entities}

    # --- Person entity ---
    assert EntityType.PERSON in types_found, (
        f"No PERSON entity found. Entities: "
        f"{[(e.name, e.entity_type.value) for e in result.entities]}"
    )
    persons = result.entities_by_type(EntityType.PERSON)
    person_names = [p.name.lower() for p in persons]
    assert any("sarah" in n for n in person_names), (
        f"Expected 'Sarah' among person entities, got: {person_names}"
    )

    # --- Project entity ---
    assert EntityType.PROJECT in types_found, (
        f"No PROJECT entity found. Entities: "
        f"{[(e.name, e.entity_type.value) for e in result.entities]}"
    )
    projects = result.entities_by_type(EntityType.PROJECT)
    project_names = [p.name.lower() for p in projects]
    assert any("launch" in n for n in project_names), (
        f"Expected a project containing 'launch', got: {project_names}"
    )
