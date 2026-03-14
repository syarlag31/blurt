"""Entity extraction service for Blurt.

Extracts entities (person, place, project, organization) from natural speech
transcripts using Gemini Flash-Lite. Zero-friction: the user never has to
tag or categorize anything — extraction happens silently in the pipeline.

Uses the FAST model tier (Flash-Lite) for cost-efficient, low-latency
extraction suitable for real-time voice processing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from blurt.clients.gemini import GeminiClient, GeminiResponse, ModelTier
from blurt.core.models import Entity, EntityType

logger = logging.getLogger(__name__)

# System instruction for entity extraction via Gemini Flash-Lite
ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction engine for a personal knowledge assistant.
Extract all entities from the user's natural speech transcript.

Entity types:
- person: People mentioned by name, title, or role (e.g., "Sarah", "my manager", "Dr. Chen")
- place: Physical locations, addresses, venues, cities, countries (e.g., "the office", "Berlin", "Room 302")
- project: Named projects, initiatives, deliverables, codenames (e.g., "Project Atlas", "the Q2 deck", "v2 migration")
- organization: Companies, teams, departments, institutions (e.g., "Google", "the marketing team", "MIT")
- topic: Abstract subjects, themes, or areas of knowledge (e.g., "machine learning", "Q2 planning", "design systems")
- tool: Software, apps, platforms, or technologies (e.g., "Notion", "Slack", "Figma", "Google Calendar")

Rules:
1. Extract ALL entities, even if mentioned casually or indirectly.
2. Normalize names: capitalize properly, expand obvious abbreviations.
3. For ambiguous references like "the team" or "the office", still extract them with the best entity_type.
4. Include role/relationship info in metadata when available (e.g., "my manager" → metadata: {"role": "manager", "relationship": "reports_to"}).
5. Include temporal context in metadata when relevant (e.g., "new hire" → metadata: {"temporal": "recent"}).
6. Do NOT extract generic nouns that aren't specific entities (e.g., "meeting", "idea", "thing").
7. Return an empty list if no entities are found.

Respond with ONLY a JSON array of objects, each with:
- "name": string (the entity's name or reference)
- "entity_type": string (one of: "person", "place", "project", "organization", "topic", "tool")
- "metadata": object (optional additional context)

Example input: "I talked to Sarah about the Atlas project at the downtown office yesterday"
Example output:
[
  {"name": "Sarah", "entity_type": "person", "metadata": {}},
  {"name": "Atlas", "entity_type": "project", "metadata": {}},
  {"name": "downtown office", "entity_type": "place", "metadata": {}}
]
"""


@dataclass
class ExtractionResult:
    """Result of entity extraction from a transcript.

    Attributes:
        entities: List of extracted entities.
        raw_text: The original transcript text.
        model_response: Raw response from Gemini (for debugging).
        latency_ms: Extraction latency in milliseconds.
        error: Error message if extraction failed, None otherwise.
    """

    entities: list[Entity] = field(default_factory=list)
    raw_text: str = ""
    model_response: str = ""
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether extraction completed without error."""
        return self.error is None

    @property
    def entity_count(self) -> int:
        """Number of entities extracted."""
        return len(self.entities)

    def entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Filter entities by type."""
        return [e for e in self.entities if e.entity_type == entity_type]


class EntityExtractor:
    """Extracts entities from natural speech transcripts using Gemini Flash-Lite.

    Usage::

        extractor = EntityExtractor(gemini_client)
        result = await extractor.extract("Tell Sarah about the Atlas project")
        for entity in result.entities:
            print(f"{entity.name} ({entity.entity_type})")

    The extractor uses the FAST tier (Flash-Lite) for low-latency extraction
    suitable for the real-time voice pipeline.
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        system_prompt: str = ENTITY_EXTRACTION_SYSTEM_PROMPT,
        temperature: float = 0.05,
        max_output_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._extraction_count = 0
        self._error_count = 0

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities from a natural speech transcript.

        Args:
            text: The transcript text to extract entities from.

        Returns:
            ExtractionResult with extracted entities and metadata.
        """
        if not text or not text.strip():
            return ExtractionResult(raw_text=text)

        self._extraction_count += 1
        response = None

        try:
            response = await self._client.generate(
                prompt=text.strip(),
                tier=ModelTier.FAST,
                system_instruction=self._system_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                response_mime_type="application/json",
            )

            entities = self._parse_response(response)

            return ExtractionResult(
                entities=entities,
                raw_text=text,
                model_response=response.text,
                latency_ms=response.latency_ms,
            )

        except json.JSONDecodeError as e:
            self._error_count += 1
            logger.warning("Failed to parse entity extraction response: %s", e)
            return ExtractionResult(
                raw_text=text,
                model_response=getattr(response, "text", ""),  # noqa: F821
                error=f"JSON parse error: {e}",
            )
        except Exception as e:
            self._error_count += 1
            logger.error("Entity extraction failed: %s", e)
            return ExtractionResult(
                raw_text=text,
                error=str(e),
            )

    def _parse_response(self, response: GeminiResponse) -> list[Entity]:
        """Parse a Gemini response into Entity objects.

        Handles various response formats robustly:
        - JSON array of entity objects
        - JSON object with an "entities" key
        - Markdown-wrapped JSON
        """
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [line for line in lines[1:] if not line.strip().startswith("```")]
            raw = "\n".join(lines)

        parsed = json.loads(raw)

        # Handle {"entities": [...]} wrapper
        if isinstance(parsed, dict):
            parsed = parsed.get("entities", [])

        if not isinstance(parsed, list):
            logger.warning("Unexpected response format: %s", type(parsed))
            return []

        entities: list[Entity] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "").strip()
            raw_type = item.get("entity_type", "").strip().lower()
            metadata = item.get("metadata", {})

            if not name:
                continue

            # Validate entity type
            try:
                entity_type = EntityType(raw_type)
            except ValueError:
                logger.debug("Skipping unknown entity type: %s", raw_type)
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            entities.append(
                Entity(
                    name=name,
                    entity_type=entity_type,
                    metadata=metadata,
                )
            )

        return entities

    @property
    def stats(self) -> dict[str, Any]:
        """Extraction statistics."""
        return {
            "extraction_count": self._extraction_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._extraction_count
                if self._extraction_count > 0
                else 0.0
            ),
        }
