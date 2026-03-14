"""Core entity extraction service for Blurt.

Processes natural speech text and identifies entities across all types
(person, place, project, organization, topic, tool) using Gemini Flash-Lite
for fast, low-cost extraction. Also extracts relationships between entities,
facts/preferences, and temporal references.

This is the primary NLP pipeline stage that feeds the semantic memory tier.
Every blurt flows through this extractor before entities are stored in the
knowledge graph.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from blurt.clients.gemini import GeminiClient, GeminiResponse, ModelTier
from blurt.models.entities import EntityType, FactType, RelationshipType

logger = logging.getLogger(__name__)


# ── Extraction result models ─────────────────────────────────────

class TemporalType(str, Enum):
    """Types of temporal references in speech."""

    ABSOLUTE = "absolute"     # "March 15th", "next Tuesday"
    RELATIVE = "relative"     # "tomorrow", "in 2 hours"
    RECURRING = "recurring"   # "every Monday", "weekly"
    DURATION = "duration"     # "for 30 minutes", "2 hours"
    VAGUE = "vague"           # "soon", "later", "sometime"


@dataclass
class ExtractedEntity:
    """An entity identified from natural speech.

    Attributes:
        name: The entity name as mentioned in speech.
        entity_type: Classification of the entity type.
        normalized_name: Lowercase, stripped canonical name.
        aliases: Alternate names or references found in context.
        attributes: Key-value properties extracted about the entity.
        confidence: Extraction confidence score (0.0-1.0).
        span: Character span (start, end) in the original text.
        context_snippet: Surrounding text providing context.
    """

    name: str
    entity_type: EntityType
    normalized_name: str = ""
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    span: tuple[int, int] | None = None
    context_snippet: str = ""

    def __post_init__(self) -> None:
        if not self.normalized_name:
            self.normalized_name = self.name.lower().strip()


@dataclass
class ExtractedRelationship:
    """A relationship between two entities identified from speech.

    Attributes:
        source_name: Name of the source entity.
        target_name: Name of the target entity.
        relationship_type: The type of relationship.
        context_snippet: The text passage indicating this relationship.
        confidence: Extraction confidence (0.0-1.0).
    """

    source_name: str
    target_name: str
    relationship_type: RelationshipType
    context_snippet: str = ""
    confidence: float = 1.0


@dataclass
class ExtractedFact:
    """A fact or preference extracted from speech.

    Attributes:
        content: The fact statement.
        fact_type: Classification of the fact type.
        subject_entity_name: Entity the fact is about (if any).
        confidence: Extraction confidence (0.0-1.0).
    """

    content: str
    fact_type: FactType
    subject_entity_name: str | None = None
    confidence: float = 1.0


@dataclass
class TemporalReference:
    """A temporal reference extracted from speech.

    Attributes:
        text: The original temporal expression.
        temporal_type: Classification of the temporal reference.
        resolved_datetime: Parsed datetime if resolvable.
        is_deadline: Whether this is a deadline/due date.
    """

    text: str
    temporal_type: TemporalType
    resolved_datetime: datetime | None = None
    is_deadline: bool = False


@dataclass
class ExtractionResult:
    """Complete extraction result from processing a blurt.

    Contains all entities, relationships, facts, and temporal references
    identified in the input text. This feeds directly into the semantic
    memory tier for knowledge graph construction.

    Attributes:
        entities: All identified entities.
        relationships: Relationships between entities.
        facts: Facts/preferences learned from speech.
        temporal_references: Time expressions found.
        raw_text: The original input text.
        model_used: Which model performed the extraction.
        latency_ms: Extraction latency in milliseconds.
    """

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    temporal_references: list[TemporalReference] = field(default_factory=list)
    raw_text: str = ""
    model_used: str = ""
    latency_ms: float = 0.0

    @property
    def entity_count(self) -> int:
        """Total number of entities extracted."""
        return len(self.entities)

    @property
    def has_entities(self) -> bool:
        """Whether any entities were found."""
        return len(self.entities) > 0

    @property
    def entity_types_found(self) -> set[EntityType]:
        """Set of entity types present in the extraction."""
        return {e.entity_type for e in self.entities}

    def entities_by_type(self, entity_type: EntityType) -> list[ExtractedEntity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]


# ── Extraction prompt ─────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction system for a personal AI assistant. \
Extract structured information from natural speech transcriptions.

Your job is to identify:
1. **Entities**: People, places, projects, organizations, topics, and tools mentioned.
2. **Relationships**: How entities relate to each other or to the speaker.
3. **Facts**: Statements of fact, preferences, habits, associations, or aliases.
4. **Temporal references**: Any time expressions (dates, deadlines, durations, recurring patterns).

Rules:
- Extract entities as they are naturally mentioned; do not infer entities not present.
- "I", "me", "my" refer to the speaker — do not extract as entities.
- Classify each entity into exactly one type: person, place, project, organization, topic, tool.
- For relationships, use the speaker as implicit source when they describe their own relationships.
- Confidence should reflect how clearly the entity/fact is stated (1.0 = explicit, 0.5 = implied).
- Return valid JSON only, no markdown fences or commentary.
"""

_EXTRACTION_USER_PROMPT = """\
Extract all entities, relationships, facts, and temporal references from this speech:

"{text}"

Return a JSON object with this exact structure:
{{
  "entities": [
    {{
      "name": "string",
      "type": "person|place|project|organization|topic|tool",
      "aliases": ["optional alternate names"],
      "attributes": {{"key": "value pairs"}},
      "confidence": 0.0-1.0
    }}
  ],
  "relationships": [
    {{
      "source": "entity name or 'speaker'",
      "target": "entity name",
      "type": "works_with|manages|managed_by|part_of|contains|related_to|located_at|knows|collaborates_on|mentioned_with",
      "confidence": 0.0-1.0
    }}
  ],
  "facts": [
    {{
      "content": "the fact statement",
      "type": "attribute|preference|habit|association|alias",
      "subject": "entity name or null",
      "confidence": 0.0-1.0
    }}
  ],
  "temporal_references": [
    {{
      "text": "the time expression",
      "type": "absolute|relative|recurring|duration|vague"
    }}
  ]
}}
"""


# ── Type mapping helpers ──────────────────────────────────────────

_ENTITY_TYPE_MAP: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "place": EntityType.PLACE,
    "project": EntityType.PROJECT,
    "organization": EntityType.ORGANIZATION,
    "topic": EntityType.TOPIC,
    "tool": EntityType.TOOL,
}

_RELATIONSHIP_TYPE_MAP: dict[str, RelationshipType] = {
    "works_with": RelationshipType.WORKS_WITH,
    "manages": RelationshipType.MANAGES,
    "managed_by": RelationshipType.MANAGED_BY,
    "part_of": RelationshipType.PART_OF,
    "contains": RelationshipType.CONTAINS,
    "related_to": RelationshipType.RELATED_TO,
    "located_at": RelationshipType.LOCATED_AT,
    "knows": RelationshipType.KNOWS,
    "collaborates_on": RelationshipType.COLLABORATES_ON,
    "mentioned_with": RelationshipType.MENTIONED_WITH,
}

_FACT_TYPE_MAP: dict[str, FactType] = {
    "attribute": FactType.ATTRIBUTE,
    "preference": FactType.PREFERENCE,
    "habit": FactType.HABIT,
    "association": FactType.ASSOCIATION,
    "alias": FactType.ALIAS,
}

_TEMPORAL_TYPE_MAP: dict[str, TemporalType] = {
    "absolute": TemporalType.ABSOLUTE,
    "relative": TemporalType.RELATIVE,
    "recurring": TemporalType.RECURRING,
    "duration": TemporalType.DURATION,
    "vague": TemporalType.VAGUE,
}


# ── Entity extraction service ────────────────────────────────────

class EntityExtractionError(Exception):
    """Raised when entity extraction fails."""

    pass


class EntityExtractor:
    """Core entity extraction service using Gemini Flash-Lite.

    Processes natural speech text and identifies entities, relationships,
    facts, and temporal references. Uses the FAST model tier (Flash-Lite)
    for cost-effective, low-latency extraction.

    Usage::

        async with GeminiClient(config) as client:
            extractor = EntityExtractor(client)
            result = await extractor.extract("Sarah and I are meeting at Notion HQ tomorrow")
            for entity in result.entities:
                print(f"{entity.name} ({entity.entity_type.value})")

    The extractor handles:
    - All 6 entity types: person, place, project, organization, topic, tool
    - Inter-entity relationships with typed edges
    - Fact/preference/habit extraction
    - Temporal reference detection
    - Graceful error handling with fallback to regex-based extraction
    """

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        *,
        model_tier: ModelTier = ModelTier.FAST,
        min_confidence: float = 0.3,
        max_entities_per_extraction: int = 50,
    ) -> None:
        """Initialize the entity extractor.

        Args:
            gemini_client: Gemini API client. If None, uses fallback extraction.
            model_tier: Which model tier to use (default: FAST/Flash-Lite).
            min_confidence: Minimum confidence threshold for including entities.
            max_entities_per_extraction: Safety limit on entities per extraction.
        """
        self._client = gemini_client
        self._model_tier = model_tier
        self._min_confidence = min_confidence
        self._max_entities = max_entities_per_extraction

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and structured data from natural speech text.

        This is the main entry point. Sends text to Gemini Flash-Lite for
        NLP-based extraction, with fallback to regex-based extraction if
        the LLM is unavailable.

        Args:
            text: Natural speech text to analyze.

        Returns:
            ExtractionResult containing all identified entities, relationships,
            facts, and temporal references.

        Raises:
            EntityExtractionError: If extraction fails irrecoverably.
        """
        if not text or not text.strip():
            return ExtractionResult(raw_text=text)

        text = text.strip()

        # Try LLM-based extraction first
        if self._client is not None and self._client.is_connected:
            try:
                return await self._extract_with_llm(text)
            except Exception as e:
                logger.warning(
                    "LLM extraction failed, falling back to regex: %s", e
                )

        # Fallback: regex-based extraction
        return self._extract_with_regex(text)

    async def extract_batch(
        self,
        texts: list[str],
    ) -> list[ExtractionResult]:
        """Extract entities from multiple texts.

        Args:
            texts: List of speech texts to analyze.

        Returns:
            List of ExtractionResult, one per input text.
        """
        import asyncio

        tasks = [self.extract(text) for text in texts]
        return list(await asyncio.gather(*tasks))

    # ── LLM-based extraction ─────────────────────────────────────

    async def _extract_with_llm(self, text: str) -> ExtractionResult:
        """Extract entities using Gemini Flash-Lite.

        Sends a structured extraction prompt and parses the JSON response.
        """
        assert self._client is not None

        prompt = _EXTRACTION_USER_PROMPT.format(text=text)

        response: GeminiResponse = await self._client.generate(
            prompt=prompt,
            tier=self._model_tier,
            system_instruction=_EXTRACTION_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        try:
            data = self._parse_llm_response(response.text)
        except (json.JSONDecodeError, ValueError) as e:
            raise EntityExtractionError(
                f"Failed to parse LLM extraction response: {e}"
            ) from e

        return self._build_result_from_llm(data, text, response)

    def _parse_llm_response(self, response_text: str) -> dict[str, Any]:
        """Parse and validate the JSON response from the LLM.

        Handles common LLM output quirks like markdown fences and
        trailing commas.
        """
        cleaned = response_text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            # Remove ```json or ``` prefix and ``` suffix
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Remove trailing commas before } or ] (common LLM error)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        data = json.loads(cleaned)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        return data

    def _build_result_from_llm(
        self,
        data: dict[str, Any],
        raw_text: str,
        response: GeminiResponse,
    ) -> ExtractionResult:
        """Build an ExtractionResult from parsed LLM JSON output."""
        entities = self._parse_entities(data.get("entities", []))
        relationships = self._parse_relationships(data.get("relationships", []))
        facts = self._parse_facts(data.get("facts", []))
        temporals = self._parse_temporal_references(
            data.get("temporal_references", [])
        )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            facts=facts,
            temporal_references=temporals,
            raw_text=raw_text,
            model_used=response.model,
            latency_ms=response.latency_ms,
        )

    def _parse_entities(
        self, raw_entities: list[dict[str, Any]]
    ) -> list[ExtractedEntity]:
        """Parse entity objects from LLM output."""
        entities: list[ExtractedEntity] = []

        for item in raw_entities[: self._max_entities]:
            try:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue

                type_str = str(item.get("type", "")).lower().strip()
                entity_type = _ENTITY_TYPE_MAP.get(type_str)
                if entity_type is None:
                    logger.debug("Unknown entity type '%s', defaulting to TOPIC", type_str)
                    entity_type = EntityType.TOPIC

                confidence = min(max(float(item.get("confidence", 0.8)), 0.0), 1.0)
                if confidence < self._min_confidence:
                    continue

                entities.append(
                    ExtractedEntity(
                        name=name,
                        entity_type=entity_type,
                        aliases=item.get("aliases", []) or [],
                        attributes=item.get("attributes", {}) or {},
                        confidence=confidence,
                    )
                )
            except (TypeError, ValueError) as e:
                logger.debug("Skipping malformed entity: %s — %s", item, e)

        return entities

    def _parse_relationships(
        self, raw_relationships: list[dict[str, Any]]
    ) -> list[ExtractedRelationship]:
        """Parse relationship objects from LLM output."""
        relationships: list[ExtractedRelationship] = []

        for item in raw_relationships:
            try:
                source = str(item.get("source", "")).strip()
                target = str(item.get("target", "")).strip()
                if not source or not target:
                    continue

                type_str = str(item.get("type", "")).lower().strip()
                rel_type = _RELATIONSHIP_TYPE_MAP.get(type_str)
                if rel_type is None:
                    rel_type = RelationshipType.RELATED_TO

                confidence = min(max(float(item.get("confidence", 0.8)), 0.0), 1.0)
                if confidence < self._min_confidence:
                    continue

                relationships.append(
                    ExtractedRelationship(
                        source_name=source,
                        target_name=target,
                        relationship_type=rel_type,
                        confidence=confidence,
                    )
                )
            except (TypeError, ValueError) as e:
                logger.debug("Skipping malformed relationship: %s — %s", item, e)

        return relationships

    def _parse_facts(
        self, raw_facts: list[dict[str, Any]]
    ) -> list[ExtractedFact]:
        """Parse fact objects from LLM output."""
        facts: list[ExtractedFact] = []

        for item in raw_facts:
            try:
                content = str(item.get("content", "")).strip()
                if not content:
                    continue

                type_str = str(item.get("type", "")).lower().strip()
                fact_type = _FACT_TYPE_MAP.get(type_str)
                if fact_type is None:
                    fact_type = FactType.ATTRIBUTE

                confidence = min(max(float(item.get("confidence", 0.8)), 0.0), 1.0)
                if confidence < self._min_confidence:
                    continue

                subject = item.get("subject")
                if subject is not None:
                    subject = str(subject).strip() or None

                facts.append(
                    ExtractedFact(
                        content=content,
                        fact_type=fact_type,
                        subject_entity_name=subject,
                        confidence=confidence,
                    )
                )
            except (TypeError, ValueError) as e:
                logger.debug("Skipping malformed fact: %s — %s", item, e)

        return facts

    def _parse_temporal_references(
        self, raw_temporals: list[dict[str, Any]]
    ) -> list[TemporalReference]:
        """Parse temporal reference objects from LLM output."""
        temporals: list[TemporalReference] = []

        for item in raw_temporals:
            try:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                type_str = str(item.get("type", "")).lower().strip()
                temporal_type = _TEMPORAL_TYPE_MAP.get(type_str)
                if temporal_type is None:
                    temporal_type = TemporalType.VAGUE

                temporals.append(
                    TemporalReference(
                        text=text,
                        temporal_type=temporal_type,
                    )
                )
            except (TypeError, ValueError) as e:
                logger.debug("Skipping malformed temporal: %s — %s", item, e)

        return temporals

    # ── Regex-based fallback extraction ───────────────────────────

    def _extract_with_regex(self, text: str) -> ExtractionResult:
        """Fallback extraction using regex patterns.

        Used when the LLM is unavailable. Identifies entities through
        heuristic patterns:
        - Capitalized words/phrases → potential person/place/organization
        - Known tool names → tool entities
        - Time expressions → temporal references
        - Relationship keywords → relationships
        """
        entities: list[ExtractedEntity] = []
        temporals: list[TemporalReference] = []
        facts: list[ExtractedFact] = []

        # Extract capitalized proper nouns (likely people, places, orgs)
        entities.extend(self._extract_proper_nouns(text))

        # Extract known tools/technologies
        entities.extend(self._extract_tools(text))

        # Extract temporal references
        temporals.extend(self._extract_temporal_regex(text))

        # Extract simple facts/preferences
        facts.extend(self._extract_facts_regex(text))

        # Deduplicate entities by normalized name
        entities = self._deduplicate_entities(entities)

        return ExtractionResult(
            entities=entities,
            facts=facts,
            temporal_references=temporals,
            raw_text=text,
            model_used="regex-fallback",
        )

    def _extract_proper_nouns(self, text: str) -> list[ExtractedEntity]:
        """Extract capitalized proper nouns as potential entities."""
        entities: list[ExtractedEntity] = []

        # Match capitalized words (2+ chars) — uses word boundary instead
        # of lookbehind with ^ (which Python re doesn't support in lookbehinds).
        # This catches "Sarah", "Google", "New York", etc.
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        matches = re.finditer(pattern, text)

        # Skip common sentence-starting words
        skip_words = {
            "I", "The", "This", "That", "It", "We", "They", "He", "She",
            "My", "Our", "Your", "His", "Her", "Its", "Their", "There",
            "Here", "What", "When", "Where", "Why", "How", "Can", "Could",
            "Would", "Should", "Will", "Do", "Does", "Did", "Is", "Are",
            "Was", "Were", "Have", "Has", "Had", "But", "And", "Or", "So",
            "If", "Then", "Also", "Just", "Maybe", "Actually", "Really",
            "Ok", "Okay", "Yeah", "Yes", "No", "Not", "Don", "Let",
            "Need", "Want", "Think", "Know", "Like", "Got", "Going",
            "Today", "Tomorrow", "Yesterday", "Now", "Still", "Already",
        }

        for match in matches:
            name = match.group(1).strip()
            if name in skip_words or len(name) < 2:
                continue

            # Heuristic type classification
            entity_type = self._classify_proper_noun(name, text)

            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    confidence=0.5,  # Lower confidence for regex
                    span=(match.start(1), match.end(1)),
                    context_snippet=text[
                        max(0, match.start() - 30): min(len(text), match.end() + 30)
                    ],
                )
            )

        return entities

    def _classify_proper_noun(self, name: str, context: str) -> EntityType:
        """Heuristic classification of a proper noun based on context."""
        name_lower = name.lower()
        context_lower = context.lower()

        # Person indicators
        person_patterns = [
            f"with {name_lower}",
            f"{name_lower} said",
            f"{name_lower} told",
            f"{name_lower}'s",
            f"meet {name_lower}",
            f"tell {name_lower}",
            f"ask {name_lower}",
            f"call {name_lower}",
            f"{name_lower} is my",
            f"manager {name_lower}",
        ]
        if any(p in context_lower for p in person_patterns):
            return EntityType.PERSON

        # Place indicators
        place_patterns = [
            f"at {name_lower}",
            f"in {name_lower}",
            f"to {name_lower}",
            f"from {name_lower}",
            f"near {name_lower}",
            f"{name_lower} office",
            f"{name_lower} building",
        ]
        if any(p in context_lower for p in place_patterns):
            return EntityType.PLACE

        # Organization indicators (multi-word or known suffixes)
        org_suffixes = {"Inc", "Corp", "LLC", "Ltd", "Co", "HQ"}
        if any(name.endswith(f" {s}") for s in org_suffixes):
            return EntityType.ORGANIZATION

        # Project indicators
        project_patterns = [
            f"{name_lower} project",
            f"project {name_lower}",
            f"working on {name_lower}",
            f"launch {name_lower}",
            f"ship {name_lower}",
            f"{name_lower} deadline",
            f"{name_lower} milestone",
        ]
        if any(p in context_lower for p in project_patterns):
            return EntityType.PROJECT

        # Default to person for single capitalized words, topic for multi-word
        if " " not in name:
            return EntityType.PERSON
        return EntityType.TOPIC

    _KNOWN_TOOLS = {
        "notion": EntityType.TOOL,
        "slack": EntityType.TOOL,
        "figma": EntityType.TOOL,
        "jira": EntityType.TOOL,
        "github": EntityType.TOOL,
        "linear": EntityType.TOOL,
        "google docs": EntityType.TOOL,
        "google sheets": EntityType.TOOL,
        "google calendar": EntityType.TOOL,
        "trello": EntityType.TOOL,
        "asana": EntityType.TOOL,
        "zoom": EntityType.TOOL,
        "teams": EntityType.TOOL,
        "discord": EntityType.TOOL,
        "vscode": EntityType.TOOL,
        "vs code": EntityType.TOOL,
        "cursor": EntityType.TOOL,
        "obsidian": EntityType.TOOL,
        "miro": EntityType.TOOL,
        "confluence": EntityType.TOOL,
        "calendar": EntityType.TOOL,
        "excel": EntityType.TOOL,
        "powerpoint": EntityType.TOOL,
        "keynote": EntityType.TOOL,
    }

    def _extract_tools(self, text: str) -> list[ExtractedEntity]:
        """Extract known tool/technology names from text."""
        entities: list[ExtractedEntity] = []
        text_lower = text.lower()

        for tool_name, entity_type in self._KNOWN_TOOLS.items():
            if tool_name in text_lower:
                # Find the actual-case version in the text
                idx = text_lower.index(tool_name)
                actual_name = text[idx: idx + len(tool_name)]

                entities.append(
                    ExtractedEntity(
                        name=actual_name,
                        entity_type=entity_type,
                        confidence=0.8,
                        span=(idx, idx + len(tool_name)),
                    )
                )

        return entities

    _TEMPORAL_PATTERNS: list[tuple[str, TemporalType]] = [
        # Relative time
        (r"\b(tomorrow|today|yesterday|tonight|this morning|this afternoon|this evening)\b", TemporalType.RELATIVE),
        (r"\b(next|this|last)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", TemporalType.RELATIVE),
        (r"\b(in\s+\d+\s+(?:minutes?|hours?|days?|weeks?|months?))\b", TemporalType.RELATIVE),
        # Absolute time
        (r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b", TemporalType.ABSOLUTE),
        (r"\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b", TemporalType.ABSOLUTE),
        (r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", TemporalType.ABSOLUTE),
        # Recurring
        (r"\b(every\s+(?:day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|evening|night))\b", TemporalType.RECURRING),
        (r"\b(daily|weekly|monthly|biweekly|annually)\b", TemporalType.RECURRING),
        # Duration
        (r"\b(for\s+\d+\s+(?:minutes?|hours?|days?|weeks?|months?))\b", TemporalType.DURATION),
        (r"\b(\d+\s+(?:minutes?|hours?)\s+(?:long|meeting|call|session))\b", TemporalType.DURATION),
        # Vague
        (r"\b(soon|later|sometime|eventually|at some point|when I get a chance)\b", TemporalType.VAGUE),
    ]

    def _extract_temporal_regex(self, text: str) -> list[TemporalReference]:
        """Extract temporal references using regex patterns."""
        temporals: list[TemporalReference] = []
        seen: set[str] = set()

        for pattern, temporal_type in self._TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(1).strip()
                if matched_text.lower() not in seen:
                    seen.add(matched_text.lower())
                    temporals.append(
                        TemporalReference(
                            text=matched_text,
                            temporal_type=temporal_type,
                        )
                    )

        return temporals

    _FACT_PATTERNS: list[tuple[str, FactType]] = [
        (r"(?:I|i)\s+(?:prefer|like|love|enjoy|want)\s+(.+?)(?:\.|$)", FactType.PREFERENCE),
        (r"(?:I|i)\s+(?:always|usually|normally|typically)\s+(.+?)(?:\.|$)", FactType.HABIT),
        (r"(?:I|i)\s+(?:hate|dislike|don't like|can't stand)\s+(.+?)(?:\.|$)", FactType.PREFERENCE),
        (r"(\w+)\s+is\s+my\s+(\w+)", FactType.ATTRIBUTE),
        (r"my\s+(\w+)\s+is\s+(\w+)", FactType.ATTRIBUTE),
    ]

    def _extract_facts_regex(self, text: str) -> list[ExtractedFact]:
        """Extract simple facts/preferences using regex patterns."""
        facts: list[ExtractedFact] = []

        for pattern, fact_type in self._FACT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                content = match.group(0).strip()
                if len(content) > 5:  # Skip trivially short matches
                    facts.append(
                        ExtractedFact(
                            content=content,
                            fact_type=fact_type,
                            confidence=0.5,
                        )
                    )

        return facts

    def _deduplicate_entities(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Remove duplicate entities, keeping the highest confidence version."""
        seen: dict[str, ExtractedEntity] = {}

        for entity in entities:
            key = entity.normalized_name
            if key in seen:
                existing = seen[key]
                if entity.confidence > existing.confidence:
                    seen[key] = entity
            else:
                seen[key] = entity

        return list(seen.values())
