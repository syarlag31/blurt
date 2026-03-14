"""Local rule-based entity extractor for offline operation.

Extracts entities, relationships, facts, and temporal references from
natural speech text using regex patterns, keyword dictionaries, and
heuristic rules — entirely offline with no API calls.

Wraps and extends the regex fallback logic from EntityExtractor to
provide a standalone local extraction service with full feature parity.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from blurt.extraction.entities import (
    EntityExtractor as CoreExtractor,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionResult,
    TemporalReference,
    TemporalType,
)
from blurt.models.entities import EntityType, FactType, RelationshipType

logger = logging.getLogger(__name__)


# Additional relationship patterns not in the core extractor
_RELATIONSHIP_PATTERNS: list[tuple[str, RelationshipType]] = [
    (r"(\w+(?:\s+\w+)?)\s+is\s+my\s+(?:manager|boss|supervisor)", RelationshipType.MANAGED_BY),
    (r"(\w+(?:\s+\w+)?)\s+(?:manages|leads|supervises)\s+(?:me|us|the team)", RelationshipType.MANAGES),
    (r"(?:I|i)\s+(?:work|working)\s+with\s+(\w+(?:\s+\w+)?)", RelationshipType.WORKS_WITH),
    (r"(\w+(?:\s+\w+)?)\s+is\s+(?:on|part of)\s+(?:my|the|our)\s+team", RelationshipType.WORKS_WITH),
    (r"(\w+(?:\s+\w+)?)\s+(?:works|is)\s+at\s+(\w+(?:\s+\w+)?)", RelationshipType.EMPLOYED_BY),
    (r"(?:I|i)\s+(?:know|met)\s+(\w+(?:\s+\w+)?)", RelationshipType.KNOWS),
    (r"(\w+(?:\s+\w+)?)\s+is\s+(?:a\s+)?(?:friend|buddy|pal)", RelationshipType.KNOWS),
    (r"(\w+(?:\s+\w+)?)\s+is\s+(?:my\s+)?(?:brother|sister|mom|dad|parent|sibling|spouse|partner|wife|husband)", RelationshipType.KNOWS),
    (r"(?:working|work)\s+on\s+(\w+(?:\s+\w+)?)\s+with\s+(\w+(?:\s+\w+)?)", RelationshipType.COLLABORATES_ON),
]

# Organization indicators (beyond what core extractor has)
_ORG_KEYWORDS = {
    "team", "company", "corp", "corporation", "inc", "llc", "ltd",
    "department", "division", "group", "studio", "lab", "labs",
    "foundation", "institute", "university", "school", "college",
    "agency", "firm", "consultancy", "startup",
}

# Project indicators
_PROJECT_KEYWORDS = {
    "project", "initiative", "sprint", "release", "version",
    "migration", "redesign", "refactor", "launch", "rollout",
    "campaign", "proposal", "prototype", "mvp", "poc",
    "deck", "plan", "roadmap", "pipeline",
}


class LocalEntityExtractor:
    """Rule-based entity extractor for fully offline operation.

    Extracts entities, relationships, facts, and temporal references
    using regex patterns and heuristic rules. No external API calls.

    Builds on the regex fallback from CoreExtractor and adds:
    - Relationship extraction from contextual patterns
    - Enhanced entity type classification
    - More comprehensive temporal reference detection
    - Richer fact/preference extraction

    Usage::

        extractor = LocalEntityExtractor()
        result = await extractor.extract(
            "Sarah and I are meeting at Notion HQ tomorrow at 3pm"
        )
        for entity in result.entities:
            print(f"{entity.name} ({entity.entity_type.value})")
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.3,
        max_entities_per_extraction: int = 50,
    ) -> None:
        self._min_confidence = min_confidence
        self._max_entities = max_entities_per_extraction
        # Core extractor in fallback mode (no Gemini client)
        self._core = CoreExtractor(
            gemini_client=None,
            min_confidence=min_confidence,
            max_entities_per_extraction=max_entities_per_extraction,
        )
        # Precompile relationship patterns
        self._relationship_patterns = [
            (re.compile(p, re.IGNORECASE), rt)
            for p, rt in _RELATIONSHIP_PATTERNS
        ]
        self._extraction_count = 0

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and structured data from text.

        Uses the core extractor's regex fallback plus additional
        relationship and enhanced entity extraction.

        Args:
            text: Natural speech text to analyze.

        Returns:
            ExtractionResult with entities, relationships, facts,
            and temporal references.
        """
        if not text or not text.strip():
            return ExtractionResult(raw_text=text or "")

        text = text.strip()
        self._extraction_count += 1
        start = time.monotonic()

        # Use core extractor's regex fallback for base extraction
        base_result = self._core._extract_with_regex(text)

        # Enhanced entity classification
        enhanced_entities = self._enhance_entity_types(base_result.entities, text)

        # Extract relationships
        relationships = self._extract_relationships(text, enhanced_entities)

        # Enhanced fact extraction
        facts = list(base_result.facts)
        facts.extend(self._extract_enhanced_facts(text))

        # Enhanced temporal extraction
        temporals = list(base_result.temporal_references)
        temporals.extend(self._extract_enhanced_temporals(text))

        # Deduplicate
        enhanced_entities = self._deduplicate_entities(enhanced_entities)
        facts = self._deduplicate_facts(facts)
        temporals = self._deduplicate_temporals(temporals)

        elapsed = (time.monotonic() - start) * 1000

        return ExtractionResult(
            entities=enhanced_entities[:self._max_entities],
            relationships=relationships,
            facts=facts,
            temporal_references=temporals,
            raw_text=text,
            model_used="local-rules",
            latency_ms=elapsed,
        )

    async def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract entities from multiple texts."""
        return [await self.extract(t) for t in texts]

    def _enhance_entity_types(
        self, entities: list[ExtractedEntity], text: str
    ) -> list[ExtractedEntity]:
        """Enhance entity type classification with additional heuristics."""
        text_lower = text.lower()
        enhanced = []

        for entity in entities:
            new_type = self._reclassify_entity(entity, text_lower)
            if new_type != entity.entity_type:
                entity = ExtractedEntity(
                    name=entity.name,
                    entity_type=new_type,
                    normalized_name=entity.normalized_name,
                    aliases=entity.aliases,
                    attributes=entity.attributes,
                    confidence=entity.confidence,
                    span=entity.span,
                    context_snippet=entity.context_snippet,
                )
            enhanced.append(entity)

        return enhanced

    def _reclassify_entity(
        self, entity: ExtractedEntity, text_lower: str
    ) -> EntityType:
        """Reclassify an entity type using enhanced heuristics."""
        name_lower = entity.normalized_name

        # Check for organization keywords in context
        for kw in _ORG_KEYWORDS:
            if f"{name_lower} {kw}" in text_lower or f"{kw} {name_lower}" in text_lower:
                return EntityType.ORGANIZATION

        # Check for project keywords in context
        for kw in _PROJECT_KEYWORDS:
            if f"{name_lower} {kw}" in text_lower or f"{kw} {name_lower}" in text_lower:
                return EntityType.PROJECT

        # "at <Entity>" often means place
        if f"at {name_lower}" in text_lower:
            # But not "at 3pm" or "at home" → check it's not time-like
            if not re.search(rf"at\s+{re.escape(name_lower)}\s*(?:am|pm|\d)", text_lower):
                return EntityType.PLACE

        return entity.entity_type

    def _extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        """Extract relationships from text using pattern matching."""
        relationships: list[ExtractedRelationship] = []

        for pattern, rel_type in self._relationship_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0].strip()
                    target = groups[1].strip()
                elif len(groups) == 1:
                    source = "speaker"
                    target = groups[0].strip()
                else:
                    continue

                relationships.append(
                    ExtractedRelationship(
                        source_name=source,
                        target_name=target,
                        relationship_type=rel_type,
                        context_snippet=text[
                            max(0, match.start() - 20): min(len(text), match.end() + 20)
                        ],
                        confidence=0.6,
                    )
                )

        # Create MENTIONED_WITH relationships for co-mentioned entities
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1.normalized_name != e2.normalized_name:
                    relationships.append(
                        ExtractedRelationship(
                            source_name=e1.name,
                            target_name=e2.name,
                            relationship_type=RelationshipType.MENTIONED_WITH,
                            context_snippet=text[:200],
                            confidence=0.5,
                        )
                    )

        return relationships

    def _extract_enhanced_facts(self, text: str) -> list[ExtractedFact]:
        """Extract additional facts beyond the core extractor's patterns."""
        facts: list[ExtractedFact] = []

        # "X is my Y" patterns for attribute facts
        attr_pattern = re.compile(
            r"(\w+(?:\s+\w+)?)\s+is\s+my\s+(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        for match in attr_pattern.finditer(text):
            subject = match.group(1).strip()
            attribute = match.group(2).strip()
            # Skip common false positives
            if subject.lower() in {"this", "that", "it", "here", "there", "what"}:
                continue
            facts.append(
                ExtractedFact(
                    content=f"{subject} is my {attribute}",
                    fact_type=FactType.ATTRIBUTE,
                    subject_entity_name=subject,
                    confidence=0.7,
                )
            )

        # "My X is Y" patterns
        my_pattern = re.compile(
            r"my\s+(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        for match in my_pattern.finditer(text):
            attr = match.group(1).strip()
            value = match.group(2).strip()
            facts.append(
                ExtractedFact(
                    content=f"My {attr} is {value}",
                    fact_type=FactType.ATTRIBUTE,
                    confidence=0.6,
                )
            )

        # Alias patterns: "X means Y", "X is also called Y"
        alias_patterns = [
            re.compile(
                r"(\w+(?:\s+\w+)?)\s+(?:means|refers to|is short for)\s+(.+?)(?:\.|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\w+(?:\s+\w+)?)\s+(?:is also called|is also known as|aka)\s+(.+?)(?:\.|$)",
                re.IGNORECASE,
            ),
        ]
        for pattern in alias_patterns:
            for match in pattern.finditer(text):
                alias = match.group(1).strip()
                canonical = match.group(2).strip()
                if len(alias) > 1 and len(canonical) > 1:
                    facts.append(
                        ExtractedFact(
                            content=f'"{alias}" refers to "{canonical}"',
                            fact_type=FactType.ALIAS,
                            confidence=0.7,
                        )
                    )

        # Association patterns: "X is related to Y", "X is connected to Y"
        assoc_pattern = re.compile(
            r"(\w+(?:\s+\w+)?)\s+is\s+(?:related|connected|linked|tied)\s+to\s+(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        for match in assoc_pattern.finditer(text):
            a = match.group(1).strip()
            b = match.group(2).strip()
            facts.append(
                ExtractedFact(
                    content=f"{a} is associated with {b}",
                    fact_type=FactType.ASSOCIATION,
                    subject_entity_name=a,
                    confidence=0.6,
                )
            )

        return facts

    def _extract_enhanced_temporals(self, text: str) -> list[TemporalReference]:
        """Extract additional temporal references."""
        temporals: list[TemporalReference] = []

        # "by <date>" patterns (deadlines)
        deadline_pattern = re.compile(
            r"\bby\s+((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
            r"|tomorrow|tonight|end\s+of\s+(?:day|week|month|year)"
            r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?))",
            re.IGNORECASE,
        )
        for match in deadline_pattern.finditer(text):
            temporals.append(
                TemporalReference(
                    text=match.group(1).strip(),
                    temporal_type=TemporalType.RELATIVE,
                    is_deadline=True,
                )
            )

        # "before <time>" patterns
        before_pattern = re.compile(
            r"\bbefore\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
            re.IGNORECASE,
        )
        for match in before_pattern.finditer(text):
            temporals.append(
                TemporalReference(
                    text=f"before {match.group(1).strip()}",
                    temporal_type=TemporalType.ABSOLUTE,
                    is_deadline=True,
                )
            )

        # "after <time/event>" patterns
        after_pattern = re.compile(
            r"\bafter\s+((?:the\s+)?(?:meeting|lunch|dinner|work|class|school))\b",
            re.IGNORECASE,
        )
        for match in after_pattern.finditer(text):
            temporals.append(
                TemporalReference(
                    text=f"after {match.group(1).strip()}",
                    temporal_type=TemporalType.RELATIVE,
                )
            )

        return temporals

    def _deduplicate_entities(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen: dict[str, ExtractedEntity] = {}
        for entity in entities:
            key = entity.normalized_name
            if key in seen:
                if entity.confidence > seen[key].confidence:
                    seen[key] = entity
            else:
                seen[key] = entity
        return list(seen.values())

    def _deduplicate_facts(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        """Remove duplicate facts by content."""
        seen: dict[str, ExtractedFact] = {}
        for fact in facts:
            key = fact.content.lower().strip()
            if key not in seen:
                seen[key] = fact
        return list(seen.values())

    def _deduplicate_temporals(
        self, temporals: list[TemporalReference]
    ) -> list[TemporalReference]:
        """Remove duplicate temporal references."""
        seen: set[str] = set()
        result = []
        for t in temporals:
            key = t.text.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Extraction statistics."""
        return {
            "extraction_count": self._extraction_count,
            "mode": "local-rules",
        }
