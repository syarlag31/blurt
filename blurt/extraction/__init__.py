"""Entity extraction services for Blurt.

Processes natural speech text to identify entities, relationships,
and facts using the two-model Gemini strategy (Flash-Lite for extraction).

Public API:
    EntityExtractor          – Cloud-based extraction via Gemini Flash-Lite
    ExtractionResult         – Unified extraction output
    ExtractedEntity          – A single extracted entity
    ExtractedRelationship    – A relationship between entities
    ExtractedFact            – A fact / preference / habit
    TemporalReference        – A time expression
    TemporalType             – Enum of temporal reference types
    EntityExtractionError    – Raised when extraction fails
"""

from blurt.extraction.entities import (
    EntityExtractionError,
    EntityExtractor,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionResult,
    TemporalReference,
    TemporalType,
)

__all__ = [
    "EntityExtractionError",
    "EntityExtractor",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractedRelationship",
    "ExtractionResult",
    "TemporalReference",
    "TemporalType",
]
