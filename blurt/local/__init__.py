"""Local-only adapters for offline Blurt operation.

Provides feature-parity implementations of knowledge graph storage,
intent classification, and entity extraction that function entirely
offline without any external API calls.

Usage::

    from blurt.local import (
        LocalKnowledgeGraphStore,
        LocalIntentClassifier,
        LocalEntityExtractor,
    )
"""

from blurt.local.classifier import LocalIntentClassifier
from blurt.local.extractor import LocalEntityExtractor
from blurt.local.storage import LocalKnowledgeGraphStore

__all__ = [
    "LocalEntityExtractor",
    "LocalIntentClassifier",
    "LocalKnowledgeGraphStore",
]
