"""Blurt classification pipeline.

Silent intent classification for every user input using the two-model
strategy (Flash-Lite for fast classification, Flash for ambiguity resolution).
"""

from blurt.classification.adapter import ClassificationAdapter, create_classification_adapter
from blurt.classification.models import (
    AmbiguityResolution,
    ClassificationResult,
    ClassificationStatus,
    FallbackStrategy,
)
from blurt.classification.classifier import IntentClassifier
from blurt.classification.pipeline import ClassificationPipeline

__all__ = [
    "AmbiguityResolution",
    "ClassificationAdapter",
    "ClassificationPipeline",
    "ClassificationResult",
    "ClassificationStatus",
    "FallbackStrategy",
    "IntentClassifier",
    "create_classification_adapter",
]
