"""Tests for the 7-intent enum/schema and IntentClassificationResult model."""

from __future__ import annotations

import pytest

from blurt.models.intents import (
    INTENT_ACTIONS,
    SYNCABLE_INTENTS,
    BlurtIntent,
    IntentClassificationResult,
    _INTENT_DESCRIPTIONS,
    _INTENT_EXAMPLES,
)


class TestBlurtIntentEnum:
    """Tests for the 7-intent enum definition."""

    def test_exactly_seven_intents(self) -> None:
        """The enum must have exactly 7 members."""
        assert len(BlurtIntent) == 7

    def test_all_intent_values(self) -> None:
        """All 7 intent values are present with correct string values."""
        expected = {"task", "event", "reminder", "idea", "journal", "update", "question"}
        actual = {intent.value for intent in BlurtIntent}
        assert actual == expected

    def test_intent_is_string_enum(self) -> None:
        """BlurtIntent members are usable as strings."""
        assert BlurtIntent.TASK == "task"
        assert str(BlurtIntent.EVENT) == "BlurtIntent.EVENT"
        assert BlurtIntent.JOURNAL.value == "journal"

    def test_intent_from_string(self) -> None:
        """Can create BlurtIntent from string value."""
        for intent in BlurtIntent:
            reconstructed = BlurtIntent(intent.value)
            assert reconstructed is intent

    def test_invalid_intent_raises_value_error(self) -> None:
        """Creating from invalid string raises ValueError."""
        with pytest.raises(ValueError):
            BlurtIntent("unknown")
        with pytest.raises(ValueError):
            BlurtIntent("")

    def test_every_intent_has_description(self) -> None:
        """Every intent must have a non-empty description."""
        for intent in BlurtIntent:
            desc = intent.description
            assert isinstance(desc, str)
            assert len(desc) > 20, f"{intent} description too short: '{desc}'"

    def test_every_intent_has_examples(self) -> None:
        """Every intent must have at least 3 examples."""
        for intent in BlurtIntent:
            examples = intent.examples
            assert isinstance(examples, list)
            assert len(examples) >= 3, f"{intent} has only {len(examples)} examples"
            for ex in examples:
                assert isinstance(ex, str)
                assert len(ex) > 5

    def test_descriptions_dict_covers_all_intents(self) -> None:
        """_INTENT_DESCRIPTIONS must have an entry for every BlurtIntent."""
        for intent in BlurtIntent:
            assert intent in _INTENT_DESCRIPTIONS

    def test_examples_dict_covers_all_intents(self) -> None:
        """_INTENT_EXAMPLES must have an entry for every BlurtIntent."""
        for intent in BlurtIntent:
            assert intent in _INTENT_EXAMPLES


class TestSyncableIntents:
    """Tests for SYNCABLE_INTENTS configuration."""

    def test_syncable_intents_is_frozenset(self) -> None:
        assert isinstance(SYNCABLE_INTENTS, frozenset)

    def test_syncable_intents_are_subset_of_all(self) -> None:
        all_intents = set(BlurtIntent)
        assert SYNCABLE_INTENTS.issubset(all_intents)

    def test_expected_syncable_intents(self) -> None:
        """Task, event, reminder, update should be syncable."""
        assert BlurtIntent.TASK in SYNCABLE_INTENTS
        assert BlurtIntent.EVENT in SYNCABLE_INTENTS
        assert BlurtIntent.REMINDER in SYNCABLE_INTENTS
        assert BlurtIntent.UPDATE in SYNCABLE_INTENTS

    def test_non_syncable_intents(self) -> None:
        """Idea, journal, question should NOT be syncable."""
        assert BlurtIntent.IDEA not in SYNCABLE_INTENTS
        assert BlurtIntent.JOURNAL not in SYNCABLE_INTENTS
        assert BlurtIntent.QUESTION not in SYNCABLE_INTENTS


class TestIntentActions:
    """Tests for INTENT_ACTIONS downstream routing map."""

    def test_all_intents_have_actions(self) -> None:
        for intent in BlurtIntent:
            assert intent in INTENT_ACTIONS
            assert len(INTENT_ACTIONS[intent]) > 0

    def test_syncable_intents_have_downstream_actions(self) -> None:
        """Syncable intents should have sync, update, or schedule actions."""
        for intent in SYNCABLE_INTENTS:
            actions = INTENT_ACTIONS[intent]
            has_downstream = any(
                keyword in a
                for a in actions
                for keyword in ("sync", "update", "schedule")
            )
            assert has_downstream, (
                f"{intent} is syncable but has no sync/update/schedule action: {actions}"
            )


class TestIntentClassificationResult:
    """Tests for the IntentClassificationResult Pydantic model."""

    def test_basic_creation(self) -> None:
        result = IntentClassificationResult(
            intent=BlurtIntent.TASK,
            confidence=0.92,
            raw_text="I need to buy groceries",
        )
        assert result.intent == BlurtIntent.TASK
        assert result.confidence == 0.92
        assert result.raw_text == "I need to buy groceries"
        assert result.secondary_intent is None
        assert result.secondary_confidence == 0.0
        assert result.reasoning == ""

    def test_high_confidence_property(self) -> None:
        high = IntentClassificationResult(
            intent=BlurtIntent.TASK, confidence=0.90, raw_text="test"
        )
        assert high.is_high_confidence is True

        low = IntentClassificationResult(
            intent=BlurtIntent.TASK, confidence=0.80, raw_text="test"
        )
        assert low.is_high_confidence is False

        boundary = IntentClassificationResult(
            intent=BlurtIntent.TASK, confidence=0.85, raw_text="test"
        )
        assert boundary.is_high_confidence is True

    def test_ambiguous_property(self) -> None:
        # No secondary → not ambiguous
        no_secondary = IntentClassificationResult(
            intent=BlurtIntent.TASK, confidence=0.90, raw_text="test"
        )
        assert no_secondary.is_ambiguous is False

        # Large gap → not ambiguous
        clear = IntentClassificationResult(
            intent=BlurtIntent.TASK,
            confidence=0.90,
            secondary_intent=BlurtIntent.EVENT,
            secondary_confidence=0.05,
            raw_text="test",
        )
        assert clear.is_ambiguous is False

        # Small gap → ambiguous
        ambiguous = IntentClassificationResult(
            intent=BlurtIntent.TASK,
            confidence=0.45,
            secondary_intent=BlurtIntent.EVENT,
            secondary_confidence=0.40,
            raw_text="test",
        )
        assert ambiguous.is_ambiguous is True

    def test_confidence_validation(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            IntentClassificationResult(
                intent=BlurtIntent.TASK, confidence=1.5, raw_text="test"
            )
        with pytest.raises(Exception):
            IntentClassificationResult(
                intent=BlurtIntent.TASK, confidence=-0.1, raw_text="test"
            )

    def test_secondary_confidence_validation(self) -> None:
        """Secondary confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            IntentClassificationResult(
                intent=BlurtIntent.TASK,
                confidence=0.5,
                secondary_intent=BlurtIntent.EVENT,
                secondary_confidence=1.5,
                raw_text="test",
            )

    def test_with_reasoning(self) -> None:
        result = IntentClassificationResult(
            intent=BlurtIntent.JOURNAL,
            confidence=0.88,
            raw_text="I felt really good today",
            reasoning="Emotional expression about personal experience",
        )
        assert "emotional" in result.reasoning.lower()

    def test_serialization_roundtrip(self) -> None:
        """Model should serialize to dict and back."""
        result = IntentClassificationResult(
            intent=BlurtIntent.EVENT,
            confidence=0.91,
            secondary_intent=BlurtIntent.TASK,
            secondary_confidence=0.05,
            raw_text="Meeting at 3pm",
            reasoning="Time-bound occurrence",
        )
        data = result.model_dump()
        restored = IntentClassificationResult.model_validate(data)
        assert restored.intent == result.intent
        assert restored.confidence == result.confidence
        assert restored.secondary_intent == result.secondary_intent
        assert restored.raw_text == result.raw_text

    def test_json_serialization(self) -> None:
        """Model should serialize to JSON string."""
        result = IntentClassificationResult(
            intent=BlurtIntent.IDEA,
            confidence=0.87,
            raw_text="What if we used AI for cooking?",
        )
        json_str = result.model_dump_json()
        assert '"idea"' in json_str
        assert "0.87" in json_str
