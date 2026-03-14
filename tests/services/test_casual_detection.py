"""Tests for casual observation detection.

Verifies that the casual detection service correctly identifies casual remarks,
substantive observations, and ambiguous inputs. Critically, this module NEVER
filters or drops observations — it only enriches them with metadata.

Covers:
- Empty and near-empty inputs detected as casual
- Single filler words (hmm, yeah, ok, cool) detected as casual
- Casual prefixes (oh well, nice weather, that's funny) detected as casual
- Substantive inputs (tasks, events, questions) detected as substantive
- Mixed signals resolved correctly
- Half-finished thoughts handled gracefully
- Long rambling input treated as substantive (more signal = more data)
- Detection never returns None — every input gets a result
"""

from __future__ import annotations

import pytest

from blurt.services.casual_detection import (
    CasualDetectionResult,
    ObservationType,
    detect_casual,
)


# ---------------------------------------------------------------------------
# Casual: filler words and throwaway remarks
# ---------------------------------------------------------------------------


class TestCasualDetection:
    """Verify filler words, reactions, and throwaway remarks are tagged casual."""

    def test_empty_string_is_casual(self):
        result = detect_casual("")
        assert result.is_casual
        assert result.observation_type == ObservationType.CASUAL
        assert result.confidence >= 0.9
        assert result.word_count == 0

    def test_whitespace_only_is_casual(self):
        result = detect_casual("   ")
        assert result.is_casual
        assert result.word_count == 0

    @pytest.mark.parametrize("word", [
        "hmm", "huh", "yeah", "ok", "okay", "sure", "cool", "nice",
        "wow", "meh", "ugh", "whatever", "right", "yep", "nah",
        "interesting", "funny", "weird",
    ])
    def test_single_filler_words_are_casual(self, word: str):
        result = detect_casual(word)
        assert result.is_casual, f"'{word}' should be detected as casual"
        assert result.observation_type == ObservationType.CASUAL
        assert result.confidence >= 0.9

    @pytest.mark.parametrize("word", ["Hmm", "YEAH", "Ok", "COOL", "Nice"])
    def test_case_insensitive_filler_detection(self, word: str):
        result = detect_casual(word)
        assert result.is_casual, f"'{word}' should be casual regardless of case"

    def test_filler_with_punctuation(self):
        """Filler words with trailing punctuation should still be casual."""
        for word in ["hmm.", "yeah!", "ok?", "cool,"]:
            result = detect_casual(word)
            assert result.is_casual, f"'{word}' should be casual with punctuation"


class TestCasualPrefixes:
    """Verify multi-word casual phrases are detected."""

    @pytest.mark.parametrize("phrase", [
        "huh, interesting",
        "oh well",
        "oh wow, really",
        "nice weather today",
        "that's funny",
        "that's cool I guess",
        "not bad",
        "i guess so",
        "who knows",
    ])
    def test_casual_prefixes_detected(self, phrase: str):
        result = detect_casual(phrase)
        assert result.observation_type in (ObservationType.CASUAL, ObservationType.AMBIGUOUS), \
            f"'{phrase}' should be casual or ambiguous, got {result.observation_type}"

    def test_nice_weather_is_casual(self):
        result = detect_casual("nice weather today")
        assert result.is_casual

    def test_oh_well_is_casual(self):
        result = detect_casual("oh well")
        assert result.is_casual

    def test_that_is_funny_is_casual(self):
        result = detect_casual("that's funny")
        assert result.is_casual


class TestShortUtterances:
    """Short utterances without substantive markers lean casual."""

    @pytest.mark.parametrize("text", [
        "so yeah",
        "pretty much",
        "I see",
        "got it",
    ])
    def test_short_non_substantive_is_casual(self, text: str):
        result = detect_casual(text)
        assert result.observation_type in (ObservationType.CASUAL, ObservationType.AMBIGUOUS), \
            f"'{text}' should be casual or ambiguous"
        assert result.word_count <= 3


# ---------------------------------------------------------------------------
# Substantive: actionable or meaningful content
# ---------------------------------------------------------------------------


class TestSubstantiveDetection:
    """Verify that meaningful content is detected as substantive."""

    @pytest.mark.parametrize("text", [
        "I need to buy groceries",
        "I have to call the dentist tomorrow",
        "Meeting with Sarah at 3pm",
        "Remind me to take my meds at 9pm",
        "Don't forget to water the plants tomorrow",
        "I finished that report",
        "Cancel the dentist appointment",
        "What did I say about that project last week?",
    ])
    def test_actionable_inputs_are_substantive(self, text: str):
        result = detect_casual(text)
        assert result.is_substantive, \
            f"'{text}' should be substantive, got {result.observation_type}"
        assert result.observation_type == ObservationType.SUBSTANTIVE

    def test_task_with_deadline(self):
        result = detect_casual("I need to submit the report by Friday at 5pm")
        assert result.is_substantive
        assert result.confidence >= 0.7

    def test_emotional_reflection(self):
        result = detect_casual("I've been feeling really stressed about work lately")
        assert result.is_substantive

    def test_creative_idea(self):
        result = detect_casual("What if we combined the recommendation engine with mood data?")
        assert result.is_substantive

    def test_question_about_personal_history(self):
        result = detect_casual("When did I last meet with the team about the Q3 plan?")
        assert result.is_substantive


class TestLongUtterances:
    """Long utterances without casual markers are treated as substantive."""

    def test_long_rambling_is_substantive(self):
        text = "So I was walking down the street and I saw this really cool art installation"
        result = detect_casual(text)
        assert result.observation_type in (ObservationType.SUBSTANTIVE, ObservationType.AMBIGUOUS)
        assert result.word_count > 6

    def test_long_reflection_is_substantive(self):
        text = (
            "I've been thinking about my career and I'm not sure if I should "
            "stay in my current role or explore something new"
        )
        result = detect_casual(text)
        assert result.is_substantive


# ---------------------------------------------------------------------------
# Ambiguous: could go either way
# ---------------------------------------------------------------------------


class TestAmbiguousDetection:
    """Inputs that could be either casual or substantive."""

    def test_medium_length_no_markers(self):
        result = detect_casual("that was a thing")
        assert result.observation_type in (
            ObservationType.CASUAL,
            ObservationType.AMBIGUOUS,
        )

    def test_ambiguous_has_moderate_confidence(self):
        result = detect_casual("something something interesting")
        # Should not have extreme confidence in either direction
        assert 0.3 <= result.confidence <= 0.85


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_detection_always_returns_result(self):
        """detect_casual never returns None, always returns a CasualDetectionResult."""
        inputs = ["", "x", "hmm", "normal text", "a" * 1000, "!@#$%"]
        for text in inputs:
            result = detect_casual(text)
            assert isinstance(result, CasualDetectionResult), f"None result for '{text}'"
            assert isinstance(result.observation_type, ObservationType)
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0

    def test_casual_prefix_with_substantive_content(self):
        """Input starting casual but containing substantive content."""
        result = detect_casual("oh right, I need to call the dentist tomorrow")
        # Should be substantive because of the actionable content
        assert result.is_substantive

    def test_signals_are_populated(self):
        """Detection result includes signals explaining the classification."""
        result = detect_casual("hmm")
        assert len(result.signals) > 0

    def test_word_count_accurate(self):
        result = detect_casual("one two three four five")
        assert result.word_count == 5

    def test_frozen_dataclass(self):
        """CasualDetectionResult is immutable."""
        result = detect_casual("test")
        with pytest.raises(AttributeError):
            result.observation_type = ObservationType.CASUAL  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration with observation flow — casual is NEVER filtered
# ---------------------------------------------------------------------------


class TestCasualNeverFiltered:
    """Verify that casual detection is enrichment-only, never filtering."""

    def test_casual_result_does_not_indicate_drop(self):
        """There is no 'drop' or 'filter' signal in CasualDetectionResult."""
        result = detect_casual("hmm")
        # The result has no drop/filter field — by design
        assert not hasattr(result, "should_drop")
        assert not hasattr(result, "should_filter")

    def test_all_observation_types_are_valid(self):
        """Every ObservationType value is a valid type — none means 'skip'."""
        for otype in ObservationType:
            assert otype.value in ("casual", "substantive", "ambiguous")
            # None of these mean "drop" or "ignore"

    def test_casual_has_positive_framing(self):
        """Casual observations are framed positively — they contribute to patterns."""
        result = detect_casual("whatever")
        assert result.is_casual
        # The type name is "casual", not "trivial", "worthless", "noise", etc.
        assert result.observation_type.value == "casual"


# ---------------------------------------------------------------------------
# Bulk classification — varied inputs all get results
# ---------------------------------------------------------------------------


class TestBulkClassification:
    """Test a realistic mix of inputs all get classified."""

    def test_conversation_stream_all_classified(self):
        """Every input in a realistic conversation gets a casual detection result."""
        inputs = [
            "hmm",
            "oh right, I need to call Sarah",
            "nice weather today",
            "meeting with the team at 3pm",
            "whatever",
            "actually make it 4pm",
            "I was thinking about that project",
            "huh",
            "oh well",
            "remind me to buy milk",
            "",
            "that's interesting I guess",
            "feeling pretty good about the presentation",
            "yeah",
            "what time is it?",
        ]

        results = [detect_casual(text) for text in inputs]

        # Every input got a result
        assert len(results) == len(inputs)
        for i, result in enumerate(results):
            assert result is not None, f"Input {i} ('{inputs[i]}') got None result"
            assert isinstance(result.observation_type, ObservationType)

        # At least some casual and some substantive
        casual_count = sum(1 for r in results if r.is_casual)
        substantive_count = sum(1 for r in results if r.is_substantive)
        assert casual_count >= 4, f"Expected at least 4 casual, got {casual_count}"
        assert substantive_count >= 3, f"Expected at least 3 substantive, got {substantive_count}"
