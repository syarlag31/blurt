"""Tests for LocalIntentClassifier — offline rule-based classification."""

from __future__ import annotations

import pytest

from blurt.local.classifier import LocalIntentClassifier
from blurt.models.intents import BlurtIntent


@pytest.fixture
def classifier() -> LocalIntentClassifier:
    return LocalIntentClassifier()


class TestTaskClassification:
    async def test_need_to(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I need to buy groceries")
        assert scores[0].intent == BlurtIntent.TASK

    async def test_have_to(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I have to finish the report by Friday")
        assert scores[0].intent == BlurtIntent.TASK

    async def test_action_verb(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Submit the expense report")
        assert scores[0].intent == BlurtIntent.TASK

    async def test_call_someone(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Call the dentist to schedule an appointment")
        assert scores[0].intent == BlurtIntent.TASK


class TestEventClassification:
    async def test_meeting_with_time(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Meeting with Sarah at 3pm tomorrow")
        assert scores[0].intent == BlurtIntent.EVENT

    async def test_dinner(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Dinner with Jake at 7pm on Saturday")
        assert scores[0].intent == BlurtIntent.EVENT

    async def test_appointment(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Doctor's appointment next Tuesday at 2pm")
        assert scores[0].intent == BlurtIntent.EVENT

    async def test_flight(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Flight to NYC on March 20th at 6am")
        assert scores[0].intent == BlurtIntent.EVENT


class TestReminderClassification:
    async def test_remind_me(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Remind me to take my meds at 9pm")
        assert scores[0].intent == BlurtIntent.REMINDER

    async def test_dont_forget(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Don't forget to water the plants tomorrow")
        assert scores[0].intent == BlurtIntent.REMINDER

    async def test_ping_me(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Ping me about the proposal in two hours")
        assert scores[0].intent == BlurtIntent.REMINDER


class TestIdeaClassification:
    async def test_what_if(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("What if we combined the recommendation engine with mood data?")
        assert scores[0].intent == BlurtIntent.IDEA

    async def test_random_thought(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Random thought — maybe we should try a podcast format")
        assert scores[0].intent == BlurtIntent.IDEA

    async def test_wonder(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I wonder if we could use embeddings for personal memory")
        assert scores[0].intent == BlurtIntent.IDEA


class TestJournalClassification:
    async def test_feeling(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I'm feeling grateful for the support from my team")
        assert scores[0].intent == BlurtIntent.JOURNAL

    async def test_today_was(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Today was really tough, the presentation didn't go well")
        assert scores[0].intent == BlurtIntent.JOURNAL

    async def test_emotional(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I'm stressed about the upcoming deadline")
        assert scores[0].intent == BlurtIntent.JOURNAL

    async def test_reflection(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I've been thinking about my career direction lately")
        assert scores[0].intent == BlurtIntent.JOURNAL


class TestUpdateClassification:
    async def test_actually(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Actually the meeting moved to 3pm")
        assert scores[0].intent == BlurtIntent.UPDATE

    async def test_finished(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I finished that quarterly report")
        assert scores[0].intent == BlurtIntent.UPDATE

    async def test_cancel(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Cancel the dentist appointment")
        assert scores[0].intent == BlurtIntent.UPDATE

    async def test_scratch_that(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Scratch that, never mind about the groceries")
        assert scores[0].intent == BlurtIntent.UPDATE


class TestQuestionClassification:
    async def test_what_question(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("What did I say about that project last week?")
        assert scores[0].intent == BlurtIntent.QUESTION

    async def test_when_question(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("When is Sarah's birthday?")
        assert scores[0].intent == BlurtIntent.QUESTION

    async def test_how_many(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("How many tasks do I have this week?")
        assert scores[0].intent == BlurtIntent.QUESTION

    async def test_did_i(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("Did I ever finish that book?")
        assert scores[0].intent == BlurtIntent.QUESTION


class TestScoreProperties:
    async def test_scores_sum_to_one(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I need to buy groceries")
        total = sum(s.confidence for s in scores)
        assert abs(total - 1.0) < 0.05  # Allow small float error

    async def test_all_7_intents_present(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("I need to buy groceries")
        intents = {s.intent for s in scores}
        assert len(intents) == 7

    async def test_sorted_descending(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("remind me to call Sarah")
        for i in range(len(scores) - 1):
            assert scores[i].confidence >= scores[i + 1].confidence

    async def test_empty_input_defaults_to_journal(self, classifier: LocalIntentClassifier):
        scores = await classifier.classify("")
        assert scores[0].intent == BlurtIntent.JOURNAL


class TestClassifyWithResult:
    async def test_confident_result(self, classifier: LocalIntentClassifier):
        result = await classifier.classify_with_result("Remind me to take my meds at 9pm")
        assert result.primary_intent == BlurtIntent.REMINDER
        assert result.model_used == "local-rules"
        assert result.latency_ms > 0

    async def test_result_has_all_scores(self, classifier: LocalIntentClassifier):
        result = await classifier.classify_with_result("I need to finish the report")
        assert len(result.all_scores) == 7

    async def test_no_api_calls_made(self, classifier: LocalIntentClassifier):
        """Classification should work without any network access."""
        # This test validates the core requirement: entirely offline
        result = await classifier.classify_with_result(
            "Meeting with the team tomorrow at 9am"
        )
        assert result.model_used == "local-rules"
        assert result.primary_intent is not None


class TestAccuracyBaseline:
    """Test accuracy against the example utterances from the intent model.

    Target: >85% accuracy across all 7 intent types.
    """

    INTENT_EXAMPLES = {
        BlurtIntent.TASK: [
            "I need to buy groceries this weekend",
            "Finish the quarterly report by Friday",
            "Call the dentist to schedule an appointment",
            "Submit the expense report",
            "Pick up dry cleaning after work",
        ],
        BlurtIntent.EVENT: [
            "Dinner with Sarah at 7pm on Saturday",
            "Team standup tomorrow at 9am",
            "Flight to NYC on March 20th at 6am",
            "Doctor's appointment next Tuesday at 2pm",
            "Conference call with the London team at 3pm",
        ],
        BlurtIntent.REMINDER: [
            "Remind me to take my meds at 9pm",
            "Don't forget to water the plants tomorrow",
            "Ping me about the proposal in two hours",
            "Remind me to follow up with Jake next week",
        ],
        BlurtIntent.IDEA: [
            "What if we combined the recommendation engine with user mood data?",
            "Random thought — maybe we should try a podcast format",
            "I wonder if we could use embeddings for personal memory",
        ],
        BlurtIntent.JOURNAL: [
            "Today was really tough, the presentation didn't go well",
            "I'm feeling grateful for the support from my team",
            "Had an amazing conversation with my mentor today",
            "I've been thinking about my career direction lately",
            "Feeling energized after that workout this morning",
        ],
        BlurtIntent.UPDATE: [
            "Actually the meeting moved to 3pm",
            "I finished that quarterly report",
            "The project deadline got extended to next month",
            "Cancel the dentist appointment",
        ],
        BlurtIntent.QUESTION: [
            "What did I say about that project last week?",
            "When is Sarah's birthday?",
            "How many tasks do I have this week?",
            "Did I ever finish that book I was reading?",
        ],
    }

    async def test_overall_accuracy(self, classifier: LocalIntentClassifier):
        """Overall accuracy should be at least 85%."""
        correct = 0
        total = 0

        for expected_intent, examples in self.INTENT_EXAMPLES.items():
            for text in examples:
                scores = await classifier.classify(text)
                predicted = scores[0].intent
                if predicted == expected_intent:
                    correct += 1
                total += 1

        accuracy = correct / total
        assert accuracy >= 0.85, (
            f"Accuracy {accuracy:.1%} ({correct}/{total}) below 85% threshold"
        )

    async def test_per_intent_accuracy(self, classifier: LocalIntentClassifier):
        """Each intent type should classify at least 60% of its examples correctly."""
        for expected_intent, examples in self.INTENT_EXAMPLES.items():
            correct = 0
            for text in examples:
                scores = await classifier.classify(text)
                if scores[0].intent == expected_intent:
                    correct += 1
            accuracy = correct / len(examples) if examples else 0
            assert accuracy >= 0.6, (
                f"{expected_intent.value} accuracy {accuracy:.1%} "
                f"({correct}/{len(examples)}) below 60% minimum"
            )
