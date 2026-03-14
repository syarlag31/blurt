"""Intent classifier using Gemini Flash-Lite.

Classifies user input into one of 7 intent types using structured
JSON output from the Gemini API. Uses the FAST tier (Flash-Lite)
for low-latency classification.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from blurt.clients.gemini import GeminiClient, GeminiError, ModelTier
from blurt.classification.models import IntentScore
from blurt.models.intents import BlurtIntent

logger = logging.getLogger(__name__)

# System instruction for intent classification
_CLASSIFICATION_SYSTEM_PROMPT = """\
You are a silent intent classifier for a personal AI assistant called Blurt.
Your job is to classify user input into exactly one primary intent and provide
confidence scores for all 7 intent types.

The 7 intent types are:
- task: An actionable item the user wants to do or needs to complete. May have \
an implicit or explicit deadline. Examples: "I need to buy groceries", \
"finish the report by Friday", "call the dentist to schedule an appointment", \
"submit the expense report", "pick up dry cleaning after work".
- event: A calendar-bound occurrence with a specific time, date, or place. \
Examples: "dinner with Sarah at 7pm on Saturday", "team standup tomorrow at 9am", \
"flight to NYC on March 20th at 6am", "doctor's appointment next Tuesday at 2pm".
- reminder: A time-triggered nudge — lighter than a task, just needs a ping. \
Often starts with "remind me". Examples: "remind me to take my meds at 9pm", \
"don't forget to water the plants tomorrow", "ping me about the proposal in two hours".
- idea: A creative thought, hypothesis, brainstorm, or conceptual note. Not immediately \
actionable. Examples: "what if we combined X with Y", "I think the market is shifting \
toward subscription models", "random thought — maybe we should try a podcast format".
- journal: Personal reflection, emotional expression, or narrative about feelings \
and experiences. Examples: "today was really tough", "I'm feeling grateful for \
the support from my team", "had an amazing conversation with my mentor today".
- update: A status change on something already tracked. Modifies existing knowledge. \
Examples: "actually the meeting moved to 3pm", "I finished that report", \
"the project deadline got extended", "cancel the dentist appointment".
- question: A query seeking information from personal knowledge or general info. \
Examples: "what did I say about that project last week?", "when is Sarah's birthday?", \
"how many tasks do I have this week?", "did I ever finish that book?".

Rules:
1. Every input MUST be classified — there is no "unknown" or "other" category
2. Return confidence scores for ALL 7 intents (must sum to approximately 1.0)
3. Be generous with confidence when the intent is clear (>0.85 for obvious intents)
4. For ambiguous inputs, distribute scores more evenly
5. Context clues: time references with actions suggest event/reminder, action verbs suggest task
6. Personal reflections and feelings default to journal
7. "remind me" phrases are almost always reminder, not task
8. Questions ending with "?" are almost always question, unless "what if" (idea)
9. "cancel", "actually", "scratch that", "finished", "done with" suggest update
10. Do NOT ask the user for clarification — just classify silently

Respond ONLY with valid JSON in this exact format:
{
  "primary_intent": "<intent_type>",
  "confidence_scores": {
    "task": 0.0,
    "event": 0.0,
    "reminder": 0.0,
    "idea": 0.0,
    "journal": 0.0,
    "update": 0.0,
    "question": 0.0
  }
}"""

# System instruction for ambiguity resolution (uses the SMART model)
_RESOLUTION_SYSTEM_PROMPT = """\
You are resolving an ambiguous intent classification for a personal AI assistant.
The fast classifier was unsure about the user's intent. Analyze the input more carefully.

Consider:
- What is the user's PRIMARY goal with this statement?
- Are there multiple intents embedded in one statement?
- What action would best serve the user?

If the input contains multiple distinct intents, set "multi_intent" to true and list them.

The 7 intent types are: task, event, reminder, idea, journal, update, question.

Respond ONLY with valid JSON:
{
  "primary_intent": "<intent_type>",
  "confidence": 0.0,
  "multi_intent": false,
  "intents": [
    {"intent": "<type>", "confidence": 0.0, "segment": "relevant text portion"}
  ],
  "reasoning": "brief explanation"
}"""


class IntentClassifier:
    """Classifies user input into Blurt intent types using Gemini.

    Uses the two-model strategy:
    - Flash-Lite (FAST tier) for initial classification
    - Flash (SMART tier) for resolving ambiguous cases

    Usage::

        classifier = IntentClassifier(gemini_client)
        scores = await classifier.classify("I need to buy groceries")
        # => [IntentScore(intent=TASK, confidence=0.92), ...]
    """

    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    async def classify(self, text: str) -> list[IntentScore]:
        """Classify text input and return scored intents.

        Args:
            text: The user's input text to classify.

        Returns:
            List of IntentScore sorted by confidence (highest first).

        Raises:
            ClassificationError: If classification fails after retries.
        """
        try:
            response = await self._client.generate(
                prompt=text,
                tier=ModelTier.FAST,
                system_instruction=_CLASSIFICATION_SYSTEM_PROMPT,
                temperature=0.05,  # Very low temp for consistent classification
                max_output_tokens=256,
                response_mime_type="application/json",
            )
            return self._parse_classification_response(response.text)
        except GeminiError:
            raise
        except Exception as e:
            raise ClassificationError(f"Classification failed: {e}") from e

    async def resolve_ambiguity(self, text: str) -> dict[str, Any]:
        """Use the smarter model to resolve an ambiguous classification.

        Args:
            text: The user's input text to re-analyze.

        Returns:
            Dict with resolved classification details.

        Raises:
            ClassificationError: If resolution fails.
        """
        try:
            response = await self._client.generate(
                prompt=text,
                tier=ModelTier.SMART,
                system_instruction=_RESOLUTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=512,
                response_mime_type="application/json",
            )
            return self._parse_resolution_response(response.text)
        except GeminiError:
            raise
        except Exception as e:
            raise ClassificationError(f"Ambiguity resolution failed: {e}") from e

    def _parse_classification_response(self, raw_text: str) -> list[IntentScore]:
        """Parse the classifier's JSON response into IntentScores."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ClassificationError(
                f"Invalid JSON from classifier: {raw_text[:200]}"
            ) from e

        scores_dict = data.get("confidence_scores", {})
        if not scores_dict:
            # Fallback: try to extract from primary_intent alone
            primary = data.get("primary_intent", "journal")
            return self._build_single_intent_scores(primary)

        scores: list[IntentScore] = []
        for intent_name, confidence in scores_dict.items():
            try:
                intent = BlurtIntent(intent_name.lower())
                scores.append(IntentScore(intent=intent, confidence=float(confidence)))
            except (ValueError, TypeError):
                logger.warning("Unknown intent in response: %s", intent_name)
                continue

        # Ensure all 7 intents are present
        seen = {s.intent for s in scores}
        for intent in BlurtIntent:
            if intent not in seen:
                scores.append(IntentScore(intent=intent, confidence=0.0))

        # Normalize scores to sum to ~1.0
        scores = self._normalize_scores(scores)

        # Sort by confidence descending
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores

    def _parse_resolution_response(self, raw_text: str) -> dict[str, Any]:
        """Parse the resolution model's JSON response."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ClassificationError(
                f"Invalid JSON from resolution model: {raw_text[:200]}"
            ) from e

        primary = data.get("primary_intent", "journal")
        try:
            intent = BlurtIntent(primary.lower())
        except ValueError:
            intent = BlurtIntent.JOURNAL

        return {
            "primary_intent": intent,
            "confidence": float(data.get("confidence", 0.5)),
            "multi_intent": bool(data.get("multi_intent", False)),
            "intents": data.get("intents", []),
            "reasoning": data.get("reasoning", ""),
        }

    @staticmethod
    def _build_single_intent_scores(primary_name: str) -> list[IntentScore]:
        """Build a full scores list from just a primary intent name."""
        try:
            primary = BlurtIntent(primary_name.lower())
        except ValueError:
            primary = BlurtIntent.JOURNAL

        scores = []
        for intent in BlurtIntent:
            conf = 0.85 if intent == primary else (0.15 / 6)
            scores.append(IntentScore(intent=intent, confidence=conf))

        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores

    @staticmethod
    def _normalize_scores(scores: list[IntentScore]) -> list[IntentScore]:
        """Normalize scores so they sum to 1.0."""
        total = sum(s.confidence for s in scores)
        if total <= 0:
            # Equal distribution fallback
            equal = 1.0 / len(scores) if scores else 0.0
            for s in scores:
                s.confidence = equal
        elif abs(total - 1.0) > 0.01:
            for s in scores:
                s.confidence /= total
        return scores


class ClassificationError(Exception):
    """Error during intent classification."""

    pass
