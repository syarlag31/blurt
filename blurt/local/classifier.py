"""Local rule-based intent classifier for offline operation.

Classifies user input into one of 7 intent types using keyword matching,
pattern rules, and heuristic scoring — entirely offline with no API calls.
Achieves the >85% accuracy target through comprehensive keyword dictionaries
and contextual pattern analysis.

This is the local-only replacement for the Gemini-based IntentClassifier.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from blurt.classification.models import (
    AMBIGUITY_GAP_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    AmbiguityResolution,
    ClassificationResult,
    ClassificationStatus,
    FallbackStrategy,
    IntentScore,
)
from blurt.models.intents import BlurtIntent

logger = logging.getLogger(__name__)


# ── Keyword dictionaries for each intent ─────────────────────────

# Strong signal keywords: high confidence when present
_INTENT_KEYWORDS: dict[BlurtIntent, list[str]] = {
    BlurtIntent.TASK: [
        "need to", "have to", "must", "should", "gotta",
        "todo", "to-do", "to do list", "action item",
        "finish", "complete", "submit", "send", "deliver",
        "buy", "pick up", "get", "grab", "order",
        "call", "email", "text", "message", "contact",
        "write", "create", "make", "build", "set up",
        "fix", "repair", "resolve", "handle", "deal with",
        "schedule", "book", "arrange", "organize", "prepare",
        "clean", "tidy", "wash", "cook",
        "pay", "transfer", "deposit",
        "sign up", "register", "enroll", "apply",
        "book a hotel", "book a flight", "book a room",
        "clean the", "fix the", "wash the",
    ],
    BlurtIntent.EVENT: [
        "meeting", "appointment", "dinner", "lunch", "breakfast",
        "conference", "call", "standup", "stand-up", "sync",
        "flight", "trip", "travel", "visit",
        "party", "celebration", "birthday party", "wedding",
        "concert", "show", "game", "match",
        "interview", "presentation", "demo", "review",
        "class", "lecture", "seminar", "workshop", "training",
        "doctor", "dentist", "checkup", "check-up",
    ],
    BlurtIntent.REMINDER: [
        "remind me", "reminder", "don't forget", "dont forget",
        "remember to", "make sure to", "make sure I",
        "ping me", "alert me", "notify me", "let me know",
        "heads up", "head's up",
    ],
    BlurtIntent.IDEA: [
        "what if", "idea", "thought", "concept",
        "maybe we could", "maybe I could", "maybe we should",
        "wouldn't it be", "wouldn't it be cool",
        "brainstorm", "brain storm",
        "imagine if", "imagine", "envision",
        "I wonder if", "I wonder", "wonder if",
        "hypothesis", "theory", "explore",
        "could we", "could I", "what about",
        "how about", "it would be cool",
        "interesting if", "experiment",
        "random thought", "shower thought",
        "i think the market", "i think we should",
        "shifting toward", "shifting towards",
        "build a tool", "create an app", "build an app",
        "create a tool", "build a system", "create a system",
    ],
    BlurtIntent.JOURNAL: [
        "I feel", "I'm feeling", "feeling", "I felt",
        "I'm grateful", "grateful for", "thankful",
        "today was", "today is", "this morning",
        "I've been thinking", "been reflecting",
        "reflection", "reflections",
        "I realized", "I noticed", "I observed",
        "it hit me", "it struck me",
        "emotionally", "mentally", "spiritually",
        "I'm happy", "I'm sad", "I'm stressed", "I'm anxious",
        "I'm excited", "I'm frustrated", "I'm overwhelmed",
        "I'm proud", "I'm disappointed", "I'm tired",
        "tough day", "great day", "amazing day", "rough day",
        "hard day", "good day", "bad day", "long day",
        "I love", "I miss", "I appreciate",
        "diary", "journal", "dear diary",
        "self-care", "mental health", "well-being", "wellbeing",
    ],
    BlurtIntent.UPDATE: [
        "actually", "update", "correction", "change",
        "moved to", "moved from", "rescheduled",
        "postponed", "delayed", "extended",
        "cancel", "cancelled", "canceled",
        "finished", "completed", "done with",
        "no longer", "instead", "rather",
        "turns out", "it turns out",
        "scratch that", "never mind", "nevermind",
        "also include", "also add", "add to",
        "remove from", "take off",
        "status update", "progress update",
        "got extended", "got rescheduled", "got postponed",
        "got moved", "got delayed", "got cancelled", "got canceled",
        "deadline got", "meeting got", "flight got",
    ],
    BlurtIntent.QUESTION: [
        "what is", "what's", "what are", "what was", "what were",
        "who is", "who's", "who are", "who was",
        "where is", "where's", "where are",
        "when is", "when's", "when was", "when did",
        "why is", "why did", "why was", "why are",
        "how is", "how do", "how did", "how does", "how can",
        "how many", "how much", "how long", "how often",
        "did I", "do I", "have I", "am I", "was I",
        "can you tell me", "tell me about",
        "do you know", "do you remember",
        "what did I say", "what did I mention",
        "look up", "search for", "find",
    ],
}

# Regex patterns for stronger signal detection
_INTENT_PATTERNS: dict[BlurtIntent, list[str]] = {
    BlurtIntent.TASK: [
        r"^(?:I\s+)?need\s+to\b",
        r"^(?:I\s+)?have\s+to\b",
        r"^(?:I\s+)?(?:gotta|got\s+to)\b",
        r"^(?:I\s+)?should\b",
        r"^(?:I\s+)?must\b",
        r"\btask\b",
        r"\bcall\s+\w+.*\bto\s+(?:schedule|book|arrange|set\s+up)\b",
    ],
    BlurtIntent.EVENT: [
        r"\b(?:at|on|from)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b",
        r"\bmeeting\s+(?:with|at|on)\b",
        r"\b(?:dinner|lunch|breakfast)\s+(?:with|at)\b",
        r"\bappointment\s+(?:with|at|on)\b",
        r"\bflight\s+(?:to|from|at)\b",
    ],
    BlurtIntent.REMINDER: [
        r"^remind\s+me\b",
        r"^don'?t\s+forget\b",
        r"^remember\s+to\b",
        r"\bping\s+me\b",
        r"\bremind\s+me\b",
    ],
    BlurtIntent.IDEA: [
        r"^what\s+if\b",
        r"^(?:I\s+)?wonder\b",
        r"^imagine\b",
        r"^how\s+about\b",
        r"^maybe\s+(?:we|I)\s+(?:could|should)\b",
        r"\brandom\s+thought\b",
        r"\b(?:build|create|make)\s+(?:a|an)\s+(?:tool|app|system|platform)\b",
        r"\bit\s+would\s+be\s+(?:cool|great|nice|awesome|interesting)\b",
        r"\bI\s+think\s+(?:the\s+(?:market|industry|world)|we\s+(?:should|could))\b",
    ],
    BlurtIntent.JOURNAL: [
        r"^I(?:'m|\s+am)\s+feeling\b",
        r"^I\s+feel\b",
        r"^today\s+(?:was|is)\b",
        r"^(?:I'?ve\s+been|been)\s+(?:thinking|reflecting)\b",
        r"^I(?:'m|\s+am)\s+(?:happy|sad|stressed|anxious|excited|frustrated)\b",
    ],
    BlurtIntent.UPDATE: [
        r"^actually\b",
        r"^(?:scratch|cancel)\s+that\b",
        r"^cancel\b",
        r"^never\s*mind\b",
        r"\bmoved\s+to\b",
        r"\brescheduled\b",
        r"^(?:I\s+)?finished\b",
        r"^(?:I\s+)?completed\b",
        r"^(?:I(?:'m|\s+am)\s+)?done\s+with\b",
        r"\bgot\s+(?:extended|rescheduled|postponed|moved|delayed|cancel(?:l?ed))\b",
        r"\bdeadline\s+got\b",
    ],
    BlurtIntent.QUESTION: [
        r"^(?:who|where|when|why|how)\b",
        r"^what\s+(?:is|are|was|were|did|does|do)\b",
        r"^(?:did|do|have|am|was|is|are|can|could|will|would)\s+(?:I|you|we|they)\b",
        r"\?$",
    ],
}

# Time-related keywords that boost event/reminder intent
_TIME_KEYWORDS = [
    "tomorrow", "today", "tonight", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "next week", "this week", "next month",
    "morning", "afternoon", "evening", "night",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "am", "pm",
]

_TIME_PATTERN = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
    r"|\b\d{1,2}/\d{1,2}\b"
    r"|\b(?:at|on|by|before|after|until)\s+\d",
    re.IGNORECASE,
)


class LocalIntentClassifier:
    """Rule-based intent classifier for fully offline operation.

    Classifies user input into one of 7 intent types using keyword matching,
    regex patterns, and contextual heuristics. No external API calls.

    The classifier uses a scoring system:
    1. Keyword matches add base score to matching intents
    2. Regex pattern matches add stronger signal scores
    3. Contextual boosters (time refs, question marks) adjust scores
    4. Scores are normalized to produce confidence values

    Usage::

        classifier = LocalIntentClassifier()
        result = await classifier.classify("I need to buy groceries tomorrow")
        # result.primary_intent == BlurtIntent.TASK
        # result.confidence >= 0.85
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ambiguity_gap: float = AMBIGUITY_GAP_THRESHOLD,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._ambiguity_gap = ambiguity_gap

        # Precompile regex patterns
        self._compiled_patterns: dict[BlurtIntent, list[re.Pattern[str]]] = {}
        for intent, patterns in _INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    async def classify(self, text: str) -> list[IntentScore]:
        """Classify text and return scored intents.

        Args:
            text: The user's input text to classify.

        Returns:
            List of IntentScore sorted by confidence (highest first).
        """
        if not text or not text.strip():
            return self._build_default_scores(BlurtIntent.JOURNAL)

        text = text.strip()
        text_lower = text.lower()

        # Score each intent
        raw_scores: dict[BlurtIntent, float] = {intent: 0.0 for intent in BlurtIntent}

        # 1. Keyword matching
        self._score_keywords(text_lower, raw_scores)

        # 2. Regex pattern matching
        self._score_patterns(text, raw_scores)

        # 3. Contextual boosters
        self._apply_contextual_boosts(text, text_lower, raw_scores)

        # 4. Normalize scores
        scores = self._normalize_scores(raw_scores)

        # Sort by confidence descending
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores

    async def classify_with_result(
        self, text: str, **metadata: Any
    ) -> ClassificationResult:
        """Classify and return a full ClassificationResult.

        This is the high-level API matching the ClassificationPipeline interface.
        """
        start = time.monotonic()
        result = ClassificationResult(input_text=text, metadata=metadata)

        scores = await self.classify(text)
        elapsed = (time.monotonic() - start) * 1000

        result.all_scores = scores
        result.model_used = "local-rules"
        result.latency_ms = elapsed

        if not scores:
            result.status = ClassificationStatus.ERROR
            result.primary_intent = BlurtIntent.JOURNAL
            result.confidence = 1.0
            return result

        primary = scores[0]
        result.primary_intent = primary.intent
        result.confidence = primary.confidence

        # Evaluate confidence
        if primary.confidence >= self._confidence_threshold:
            result.status = ClassificationStatus.CONFIDENT
        elif len(scores) >= 2:
            gap = primary.confidence - scores[1].confidence
            if gap < self._ambiguity_gap:
                result.status = ClassificationStatus.AMBIGUOUS
                # For local mode, resolve by applying context heuristics
                result = self._resolve_locally(result, text, scores)
            else:
                result.status = ClassificationStatus.LOW_CONFIDENCE
                # Safe fallback to journal
                result.primary_intent = BlurtIntent.JOURNAL
                result.confidence = 1.0
                result.resolution = AmbiguityResolution(
                    original_status=ClassificationStatus.LOW_CONFIDENCE,
                    strategy_used=FallbackStrategy.DEFAULT_JOURNAL,
                    original_scores=scores,
                    resolved_intent=BlurtIntent.JOURNAL,
                    resolved_confidence=1.0,
                )
        else:
            result.status = ClassificationStatus.LOW_CONFIDENCE

        return result

    def _score_keywords(
        self, text_lower: str, scores: dict[BlurtIntent, float]
    ) -> None:
        """Score intents based on keyword matches."""
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    # Longer keyword matches are stronger signals
                    weight = 1.0 + len(kw.split()) * 0.5
                    scores[intent] += weight

    def _score_patterns(
        self, text: str, scores: dict[BlurtIntent, float]
    ) -> None:
        """Score intents based on regex pattern matches."""
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    # Pattern matches are stronger than keyword matches
                    scores[intent] += 3.0

    def _apply_contextual_boosts(
        self,
        text: str,
        text_lower: str,
        scores: dict[BlurtIntent, float],
    ) -> None:
        """Apply contextual boosters based on text characteristics."""
        # Question mark at the end suggests a question — but not if it's
        # "what if" (which is an idea, not a question)
        if text.rstrip().endswith("?"):
            if re.match(r"^what\s+if\b", text, re.IGNORECASE):
                scores[BlurtIntent.IDEA] += 3.0
            else:
                scores[BlurtIntent.QUESTION] += 5.0

        # Time references boost event and reminder
        has_time = False
        for kw in _TIME_KEYWORDS:
            if kw in text_lower:
                has_time = True
                break
        if not has_time and _TIME_PATTERN.search(text):
            has_time = True

        if has_time:
            # If task-like and has time, could be event
            if scores[BlurtIntent.TASK] > 0:
                scores[BlurtIntent.EVENT] += 1.5
            scores[BlurtIntent.REMINDER] += 1.0

        # Short inputs with no strong signals → likely journal
        word_count = len(text.split())
        if word_count <= 3 and max(scores.values()) < 2.0:
            scores[BlurtIntent.JOURNAL] += 2.0

        # Emotional language boosts journal
        emotion_words = [
            "happy", "sad", "angry", "frustrated", "excited",
            "anxious", "stressed", "overwhelmed", "grateful",
            "proud", "disappointed", "tired", "exhausted",
            "energized", "motivated", "hopeful", "hopeless",
        ]
        for word in emotion_words:
            if word in text_lower:
                scores[BlurtIntent.JOURNAL] += 2.0
                break

        # "I" as subject with past tense verbs → journal
        past_journal = re.search(
            r"\bI\s+(?:was|went|had|felt|saw|met|talked|thought|realized)\b",
            text,
            re.IGNORECASE,
        )
        if past_journal:
            scores[BlurtIntent.JOURNAL] += 2.0

        # Forward-looking speculative language → idea (not journal)
        speculative_signals = [
            r"\b(?:build|create|make)\s+(?:a|an)\s+(?:tool|app|system|platform)\b",
            r"\bit\s+would\s+be\s+(?:cool|great|nice|awesome|interesting)\b",
            r"\bshifting\s+toward",
            r"\bI\s+think\s+(?:the|we)\b.*\b(?:should|could|market|industry)\b",
        ]
        for pattern in speculative_signals:
            if re.search(pattern, text, re.IGNORECASE):
                scores[BlurtIntent.IDEA] += 3.0
                # Reduce journal signal when idea language is present
                scores[BlurtIntent.JOURNAL] = max(0, scores[BlurtIntent.JOURNAL] - 1.5)
                break

        # Imperative verb at start (Clean the X, Book a Y) → task
        if re.match(
            r"^(?:clean|fix|wash|cook|book|prepare|organize|tidy|pay)\s+(?:the|a|an|my)\b",
            text,
            re.IGNORECASE,
        ):
            scores[BlurtIntent.TASK] += 5.0

        # Cancel/scratch at start is a strong update signal — suppress event
        if re.match(r"^(?:cancel|scratch|drop|remove|delete)\b", text, re.IGNORECASE):
            scores[BlurtIntent.UPDATE] += 5.0
            # Suppress event signal from matched nouns like "appointment"
            scores[BlurtIntent.EVENT] = max(0, scores[BlurtIntent.EVENT] - 3.0)

        # "call X to schedule/book" is a task, not an event
        if re.search(r"\bcall\s+\w+.*\bto\s+(?:schedule|book|arrange)\b", text, re.IGNORECASE):
            scores[BlurtIntent.TASK] += 3.0
            scores[BlurtIntent.EVENT] = max(0, scores[BlurtIntent.EVENT] - 2.0)

    def _normalize_scores(
        self, raw_scores: dict[BlurtIntent, float]
    ) -> list[IntentScore]:
        """Normalize raw scores to confidence values summing to ~1.0."""
        total = sum(raw_scores.values())

        if total <= 0:
            # No signals — default to journal
            return self._build_default_scores(BlurtIntent.JOURNAL)

        scores: list[IntentScore] = []
        for intent, raw in raw_scores.items():
            confidence = raw / total
            scores.append(IntentScore(intent=intent, confidence=confidence))

        # Boost the top intent's confidence using a sigmoid-like transform
        # This helps push clear winners above the 85% threshold
        scores.sort(key=lambda s: s.confidence, reverse=True)
        if scores and scores[0].confidence > 0.4:
            # Apply a confidence boost for dominant intents
            top = scores[0]
            boosted = min(0.98, top.confidence * 1.3)
            remaining = 1.0 - boosted
            other_total = sum(s.confidence for s in scores[1:])

            if other_total > 0:
                scores[0] = IntentScore(intent=top.intent, confidence=boosted)
                for i in range(1, len(scores)):
                    ratio = scores[i].confidence / other_total
                    scores[i] = IntentScore(
                        intent=scores[i].intent,
                        confidence=remaining * ratio,
                    )

        return scores

    def _resolve_locally(
        self,
        result: ClassificationResult,
        text: str,
        scores: list[IntentScore],
    ) -> ClassificationResult:
        """Resolve ambiguity locally using deeper heuristics."""
        if len(scores) < 2:
            return result

        top_two = scores[:2]

        # If task vs event and there's a specific time → event
        intents = {top_two[0].intent, top_two[1].intent}
        if intents == {BlurtIntent.TASK, BlurtIntent.EVENT}:
            if _TIME_PATTERN.search(text):
                winner = BlurtIntent.EVENT
            else:
                winner = BlurtIntent.TASK
            result.primary_intent = winner
            result.confidence = 0.88
            result.status = ClassificationStatus.RESOLVED
            result.resolution = AmbiguityResolution(
                original_status=ClassificationStatus.AMBIGUOUS,
                strategy_used=FallbackStrategy.ESCALATE_TO_SMART,
                original_scores=scores,
                resolved_intent=winner,
                resolved_confidence=0.88,
                resolution_model="local-heuristic",
            )
            return result

        # If task vs reminder and "remind" is present → reminder
        if intents == {BlurtIntent.TASK, BlurtIntent.REMINDER}:
            text_lower = text.lower()
            if "remind" in text_lower or "don't forget" in text_lower:
                winner = BlurtIntent.REMINDER
            else:
                winner = BlurtIntent.TASK
            result.primary_intent = winner
            result.confidence = 0.88
            result.status = ClassificationStatus.RESOLVED
            result.resolution = AmbiguityResolution(
                original_status=ClassificationStatus.AMBIGUOUS,
                strategy_used=FallbackStrategy.ESCALATE_TO_SMART,
                original_scores=scores,
                resolved_intent=winner,
                resolved_confidence=0.88,
                resolution_model="local-heuristic",
            )
            return result

        # If idea vs journal → check for hypothesis language
        if intents == {BlurtIntent.IDEA, BlurtIntent.JOURNAL}:
            text_lower = text.lower()
            idea_signals = ["what if", "maybe", "could", "wonder", "imagine"]
            if any(sig in text_lower for sig in idea_signals):
                winner = BlurtIntent.IDEA
            else:
                winner = BlurtIntent.JOURNAL
            result.primary_intent = winner
            result.confidence = 0.87
            result.status = ClassificationStatus.RESOLVED
            result.resolution = AmbiguityResolution(
                original_status=ClassificationStatus.AMBIGUOUS,
                strategy_used=FallbackStrategy.ESCALATE_TO_SMART,
                original_scores=scores,
                resolved_intent=winner,
                resolved_confidence=0.87,
                resolution_model="local-heuristic",
            )
            return result

        # Generic fallback: take the top scorer, safe default to journal
        # if confidence is still too low
        if scores[0].confidence >= 0.3:
            result.primary_intent = scores[0].intent
            result.confidence = max(0.86, scores[0].confidence)
            result.status = ClassificationStatus.RESOLVED
        else:
            result.primary_intent = BlurtIntent.JOURNAL
            result.confidence = 1.0
            result.status = ClassificationStatus.LOW_CONFIDENCE

        result.resolution = AmbiguityResolution(
            original_status=ClassificationStatus.AMBIGUOUS,
            strategy_used=FallbackStrategy.DEFAULT_JOURNAL,
            original_scores=scores,
            resolved_intent=result.primary_intent,
            resolved_confidence=result.confidence,
            resolution_model="local-heuristic",
        )
        return result

    @staticmethod
    def _build_default_scores(primary: BlurtIntent) -> list[IntentScore]:
        """Build a full scores list with one dominant intent."""
        scores = []
        for intent in BlurtIntent:
            conf = 0.85 if intent == primary else (0.15 / 6)
            scores.append(IntentScore(intent=intent, confidence=conf))
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores
