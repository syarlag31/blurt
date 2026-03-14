"""Tests for memory importance scoring."""

from datetime import datetime, timedelta, timezone

from blurt.core.memory.models import (
    EmotionState,
    Entity,
    IntentType,
    WorkingMemoryItem,
    EpisodicMemoryItem,
)
from blurt.core.memory.scoring import ImportanceScorer, ScoringWeights


class TestImportanceScorer:
    """Tests for the ImportanceScorer."""

    def setup_method(self) -> None:
        self.scorer = ImportanceScorer()

    def test_score_empty_working_item(self) -> None:
        """Minimal item should still produce a valid score."""
        item = WorkingMemoryItem(content="hello")
        score = self.scorer.score_working_item(item)
        assert 0.0 <= score <= 1.0

    def test_task_intent_scores_higher_than_journal(self) -> None:
        """Task intent should score higher than journal intent."""
        task_item = WorkingMemoryItem(
            content="finish report", intent=IntentType.TASK
        )
        journal_item = WorkingMemoryItem(
            content="feeling okay", intent=IntentType.JOURNAL
        )
        assert self.scorer.score_working_item(task_item) > self.scorer.score_working_item(journal_item)

    def test_more_entities_increases_score(self) -> None:
        """Items with more entities should score higher."""
        no_entities = WorkingMemoryItem(content="do something", intent=IntentType.TASK)
        with_entities = WorkingMemoryItem(
            content="meet Alice at Google",
            intent=IntentType.TASK,
            entities=[
                Entity(name="Alice", entity_type="person"),
                Entity(name="Google", entity_type="organization"),
            ],
        )
        assert self.scorer.score_working_item(with_entities) > self.scorer.score_working_item(no_entities)

    def test_emotion_intensity_increases_score(self) -> None:
        """Higher emotion intensity should increase score."""
        calm = WorkingMemoryItem(
            content="meeting at 3",
            intent=IntentType.EVENT,
            emotion=EmotionState(primary="joy", intensity=0.5, valence=0.3, arousal=0.2),
        )
        intense = WorkingMemoryItem(
            content="amazing breakthrough!",
            intent=IntentType.EVENT,
            emotion=EmotionState(primary="joy", intensity=2.8, valence=0.9, arousal=0.9),
        )
        assert self.scorer.score_working_item(intense) > self.scorer.score_working_item(calm)

    def test_higher_access_count_increases_score(self) -> None:
        """Repeatedly accessed items should score higher."""
        low_access = WorkingMemoryItem(content="test", intent=IntentType.IDEA, access_count=1)
        high_access = WorkingMemoryItem(content="test", intent=IntentType.IDEA, access_count=8)
        assert self.scorer.score_working_item(high_access) > self.scorer.score_working_item(low_access)

    def test_recency_decay(self) -> None:
        """Older items should score lower due to time decay."""
        recent = WorkingMemoryItem(
            content="test",
            intent=IntentType.TASK,
            created_at=datetime.now(timezone.utc),
        )
        old = WorkingMemoryItem(
            content="test",
            intent=IntentType.TASK,
            created_at=datetime.now(timezone.utc) - timedelta(hours=4),
        )
        assert self.scorer.score_working_item(recent) > self.scorer.score_working_item(old)

    def test_score_always_in_range(self) -> None:
        """Score should always be in [0.0, 1.0]."""
        extreme = WorkingMemoryItem(
            content="urgent!",
            intent=IntentType.TASK,
            access_count=100,
            entities=[Entity(name=f"e{i}", entity_type="person") for i in range(20)],
            emotion=EmotionState(primary="anger", intensity=3.0, valence=-1.0, arousal=1.0),
        )
        score = self.scorer.score_working_item(extreme)
        assert 0.0 <= score <= 1.0

    def test_score_episodic_with_mentions(self) -> None:
        """Episodic items with more mentions should score higher."""
        low_mentions = EpisodicMemoryItem(
            content="test", intent=IntentType.IDEA, mention_count=1
        )
        high_mentions = EpisodicMemoryItem(
            content="test", intent=IntentType.IDEA, mention_count=7
        )
        assert self.scorer.score_episodic_item(high_mentions) > self.scorer.score_episodic_item(low_mentions)

    def test_custom_weights(self) -> None:
        """Custom weights should shift scoring behavior."""
        # Scorer that only cares about repetition
        weights = ScoringWeights(
            intent_weight=0.0,
            entity_richness_weight=0.0,
            emotion_intensity_weight=0.0,
            access_count_weight=1.0,
            mention_count_weight=0.0,
            recency_weight=0.0,
        )
        scorer = ImportanceScorer(weights=weights)
        low = WorkingMemoryItem(content="x", access_count=1)
        high = WorkingMemoryItem(content="x", access_count=10)
        assert scorer.score_working_item(high) > scorer.score_working_item(low)
