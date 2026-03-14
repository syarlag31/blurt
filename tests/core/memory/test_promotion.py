"""Tests for the memory promotion pipeline."""

from datetime import datetime, timedelta, timezone

from blurt.core.memory.models import (
    EmotionState,
    Entity,
    EpisodicMemoryItem,
    IntentType,
    MemoryTier,
    WorkingMemoryItem,
)
from blurt.core.memory.promotion import (
    MemoryPromotionPipeline,
    PromotionThresholds,
)


def _make_working_item(**kwargs) -> WorkingMemoryItem:
    """Helper to create a working memory item with defaults."""
    defaults = dict(
        content="test blurt",
        intent=IntentType.TASK,
        entities=[Entity(name="Alice", entity_type="person")],
        emotion=EmotionState(primary="joy", intensity=1.5, valence=0.5, arousal=0.5),
        access_count=1,
    )
    defaults.update(kwargs)
    return WorkingMemoryItem(**defaults)  # type: ignore[arg-type]


def _make_episodic_item(**kwargs) -> EpisodicMemoryItem:
    """Helper to create an episodic memory item with defaults."""
    defaults = dict(
        content="test observation",
        intent=IntentType.TASK,
        entities=[Entity(name="Alice", entity_type="person")],
        emotion=EmotionState(primary="joy", intensity=1.5, valence=0.5, arousal=0.5),
        mention_count=3,
        access_count=5,
        importance_score=0.7,
    )
    defaults.update(kwargs)
    return EpisodicMemoryItem(**defaults)  # type: ignore[arg-type]


class TestMemoryPromotionPipeline:
    """Tests for the full promotion pipeline."""

    def setup_method(self) -> None:
        # Use lower thresholds for easier testing
        self.thresholds = PromotionThresholds(
            working_to_episodic=0.2,
            episodic_to_semantic=0.3,
            min_mentions_for_semantic=2,
            min_entities_for_semantic=1,
            working_max_age_seconds=1800.0,
            working_auto_promote_access_count=5,
        )
        self.pipeline = MemoryPromotionPipeline(thresholds=self.thresholds)

    def test_ingest_adds_to_working(self) -> None:
        """Ingesting a blurt should add it to working memory."""
        item = _make_working_item()
        self.pipeline.ingest(item)
        assert len(self.pipeline.store.working) == 1
        assert self.pipeline.store.working[0].id == item.id

    def test_working_to_episodic_by_score(self) -> None:
        """Items that exceed the importance threshold should promote."""
        item = _make_working_item(
            intent=IntentType.TASK,
            access_count=3,
            entities=[
                Entity(name="Alice", entity_type="person"),
                Entity(name="ProjectX", entity_type="project"),
            ],
        )
        self.pipeline.ingest(item)
        events = self.pipeline.run_promotion_cycle()

        # Should have promoted
        w2e_events = [e for e in events if e.source_tier == MemoryTier.WORKING]
        assert len(w2e_events) >= 1
        assert len(self.pipeline.store.working) == 0
        assert len(self.pipeline.store.episodic) >= 1

    def test_working_to_episodic_by_age(self) -> None:
        """Old working memory items should auto-promote via session timeout."""
        item = _make_working_item(
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        self.pipeline.ingest(item)
        events = self.pipeline.run_promotion_cycle()

        w2e_events = [e for e in events if e.source_tier == MemoryTier.WORKING]
        assert len(w2e_events) == 1
        assert w2e_events[0].reason == "session_timeout"

    def test_working_to_episodic_by_access_count(self) -> None:
        """Frequently accessed working items should auto-promote."""
        item = _make_working_item(access_count=6)
        self.pipeline.ingest(item)
        events = self.pipeline.run_promotion_cycle()

        w2e_events = [e for e in events if e.source_tier == MemoryTier.WORKING]
        assert len(w2e_events) == 1
        assert w2e_events[0].reason == "high_access_count"

    def test_episodic_to_semantic_creates_entity_nodes(self) -> None:
        """Promoted episodic items should create semantic entity nodes."""
        entities = [
            Entity(name="Alice", entity_type="person"),
            Entity(name="Acme Corp", entity_type="organization"),
        ]
        episodic = _make_episodic_item(
            entities=entities,
            mention_count=3,
            access_count=5,
        )
        self.pipeline.store.episodic.append(episodic)
        events = self.pipeline.run_promotion_cycle()

        e2s_events = [e for e in events if e.source_tier == MemoryTier.EPISODIC]
        assert len(e2s_events) >= 1
        assert len(self.pipeline.store.semantic) == 2

        # Check entity nodes exist
        alice = self.pipeline.store.find_semantic_by_entity("Alice")
        acme = self.pipeline.store.find_semantic_by_entity("Acme Corp")
        assert alice is not None
        assert acme is not None

    def test_semantic_relationship_edges(self) -> None:
        """Co-occurring entities should create relationship edges."""
        entities = [
            Entity(name="Alice", entity_type="person"),
            Entity(name="Bob", entity_type="person"),
        ]
        episodic = _make_episodic_item(entities=entities, mention_count=3)
        self.pipeline.store.episodic.append(episodic)
        self.pipeline.run_promotion_cycle()

        alice = self.pipeline.store.find_semantic_by_entity("Alice")
        bob = self.pipeline.store.find_semantic_by_entity("Bob")
        assert alice is not None
        assert bob is not None

        # Alice should have edge to Bob
        alice_edges = [e for e in alice.relationships if e.target_entity_id == bob.id]
        assert len(alice_edges) == 1
        assert alice_edges[0].co_mention_count == 1

        # Bob should have edge to Alice
        bob_edges = [e for e in bob.relationships if e.target_entity_id == alice.id]
        assert len(bob_edges) == 1

    def test_relationship_strength_grows(self) -> None:
        """Multiple promotions should strengthen relationship edges."""
        entities = [
            Entity(name="Alice", entity_type="person"),
            Entity(name="Bob", entity_type="person"),
        ]
        # Two separate episodic items mentioning both
        ep1 = _make_episodic_item(entities=entities, mention_count=3)
        ep2 = _make_episodic_item(entities=entities, mention_count=3)
        self.pipeline.store.episodic.extend([ep1, ep2])

        self.pipeline.run_promotion_cycle()

        alice = self.pipeline.store.find_semantic_by_entity("Alice")
        assert alice is not None
        bob = self.pipeline.store.find_semantic_by_entity("Bob")
        assert bob is not None

        edge = [e for e in alice.relationships if e.target_entity_id == bob.id][0]
        assert edge.co_mention_count == 2
        assert edge.strength > 1 / 6  # stronger than initial

    def test_no_duplicate_semantic_nodes(self) -> None:
        """Promoting same entity twice should update, not duplicate."""
        entity = Entity(name="Alice", entity_type="person")
        ep1 = _make_episodic_item(entities=[entity], mention_count=3)
        ep2 = _make_episodic_item(entities=[entity], mention_count=3)
        self.pipeline.store.episodic.extend([ep1, ep2])

        self.pipeline.run_promotion_cycle()

        # Should have only one semantic node for Alice
        alice_nodes = [
            s for s in self.pipeline.store.semantic
            if s.entity and s.entity.name == "Alice"
        ]
        assert len(alice_nodes) == 1
        assert alice_nodes[0].mention_count == 2

    def test_insufficient_mentions_blocks_semantic(self) -> None:
        """Episodic items with too few mentions should not promote."""
        episodic = _make_episodic_item(mention_count=1)
        self.pipeline.store.episodic.append(episodic)
        events = self.pipeline.run_promotion_cycle()

        e2s_events = [e for e in events if e.source_tier == MemoryTier.EPISODIC]
        assert len(e2s_events) == 0

    def test_no_entities_blocks_semantic(self) -> None:
        """Episodic items without entities should not promote to semantic."""
        episodic = _make_episodic_item(entities=[], mention_count=5)
        self.pipeline.store.episodic.append(episodic)
        events = self.pipeline.run_promotion_cycle()

        e2s_events = [e for e in events if e.source_tier == MemoryTier.EPISODIC]
        assert len(e2s_events) == 0

    def test_full_pipeline_working_to_semantic(self) -> None:
        """End-to-end: working item eventually reaches semantic via episodic."""
        item = _make_working_item(
            content="Meeting with Alice at Acme Corp about ProjectX",
            intent=IntentType.EVENT,
            access_count=6,
            entities=[
                Entity(name="Alice", entity_type="person"),
                Entity(name="Acme Corp", entity_type="organization"),
                Entity(name="ProjectX", entity_type="project"),
            ],
            emotion=EmotionState(primary="anticipation", intensity=2.0, valence=0.6, arousal=0.7),
        )
        self.pipeline.ingest(item)

        # First cycle: working → episodic
        events1 = self.pipeline.run_promotion_cycle()
        assert any(e.source_tier == MemoryTier.WORKING for e in events1)
        assert len(self.pipeline.store.episodic) == 1

        # Simulate re-mentions to boost episodic item
        ep_item = self.pipeline.store.episodic[0]
        ep_item.mention_count = 4
        ep_item.access_count = 6

        # Second cycle: episodic → semantic
        events2 = self.pipeline.run_promotion_cycle()
        assert any(e.source_tier == MemoryTier.EPISODIC for e in events2)
        assert len(self.pipeline.store.semantic) == 3  # Alice, Acme Corp, ProjectX

    def test_touch_working_item(self) -> None:
        """Touching a working item should increment access count."""
        item = _make_working_item(access_count=1)
        self.pipeline.ingest(item)
        result = self.pipeline.touch_working_item(item.id)
        assert result is not None
        assert result.access_count == 2

    def test_touch_episodic_item(self) -> None:
        """Touching an episodic item should increment mention count."""
        item = _make_episodic_item(mention_count=1)
        self.pipeline.store.episodic.append(item)
        result = self.pipeline.touch_episodic_item(item.id)
        assert result is not None
        assert result.mention_count == 2
        assert result.access_count == 6  # was 5, now 6

    def test_promotion_history(self) -> None:
        """Promotion events should be logged and retrievable."""
        item = _make_working_item(access_count=6)
        self.pipeline.ingest(item)
        self.pipeline.run_promotion_cycle()

        history = self.pipeline.get_promotion_history()
        assert len(history) >= 1

        # Filter by source
        item_history = self.pipeline.get_promotion_history(source_id=item.id)
        assert len(item_history) >= 1
        assert all(e.source_id == item.id for e in item_history)

    def test_tier_counts(self) -> None:
        """Tier counts should reflect current state."""
        self.pipeline.ingest(_make_working_item())
        self.pipeline.ingest(_make_working_item())
        self.pipeline.store.episodic.append(_make_episodic_item())

        counts = self.pipeline.get_tier_counts()
        assert counts["working"] == 2
        assert counts["episodic"] == 1
        assert counts["semantic"] == 0

    def test_episodic_preserves_source_link(self) -> None:
        """Promoted episodic items should link back to source working ID."""
        item = _make_working_item(access_count=6)
        self.pipeline.ingest(item)
        self.pipeline.run_promotion_cycle()

        assert len(self.pipeline.store.episodic) == 1
        episodic = self.pipeline.store.episodic[0]
        assert episodic.source_working_id == item.id

    def test_semantic_preserves_episodic_links(self) -> None:
        """Semantic nodes should track which episodic items fed them."""
        ep = _make_episodic_item(mention_count=3)
        self.pipeline.store.episodic.append(ep)
        self.pipeline.run_promotion_cycle()

        for sem in self.pipeline.store.semantic:
            assert ep.id in sem.source_episodic_ids

    def test_already_promoted_episodic_skipped(self) -> None:
        """Episodic items already fully linked to semantic should not re-promote."""
        ep = _make_episodic_item(
            entities=[Entity(name="Alice", entity_type="person")],
            mention_count=3,
        )
        self.pipeline.store.episodic.append(ep)

        # First promotion
        events1 = self.pipeline.run_promotion_cycle()
        assert len([e for e in events1 if e.source_tier == MemoryTier.EPISODIC]) >= 1

        # Second promotion — same item should be skipped
        events2 = self.pipeline.run_promotion_cycle()
        e2s_events = [e for e in events2 if e.source_tier == MemoryTier.EPISODIC]
        assert len(e2s_events) == 0


class TestPromotionThresholdEdgeCases:
    """Edge cases for promotion thresholds."""

    def test_very_high_threshold_prevents_promotion(self) -> None:
        """With a very high threshold, nothing should promote."""
        thresholds = PromotionThresholds(
            working_to_episodic=0.99,
            episodic_to_semantic=0.99,
            working_max_age_seconds=999999,
            working_auto_promote_access_count=999999,
        )
        pipeline = MemoryPromotionPipeline(thresholds=thresholds)
        pipeline.ingest(_make_working_item(access_count=1))
        events = pipeline.run_promotion_cycle()

        assert len(events) == 0
        assert len(pipeline.store.working) == 1

    def test_very_low_threshold_promotes_everything(self) -> None:
        """With very low thresholds, everything should promote."""
        thresholds = PromotionThresholds(
            working_to_episodic=0.01,
            episodic_to_semantic=0.01,
            min_mentions_for_semantic=1,
            min_entities_for_semantic=1,
        )
        pipeline = MemoryPromotionPipeline(thresholds=thresholds)
        pipeline.ingest(_make_working_item())
        events = pipeline.run_promotion_cycle()

        # Working → episodic should happen
        assert any(e.source_tier == MemoryTier.WORKING for e in events)

    def test_empty_store_promotion_cycle(self) -> None:
        """Running promotion on empty store should not crash."""
        pipeline = MemoryPromotionPipeline()
        events = pipeline.run_promotion_cycle()
        assert events == []
