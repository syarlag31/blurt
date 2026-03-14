"""Memory promotion pipeline.

Manages the flow of memory items across three tiers:
  working → episodic → semantic

Promotion decisions are based on:
- Importance scoring (relevance, repetition, recency)
- Configurable thresholds per tier transition
- Entity consolidation for semantic promotion
- Append-only episodic storage (items are never deleted, only promoted)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from blurt.core.memory.models import (
    EpisodicMemoryItem,
    MemoryTier,
    PromotionEvent,
    RelationshipEdge,
    SemanticMemoryItem,
    WorkingMemoryItem,
)
from blurt.core.memory.scoring import ImportanceScorer

logger = logging.getLogger(__name__)


@dataclass
class PromotionThresholds:
    """Thresholds that control when items get promoted between tiers."""

    # Minimum importance score for working → episodic promotion
    working_to_episodic: float = 0.3

    # Minimum importance score for episodic → semantic promotion
    episodic_to_semantic: float = 0.6

    # Minimum mention count before episodic → semantic is considered
    min_mentions_for_semantic: int = 2

    # Minimum entity count for episodic → semantic (needs entities to graph)
    min_entities_for_semantic: int = 1

    # Auto-promote working items after this many seconds (session timeout)
    working_max_age_seconds: float = 1800.0  # 30 minutes

    # Auto-promote working items with high access count regardless of score
    working_auto_promote_access_count: int = 5


@dataclass
class MemoryStore:
    """In-memory store for the three memory tiers.

    This is a simple in-process store. In production, this would be backed
    by encrypted storage (SQLite for local, cloud DB for hosted).
    """

    working: list[WorkingMemoryItem] = field(default_factory=list)
    episodic: list[EpisodicMemoryItem] = field(default_factory=list)
    semantic: list[SemanticMemoryItem] = field(default_factory=list)
    promotion_log: list[PromotionEvent] = field(default_factory=list)

    def find_semantic_by_entity(self, entity_name: str) -> SemanticMemoryItem | None:
        """Find existing semantic node by entity name."""
        for item in self.semantic:
            if item.entity and item.entity.name.lower() == entity_name.lower():
                return item
        return None

    def find_semantic_by_id(self, item_id: str) -> SemanticMemoryItem | None:
        """Find semantic node by ID."""
        for item in self.semantic:
            if item.id == item_id:
                return item
        return None


class MemoryPromotionPipeline:
    """Orchestrates memory promotion across the three-tier system.

    The pipeline evaluates all items in each tier and promotes those that
    exceed configured thresholds. Promotion is non-destructive for episodic
    (append-only) — items remain in episodic but also create/update semantic nodes.
    Working items are moved (removed from working, added to episodic).
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        scorer: ImportanceScorer | None = None,
        thresholds: PromotionThresholds | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.scorer = scorer or ImportanceScorer()
        self.thresholds = thresholds or PromotionThresholds()

    def ingest(self, item: WorkingMemoryItem) -> WorkingMemoryItem:
        """Add a new item to working memory.

        This is the entry point for all blurts into the memory system.
        """
        self.store.working.append(item)
        logger.debug("Ingested working memory item %s", item.id)
        return item

    def run_promotion_cycle(self) -> list[PromotionEvent]:
        """Run a full promotion cycle across all tiers.

        Returns a list of all promotion events that occurred.
        """
        events: list[PromotionEvent] = []

        # Phase 1: Promote working → episodic
        events.extend(self._promote_working_to_episodic())

        # Phase 2: Promote episodic → semantic
        events.extend(self._promote_episodic_to_semantic())

        return events

    def _promote_working_to_episodic(self) -> list[PromotionEvent]:
        """Evaluate working memory items for promotion to episodic tier."""
        events: list[PromotionEvent] = []
        items_to_remove: list[WorkingMemoryItem] = []

        for item in self.store.working:
            should_promote, reason, score = self._evaluate_working_promotion(item)

            if should_promote:
                episodic_item = self._working_to_episodic(item, score)
                self.store.episodic.append(episodic_item)

                event = PromotionEvent(
                    source_id=item.id,
                    source_tier=MemoryTier.WORKING,
                    target_tier=MemoryTier.EPISODIC,
                    target_id=episodic_item.id,
                    reason=reason,
                    score=score,
                )
                events.append(event)
                self.store.promotion_log.append(event)
                items_to_remove.append(item)

                logger.info(
                    "Promoted %s: working→episodic (score=%.3f, reason=%s)",
                    item.id, score, reason,
                )

        # Remove promoted items from working memory
        for item in items_to_remove:
            self.store.working.remove(item)

        return events

    def _promote_episodic_to_semantic(self) -> list[PromotionEvent]:
        """Evaluate episodic memory items for promotion to semantic tier."""
        events: list[PromotionEvent] = []

        for item in self.store.episodic:
            should_promote, reason, score = self._evaluate_episodic_promotion(item)

            if should_promote:
                semantic_items = self._episodic_to_semantic(item, score)

                for sem_item in semantic_items:
                    event = PromotionEvent(
                        source_id=item.id,
                        source_tier=MemoryTier.EPISODIC,
                        target_tier=MemoryTier.SEMANTIC,
                        target_id=sem_item.id,
                        reason=reason,
                        score=score,
                    )
                    events.append(event)
                    self.store.promotion_log.append(event)

                    logger.info(
                        "Promoted %s: episodic→semantic (entity=%s, score=%.3f, reason=%s)",
                        item.id,
                        sem_item.entity.name if sem_item.entity else "N/A",
                        score,
                        reason,
                    )

        return events

    def _evaluate_working_promotion(
        self, item: WorkingMemoryItem
    ) -> tuple[bool, str, float]:
        """Decide whether a working memory item should be promoted.

        Returns (should_promote, reason, score).
        """
        score = self.scorer.score_working_item(item)

        # Check age-based auto-promotion (session timeout)
        if item.age_seconds >= self.thresholds.working_max_age_seconds:
            return True, "session_timeout", score

        # Check access-count auto-promotion
        if item.access_count >= self.thresholds.working_auto_promote_access_count:
            return True, "high_access_count", score

        # Check score threshold
        if score >= self.thresholds.working_to_episodic:
            return True, "importance_threshold", score

        return False, "", score

    def _evaluate_episodic_promotion(
        self, item: EpisodicMemoryItem
    ) -> tuple[bool, str, float]:
        """Decide whether an episodic memory item should be promoted.

        Returns (should_promote, reason, score).
        """
        # Must have entities to promote to semantic graph
        if len(item.entities) < self.thresholds.min_entities_for_semantic:
            return False, "insufficient_entities", 0.0

        # Must have enough mentions to indicate pattern
        if item.mention_count < self.thresholds.min_mentions_for_semantic:
            return False, "insufficient_mentions", 0.0

        # Check if already fully promoted (all entities already in semantic)
        all_entities_exist = all(
            self.store.find_semantic_by_entity(e.name) is not None
            for e in item.entities
        )
        # If entities already exist, check if this episodic item is already linked
        if all_entities_exist:
            all_linked = all(
                item.id in (sem.source_episodic_ids or [])
                for e in item.entities
                if (sem := self.store.find_semantic_by_entity(e.name)) is not None
            )
            if all_linked:
                return False, "already_promoted", 0.0

        score = self.scorer.score_episodic_item(item)

        if score >= self.thresholds.episodic_to_semantic:
            return True, "importance_threshold", score

        return False, "", score

    def _working_to_episodic(
        self, item: WorkingMemoryItem, score: float
    ) -> EpisodicMemoryItem:
        """Convert a working memory item into an episodic memory item."""
        return EpisodicMemoryItem(
            content=item.content,
            intent=item.intent,
            entities=list(item.entities),  # copy
            emotion=item.emotion,
            created_at=item.created_at,
            promoted_at=datetime.now(timezone.utc),
            source_working_id=item.id,
            access_count=item.access_count,
            mention_count=1,
            importance_score=score,
            metadata=dict(item.metadata),
        )

    def _episodic_to_semantic(
        self, item: EpisodicMemoryItem, score: float
    ) -> list[SemanticMemoryItem]:
        """Promote episodic item to semantic memory.

        Creates or updates semantic nodes for each entity in the item.
        Also builds relationship edges between co-occurring entities.
        Returns the list of created/updated semantic items.
        """
        semantic_items: list[SemanticMemoryItem] = []
        now = datetime.now(timezone.utc)

        # Create/update a semantic node for each entity
        entity_nodes: dict[str, SemanticMemoryItem] = {}
        for entity in item.entities:
            existing = self.store.find_semantic_by_entity(entity.name)

            if existing:
                # Update existing node
                existing.mention_count += 1
                existing.importance_score = max(existing.importance_score, score)
                existing.last_accessed = now
                if item.id not in existing.source_episodic_ids:
                    existing.source_episodic_ids.append(item.id)
                entity_nodes[entity.name] = existing
                semantic_items.append(existing)
            else:
                # Create new semantic node
                new_node = SemanticMemoryItem(
                    entity=entity,
                    summary=item.content,
                    source_episodic_ids=[item.id],
                    mention_count=1,
                    importance_score=score,
                    created_at=now,
                    promoted_at=now,
                    last_accessed=now,
                )
                self.store.semantic.append(new_node)
                entity_nodes[entity.name] = new_node
                semantic_items.append(new_node)

        # Build relationship edges between co-occurring entities
        entity_list = list(entity_nodes.keys())
        for i, name_a in enumerate(entity_list):
            for name_b in entity_list[i + 1:]:
                node_a = entity_nodes[name_a]
                node_b = entity_nodes[name_b]
                self._update_relationship(node_a, node_b, now)
                self._update_relationship(node_b, node_a, now)

        return semantic_items

    def _update_relationship(
        self,
        source: SemanticMemoryItem,
        target: SemanticMemoryItem,
        now: datetime,
    ) -> None:
        """Update or create a co-mention relationship edge."""
        target_id = target.id

        # Find existing edge
        for edge in source.relationships:
            if edge.target_entity_id == target_id:
                edge.co_mention_count += 1
                edge.last_seen = now
                # Strength grows logarithmically with co-mentions
                edge.strength = min(
                    1.0, edge.co_mention_count / (edge.co_mention_count + 5)
                )
                return

        # Create new edge
        source.relationships.append(
            RelationshipEdge(
                target_entity_id=target_id,
                relationship_type="co_mentioned",
                co_mention_count=1,
                strength=1 / 6,  # 1 / (1 + 5)
                first_seen=now,
                last_seen=now,
            )
        )

    # --- Convenience methods ---

    def touch_working_item(self, item_id: str) -> WorkingMemoryItem | None:
        """Increment access count on a working memory item (re-mention)."""
        for item in self.store.working:
            if item.id == item_id:
                item.access_count += 1
                return item
        return None

    def touch_episodic_item(self, item_id: str) -> EpisodicMemoryItem | None:
        """Increment mention count on an episodic memory item."""
        for item in self.store.episodic:
            if item.id == item_id:
                item.mention_count += 1
                item.access_count += 1
                return item
        return None

    def get_promotion_history(
        self, source_id: str | None = None
    ) -> list[PromotionEvent]:
        """Get promotion events, optionally filtered by source ID."""
        if source_id is None:
            return list(self.store.promotion_log)
        return [e for e in self.store.promotion_log if e.source_id == source_id]

    def get_tier_counts(self) -> dict[str, int]:
        """Get current item counts per tier."""
        return {
            MemoryTier.WORKING.value: len(self.store.working),
            MemoryTier.EPISODIC.value: len(self.store.episodic),
            MemoryTier.SEMANTIC.value: len(self.store.semantic),
        }
