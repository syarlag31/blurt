"""Tests for the episodic memory compression service.

Validates:
- Daily compression of old episodes into summaries
- Weekly and monthly progressive compression
- Key fact preservation (high-emotion, entity-rich, actionable)
- Compression config (age thresholds, batch sizes)
- Summary aggregation (entity mentions, intents, emotions, signals)
- Edge cases (empty, too few episodes, disabled service)
- Anti-shame: no guilt language in summaries
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    EntityRef,
    Episode,
    EpisodeContext,
    InMemoryEpisodicStore,
    InputModality,
)
from blurt.services.compression import (
    CompressionConfig,
    CompressionTier,
    EpisodicCompressionService,
    FullCompressionResult,
    LocalSummaryGenerator,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_episode(
    user_id: str = "user-1",
    raw_text: str = "Test episode",
    intent: str = "journal",
    days_ago: int = 10,
    emotion_primary: str = "trust",
    emotion_intensity: float = 0.5,
    emotion_valence: float = 0.0,
    entities: list[EntityRef] | None = None,
    behavioral_signal: BehavioralSignal = BehavioralSignal.NONE,
    session_id: str = "session-1",
) -> Episode:
    """Create a test episode with configurable age."""
    return Episode(
        user_id=user_id,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        raw_text=raw_text,
        modality=InputModality.VOICE,
        intent=intent,
        intent_confidence=0.9,
        emotion=EmotionSnapshot(
            primary=emotion_primary,
            intensity=emotion_intensity,
            valence=emotion_valence,
        ),
        entities=entities or [],
        behavioral_signal=behavioral_signal,
        context=EpisodeContext(session_id=session_id),
    )


async def _populate_store(
    store: InMemoryEpisodicStore,
    count: int = 5,
    days_ago: int = 10,
    user_id: str = "user-1",
    intent: str = "journal",
) -> list[Episode]:
    """Create and store multiple episodes."""
    episodes = []
    for i in range(count):
        ep = _make_episode(
            user_id=user_id,
            raw_text=f"Episode {i} content",
            intent=intent,
            days_ago=days_ago,
        )
        await store.append(ep)
        episodes.append(ep)
    return episodes


# ── LocalSummaryGenerator Tests ──────────────────────────────────────


class TestLocalSummaryGenerator:
    @pytest.mark.asyncio
    async def test_generates_summary_from_episodes(self):
        gen = LocalSummaryGenerator()
        episodes = [
            _make_episode(raw_text="Had coffee with Alice", intent="journal"),
            _make_episode(raw_text="Finished the report", intent="task"),
        ]
        result = await gen.generate_summary(episodes, CompressionTier.DAILY, [])
        assert "2 entries" in result
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_includes_key_facts(self):
        gen = LocalSummaryGenerator()
        episodes = [_make_episode()]
        key_facts = ["Alice joined the team", "Project X deadline moved"]
        result = await gen.generate_summary(
            episodes, CompressionTier.DAILY, key_facts
        )
        assert "Alice joined the team" in result
        assert "Project X deadline moved" in result

    @pytest.mark.asyncio
    async def test_empty_episodes_returns_empty(self):
        gen = LocalSummaryGenerator()
        result = await gen.generate_summary([], CompressionTier.DAILY, [])
        assert result == ""

    @pytest.mark.asyncio
    async def test_includes_entity_mentions(self):
        gen = LocalSummaryGenerator()
        episodes = [
            _make_episode(
                entities=[EntityRef(name="Alice", entity_type="person")]
            ),
            _make_episode(
                entities=[EntityRef(name="Alice", entity_type="person")]
            ),
        ]
        result = await gen.generate_summary(episodes, CompressionTier.DAILY, [])
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_weekly_format(self):
        gen = LocalSummaryGenerator()
        episodes = [
            _make_episode(days_ago=35),
            _make_episode(days_ago=32),
        ]
        result = await gen.generate_summary(episodes, CompressionTier.WEEKLY, [])
        assert "2 entries" in result

    @pytest.mark.asyncio
    async def test_no_shame_language(self):
        """Summaries should never contain guilt/shame language."""
        gen = LocalSummaryGenerator()
        episodes = [_make_episode(days_ago=100)]
        result = await gen.generate_summary(episodes, CompressionTier.MONTHLY, [])
        shame_words = ["overdue", "missed", "behind", "failed", "guilt", "streak"]
        for word in shame_words:
            assert word.lower() not in result.lower(), (
                f"Summary contains shame word: {word}"
            )


# ── CompressionConfig Tests ──────────────────────────────────────────


class TestCompressionConfig:
    def test_defaults(self):
        config = CompressionConfig()
        assert config.daily_age_days == 7
        assert config.weekly_age_days == 30
        assert config.monthly_age_days == 90
        assert config.min_episodes_per_summary == 2
        assert config.enabled is True

    def test_custom_config(self):
        config = CompressionConfig(
            daily_age_days=3,
            weekly_age_days=14,
            min_episodes_per_summary=1,
        )
        assert config.daily_age_days == 3
        assert config.weekly_age_days == 14


# ── Daily Compression Tests ──────────────────────────────────────────


class TestDailyCompression:
    @pytest.mark.asyncio
    async def test_compresses_old_episodes(self):
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=5, days_ago=10)

        service = EpisodicCompressionService(store=store)
        result = await service.compress_tier("user-1", CompressionTier.DAILY)

        assert result.success
        assert result.summaries_created >= 1
        assert result.episodes_compressed == 5

    @pytest.mark.asyncio
    async def test_skips_recent_episodes(self):
        store = InMemoryEpisodicStore()
        # Episodes only 2 days old (below 7-day threshold)
        await _populate_store(store, count=5, days_ago=2)

        service = EpisodicCompressionService(store=store)
        result = await service.compress_tier("user-1", CompressionTier.DAILY)

        assert result.summaries_created == 0
        assert result.episodes_compressed == 0

    @pytest.mark.asyncio
    async def test_marks_episodes_as_compressed(self):
        store = InMemoryEpisodicStore()
        episodes = await _populate_store(store, count=3, days_ago=10)

        service = EpisodicCompressionService(store=store)
        await service.compress_tier("user-1", CompressionTier.DAILY)

        for ep in episodes:
            stored_ep = await store.get(ep.id)
            assert stored_ep is not None
            assert stored_ep.is_compressed is True
            assert stored_ep.compressed_into_id is not None

    @pytest.mark.asyncio
    async def test_creates_summary_with_correct_metadata(self):
        store = InMemoryEpisodicStore()
        entities = [
            EntityRef(name="Alice", entity_type="person"),
            EntityRef(name="Project X", entity_type="project"),
        ]
        for i in range(3):
            ep = _make_episode(
                raw_text=f"Discussed {i} with Alice on Project X",
                days_ago=10,
                entities=entities,
                intent="task",
            )
            await store.append(ep)

        service = EpisodicCompressionService(store=store)
        await service.compress_tier("user-1", CompressionTier.DAILY)

        summaries = await store.get_summaries("user-1")
        assert len(summaries) >= 1

        summary = summaries[0]
        assert summary.episode_count == 3
        assert "Alice" in summary.entity_mentions
        assert "Project X" in summary.entity_mentions
        assert "task" in summary.intent_distribution

    @pytest.mark.asyncio
    async def test_too_few_episodes_skips(self):
        store = InMemoryEpisodicStore()
        # Only 1 episode, below min_episodes_per_summary=2
        await _populate_store(store, count=1, days_ago=10)

        service = EpisodicCompressionService(store=store)
        result = await service.compress_tier("user-1", CompressionTier.DAILY)

        assert result.summaries_created == 0

    @pytest.mark.asyncio
    async def test_groups_by_date(self):
        store = InMemoryEpisodicStore()
        # 3 episodes on day 10, 3 episodes on day 11
        await _populate_store(store, count=3, days_ago=10)
        await _populate_store(store, count=3, days_ago=11)

        service = EpisodicCompressionService(store=store)
        result = await service.compress_tier("user-1", CompressionTier.DAILY)

        assert result.summaries_created == 2
        assert result.episodes_compressed == 6

    @pytest.mark.asyncio
    async def test_already_compressed_episodes_skipped(self):
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=3, days_ago=10)

        service = EpisodicCompressionService(store=store)

        # First compression
        result1 = await service.compress_tier("user-1", CompressionTier.DAILY)
        assert result1.episodes_compressed == 3

        # Second compression should find nothing
        result2 = await service.compress_tier("user-1", CompressionTier.DAILY)
        assert result2.episodes_compressed == 0

    @pytest.mark.asyncio
    async def test_respects_custom_age_threshold(self):
        store = InMemoryEpisodicStore()
        # Episodes 5 days old
        await _populate_store(store, count=3, days_ago=5)

        # Default threshold (7 days) — should skip
        service_default = EpisodicCompressionService(store=store)
        result = await service_default.compress_tier("user-1", CompressionTier.DAILY)
        assert result.episodes_compressed == 0

        # Custom threshold (3 days) — should compress
        config = CompressionConfig(daily_age_days=3)
        service_custom = EpisodicCompressionService(store=store, config=config)
        result = await service_custom.compress_tier("user-1", CompressionTier.DAILY)
        assert result.episodes_compressed == 3


# ── Key Fact Preservation Tests ──────────────────────────────────────


class TestKeyFactPreservation:
    @pytest.mark.asyncio
    async def test_preserves_high_emotion_episodes(self):
        store = InMemoryEpisodicStore()
        ep = _make_episode(
            raw_text="I'm so excited about the promotion!",
            days_ago=10,
            emotion_intensity=2.5,  # Above threshold of 2.0
        )
        await store.append(ep)
        # Add a second to meet min_episodes_per_summary
        ep2 = _make_episode(raw_text="Regular update", days_ago=10)
        await store.append(ep2)

        service = EpisodicCompressionService(store=store)
        result = await service.compress_tier("user-1", CompressionTier.DAILY)

        assert result.key_facts_preserved >= 1
        # The summary text should contain the emotional content
        summaries = await store.get_summaries("user-1")
        assert len(summaries) >= 1

    @pytest.mark.asyncio
    async def test_preserves_entity_rich_episodes(self):
        service = EpisodicCompressionService(store=InMemoryEpisodicStore())
        ep = _make_episode(
            raw_text="Meeting with Alice, Bob, and Charlie about Project X",
            entities=[
                EntityRef(name="Alice", entity_type="person"),
                EntityRef(name="Bob", entity_type="person"),
                EntityRef(name="Charlie", entity_type="person"),
            ],
        )
        key_facts = service._extract_key_facts([ep])
        assert len(key_facts) == 1
        assert "Alice" in key_facts[0]

    @pytest.mark.asyncio
    async def test_preserves_actionable_intents(self):
        service = EpisodicCompressionService(store=InMemoryEpisodicStore())
        task_ep = _make_episode(raw_text="Ship the feature by Friday", intent="task")
        event_ep = _make_episode(raw_text="Team standup at 10am", intent="event")
        journal_ep = _make_episode(raw_text="Feeling good today", intent="journal")

        key_facts = service._extract_key_facts([task_ep, event_ep, journal_ep])
        assert "Ship the feature by Friday" in key_facts
        assert "Team standup at 10am" in key_facts
        # Journal with low emotion/entities is NOT a key fact
        assert "Feeling good today" not in key_facts

    @pytest.mark.asyncio
    async def test_preserves_behavioral_signals(self):
        service = EpisodicCompressionService(store=InMemoryEpisodicStore())
        completed_ep = _make_episode(
            raw_text="Completed the review",
            behavioral_signal=BehavioralSignal.COMPLETED,
            intent="journal",  # Not in key_fact_intents
            emotion_intensity=0.1,  # Low emotion
        )
        key_facts = service._extract_key_facts([completed_ep])
        assert "Completed the review" in key_facts

    @pytest.mark.asyncio
    async def test_skipped_signal_not_key_fact(self):
        service = EpisodicCompressionService(store=InMemoryEpisodicStore())
        skipped_ep = _make_episode(
            raw_text="Skipped this one",
            behavioral_signal=BehavioralSignal.SKIPPED,
            intent="journal",
            emotion_intensity=0.1,
        )
        key_facts = service._extract_key_facts([skipped_ep])
        assert len(key_facts) == 0


# ── Full Compression Cycle Tests ─────────────────────────────────────


class TestFullCompression:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=5, days_ago=10)

        service = EpisodicCompressionService(store=store)
        result = await service.run_full_compression("user-1")

        assert isinstance(result, FullCompressionResult)
        assert result.daily.summaries_created >= 1
        assert result.daily.episodes_compressed == 5
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_disabled_compression(self):
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=5, days_ago=10)

        config = CompressionConfig(enabled=False)
        service = EpisodicCompressionService(store=store, config=config)
        result = await service.run_full_compression("user-1")

        assert result.total_summaries_created == 0
        assert result.total_episodes_compressed == 0

    @pytest.mark.asyncio
    async def test_result_serialization(self):
        result = FullCompressionResult()
        result.daily.summaries_created = 3
        result.daily.episodes_compressed = 15

        d = result.to_dict()
        assert d["daily"]["summaries_created"] == 3
        assert d["total_summaries_created"] == 3
        assert d["total_episodes_compressed"] == 15
        assert d["success"] is True

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self):
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=3, days_ago=10, user_id="user-A")
        await _populate_store(store, count=3, days_ago=10, user_id="user-B")

        service = EpisodicCompressionService(store=store)
        result_a = await service.run_full_compression("user-A")
        result_b = await service.run_full_compression("user-B")

        assert result_a.daily.episodes_compressed == 3
        assert result_b.daily.episodes_compressed == 3


# ── Batching Tests ───────────────────────────────────────────────────


class TestBatching:
    def test_batch_small_set(self):
        service = EpisodicCompressionService(store=InMemoryEpisodicStore())
        episodes = [_make_episode() for _ in range(5)]
        batches = service._batch_episodes(episodes)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_batch_large_set(self):
        config = CompressionConfig(max_episodes_per_summary=3, min_episodes_per_summary=2)
        service = EpisodicCompressionService(
            store=InMemoryEpisodicStore(), config=config
        )
        episodes = [_make_episode() for _ in range(8)]
        batches = service._batch_episodes(episodes)
        # 3 + 3 + 2 = 8, so 3 batches
        assert len(batches) == 3

    def test_batch_remainder_merged(self):
        config = CompressionConfig(max_episodes_per_summary=3, min_episodes_per_summary=2)
        service = EpisodicCompressionService(
            store=InMemoryEpisodicStore(), config=config
        )
        episodes = [_make_episode() for _ in range(4)]
        batches = service._batch_episodes(episodes)
        # 3 + 1 remainder → remainder merged into first batch → 1 batch of 4
        assert len(batches) == 1 or (len(batches) == 2 and len(batches[1]) >= 2)


# ── Compression Stats Tests ─────────────────────────────────────────


class TestCompressionStats:
    @pytest.mark.asyncio
    async def test_stats_after_compression(self):
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=5, days_ago=10)

        service = EpisodicCompressionService(store=store)
        await service.run_full_compression("user-1")

        stats = await service.get_compression_stats("user-1")
        assert stats["total_episodes"] == 5
        assert stats["total_summaries"] >= 1
        assert stats["compression_ratio"] > 0

    @pytest.mark.asyncio
    async def test_stats_empty_user(self):
        store = InMemoryEpisodicStore()
        service = EpisodicCompressionService(store=store)
        stats = await service.get_compression_stats("nonexistent")
        assert stats["total_episodes"] == 0
        assert stats["compression_ratio"] == 0.0


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_store(self):
        store = InMemoryEpisodicStore()
        service = EpisodicCompressionService(store=store)
        result = await service.run_full_compression("user-1")
        assert result.total_summaries_created == 0
        assert result.success

    @pytest.mark.asyncio
    async def test_compression_preserves_raw_episodes(self):
        """Raw episodes must never be deleted, only marked."""
        store = InMemoryEpisodicStore()
        episodes = await _populate_store(store, count=3, days_ago=10)

        service = EpisodicCompressionService(store=store)
        await service.run_full_compression("user-1")

        # All episodes still exist in store
        for ep in episodes:
            stored = await store.get(ep.id)
            assert stored is not None, "Episode was deleted instead of marked"

    @pytest.mark.asyncio
    async def test_idempotent_compression(self):
        """Running compression twice should not double-compress."""
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=5, days_ago=10)

        service = EpisodicCompressionService(store=store)

        await service.run_full_compression("user-1")
        summaries_after_first = len(await store.get_summaries("user-1"))

        result2 = await service.run_full_compression("user-1")
        summaries_after_second = len(await store.get_summaries("user-1"))

        # Second run should not create new daily summaries
        assert result2.daily.summaries_created == 0
        assert summaries_after_second == summaries_after_first

    @pytest.mark.asyncio
    async def test_custom_summary_generator(self):
        """Supports plugging in a custom summary generator."""
        store = InMemoryEpisodicStore()
        await _populate_store(store, count=3, days_ago=10)

        class CustomGenerator:
            async def generate_summary(self, episodes, tier, key_facts):
                return f"Custom: {len(episodes)} episodes at {tier.value} tier"

        service = EpisodicCompressionService(
            store=store, summary_generator=CustomGenerator()
        )
        await service.run_full_compression("user-1")

        summaries = await store.get_summaries("user-1")
        assert any("Custom:" in s.summary_text for s in summaries)

    @pytest.mark.asyncio
    async def test_as_of_parameter(self):
        """Compression should work with a custom reference time."""
        store = InMemoryEpisodicStore()
        # Episodes created 5 days ago
        await _populate_store(store, count=3, days_ago=5)

        service = EpisodicCompressionService(store=store)

        # With "now" — 5 days ago is under threshold
        result_now = await service.run_full_compression("user-1")
        assert result_now.daily.episodes_compressed == 0

        # With "as_of" 10 days in the future — 5 days ago becomes 15 days old
        future = datetime.now(timezone.utc) + timedelta(days=10)
        result_future = await service.run_full_compression("user-1", as_of=future)
        assert result_future.daily.episodes_compressed == 3
