"""Time-based episodic memory compression service.

Consolidates and summarizes older episodic memory entries while preserving
key facts, entities, emotions, and behavioral signals. Uses a progressive
compression strategy:

  1. Daily compression: Episodes older than 7 days → daily summaries
  2. Weekly compression: Daily summaries older than 30 days → weekly summaries
  3. Monthly compression: Weekly summaries older than 90 days → monthly summaries

Key design decisions:
- Raw episodes are NEVER deleted — only marked as compressed
- Summaries preserve entity mentions, intent distribution, and dominant emotions
- Each compression tier aggregates from the previous tier's summaries
- Key facts (high-emotion, high-entity, task-related) are always preserved verbatim
- Compression runs as a background service, never blocking user interaction
- Anti-shame: no "you missed X days" language in summaries
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    Episode,
    EpisodeSummary,
    EpisodicMemoryStore,
    build_summary,
)

logger = logging.getLogger(__name__)


class CompressionTier(str, Enum):
    """Progressive compression tiers with increasing time windows."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    """Configuration for the compression service.

    Attributes:
        daily_age_days: Minimum age in days before episodes get daily compression.
        weekly_age_days: Minimum age in days before daily summaries get weekly compression.
        monthly_age_days: Minimum age in days before weekly summaries get monthly compression.
        min_episodes_per_summary: Minimum episodes needed to create a summary.
        max_episodes_per_summary: Maximum episodes in a single summary batch.
        preserve_high_emotion_threshold: Emotion intensity above this is preserved verbatim.
        preserve_min_entities: Episodes with >= this many entities are key-fact preserved.
        key_fact_intents: Intent types that are always preserved as key facts.
        enabled: Whether compression is active.
    """

    daily_age_days: int = 7
    weekly_age_days: int = 30
    monthly_age_days: int = 90
    min_episodes_per_summary: int = 2
    max_episodes_per_summary: int = 100
    preserve_high_emotion_threshold: float = 2.0
    preserve_min_entities: int = 3
    key_fact_intents: tuple[str, ...] = ("task", "event", "reminder")
    enabled: bool = True


class SummaryGenerator(Protocol):
    """Protocol for generating natural language summaries.

    Implementors can use Gemini Flash or any LLM to produce human-readable
    summaries from episode batches. A local fallback is provided for
    local-only mode.
    """

    async def generate_summary(
        self,
        episodes: list[Episode],
        tier: CompressionTier,
        key_facts: list[str],
    ) -> str:
        """Generate a natural language summary from episodes.

        Args:
            episodes: The episodes to summarize.
            tier: The compression tier (daily/weekly/monthly).
            key_facts: Preserved key facts to include verbatim.

        Returns:
            A natural language summary string.
        """
        ...


class LocalSummaryGenerator:
    """Local summary generator that doesn't require an LLM.

    Produces structured summaries by aggregating episode metadata.
    Used in local-only mode or as a fallback when the LLM is unavailable.
    """

    async def generate_summary(
        self,
        episodes: list[Episode],
        tier: CompressionTier,
        key_facts: list[str],
    ) -> str:
        if not episodes:
            return ""

        sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
        start = sorted_eps[0].timestamp
        end = sorted_eps[-1].timestamp

        # Build time range description
        if tier == CompressionTier.DAILY:
            period = start.strftime("%B %d, %Y")
        elif tier == CompressionTier.WEEKLY:
            period = f"{start.strftime('%B %d')} - {end.strftime('%B %d, %Y')}"
        else:
            period = f"{start.strftime('%B %Y')}"

        # Aggregate stats
        intent_counts: Counter[str] = Counter()
        entity_names: Counter[str] = Counter()
        emotion_counts: Counter[str] = Counter()

        for ep in episodes:
            intent_counts[ep.intent] += 1
            for ent in ep.entities:
                entity_names[ent.name] += 1
            emotion_counts[ep.emotion.primary] += 1

        parts: list[str] = []
        parts.append(f"{len(episodes)} entries from {period}.")

        # Top intents
        top_intents = intent_counts.most_common(3)
        if top_intents:
            intent_parts = [f"{count} {intent}s" for intent, count in top_intents]
            parts.append(f"Included {', '.join(intent_parts)}.")

        # Top entities
        top_entities = entity_names.most_common(5)
        if top_entities:
            entity_list = [name for name, _ in top_entities]
            parts.append(f"Mentioned: {', '.join(entity_list)}.")

        # Dominant emotion
        top_emotion = emotion_counts.most_common(1)
        if top_emotion:
            parts.append(f"Primary mood: {top_emotion[0][0]}.")

        # Key facts
        if key_facts:
            parts.append("Key facts preserved:")
            for fact in key_facts[:10]:  # Cap at 10 key facts
                parts.append(f"- {fact}")

        return " ".join(parts) if not key_facts else "\n".join(parts)


@dataclass
class CompressionResult:
    """Result of a compression run."""

    tier: CompressionTier
    summaries_created: int = 0
    episodes_compressed: int = 0
    key_facts_preserved: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "summaries_created": self.summaries_created,
            "episodes_compressed": self.episodes_compressed,
            "key_facts_preserved": self.key_facts_preserved,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass
class FullCompressionResult:
    """Result of a full compression cycle across all tiers."""

    daily: CompressionResult = field(
        default_factory=lambda: CompressionResult(tier=CompressionTier.DAILY)
    )
    weekly: CompressionResult = field(
        default_factory=lambda: CompressionResult(tier=CompressionTier.WEEKLY)
    )
    monthly: CompressionResult = field(
        default_factory=lambda: CompressionResult(tier=CompressionTier.MONTHLY)
    )
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def total_summaries_created(self) -> int:
        return (
            self.daily.summaries_created
            + self.weekly.summaries_created
            + self.monthly.summaries_created
        )

    @property
    def total_episodes_compressed(self) -> int:
        return (
            self.daily.episodes_compressed
            + self.weekly.episodes_compressed
            + self.monthly.episodes_compressed
        )

    @property
    def success(self) -> bool:
        return self.daily.success and self.weekly.success and self.monthly.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "daily": self.daily.to_dict(),
            "weekly": self.weekly.to_dict(),
            "monthly": self.monthly.to_dict(),
            "total_summaries_created": self.total_summaries_created,
            "total_episodes_compressed": self.total_episodes_compressed,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
        }


class EpisodicCompressionService:
    """Time-based compression service for episodic memory.

    Progressively compresses older episodes into summaries while preserving
    key facts, entity references, emotional patterns, and behavioral signals.

    Compression is non-destructive: raw episodes are marked as compressed but
    never deleted. Summaries capture aggregated patterns for efficient
    long-term recall and semantic search.

    Usage:
        service = EpisodicCompressionService(store=my_store)
        result = await service.run_full_compression(user_id="user-123")
    """

    def __init__(
        self,
        store: EpisodicMemoryStore,
        config: CompressionConfig | None = None,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._store = store
        self._config = config or CompressionConfig()
        self._generator = summary_generator or LocalSummaryGenerator()

    @property
    def config(self) -> CompressionConfig:
        return self._config

    async def run_full_compression(
        self,
        user_id: str,
        as_of: datetime | None = None,
    ) -> FullCompressionResult:
        """Run a full compression cycle across all tiers.

        Processes daily → weekly → monthly in order, since higher tiers
        depend on lower tier summaries existing.

        Args:
            user_id: The user whose episodes to compress.
            as_of: Reference time for age calculations (default: now UTC).

        Returns:
            FullCompressionResult with details of what was compressed.
        """
        if not self._config.enabled:
            logger.info("Compression disabled, skipping")
            return FullCompressionResult()

        now = as_of or datetime.now(timezone.utc)
        result = FullCompressionResult(started_at=now)

        # Daily compression: raw episodes → daily summaries
        result.daily = await self.compress_tier(
            user_id=user_id,
            tier=CompressionTier.DAILY,
            as_of=now,
        )

        # Weekly compression: daily summaries older than weekly_age_days
        result.weekly = await self.compress_tier(
            user_id=user_id,
            tier=CompressionTier.WEEKLY,
            as_of=now,
        )

        # Monthly compression: weekly summaries older than monthly_age_days
        result.monthly = await self.compress_tier(
            user_id=user_id,
            tier=CompressionTier.MONTHLY,
            as_of=now,
        )

        result.completed_at = datetime.now(timezone.utc)

        logger.info(
            "Compression complete for user %s: %d summaries, %d episodes compressed",
            user_id,
            result.total_summaries_created,
            result.total_episodes_compressed,
        )

        return result

    async def compress_tier(
        self,
        user_id: str,
        tier: CompressionTier,
        as_of: datetime | None = None,
    ) -> CompressionResult:
        """Compress a single tier of episodes.

        Args:
            user_id: The user whose episodes to compress.
            tier: Which compression tier to process.
            as_of: Reference time for age calculations.

        Returns:
            CompressionResult for this tier.
        """
        now = as_of or datetime.now(timezone.utc)
        result = CompressionResult(tier=tier)

        try:
            if tier == CompressionTier.DAILY:
                await self._compress_daily(user_id, now, result)
            elif tier == CompressionTier.WEEKLY:
                await self._compress_weekly(user_id, now, result)
            elif tier == CompressionTier.MONTHLY:
                await self._compress_monthly(user_id, now, result)
        except Exception as exc:
            error_msg = f"Compression error for tier {tier.value}: {exc}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        return result

    async def _compress_daily(
        self,
        user_id: str,
        now: datetime,
        result: CompressionResult,
    ) -> None:
        """Compress raw episodes older than daily_age_days into daily summaries."""
        cutoff = now - timedelta(days=self._config.daily_age_days)

        # Get all uncompressed episodes older than cutoff
        from blurt.memory.episodic import TimeRangeFilter

        episodes = await self._store.query(
            user_id,
            time_range=TimeRangeFilter(end=cutoff),
            include_compressed=False,
            limit=10000,  # Get all eligible episodes
        )

        if not episodes:
            logger.debug("No episodes eligible for daily compression")
            return

        # Group episodes by date
        date_groups = self._group_by_date(episodes)

        for date_key, day_episodes in date_groups.items():
            if len(day_episodes) < self._config.min_episodes_per_summary:
                continue

            # Process in batches if needed
            batches = self._batch_episodes(day_episodes)

            for batch in batches:
                try:
                    await self._compress_batch(
                        user_id=user_id,
                        episodes=batch,
                        tier=CompressionTier.DAILY,
                    )
                    result.summaries_created += 1
                    result.episodes_compressed += len(batch)
                    result.key_facts_preserved += len(
                        self._extract_key_facts(batch)
                    )
                except Exception as exc:
                    error_msg = f"Failed to compress daily batch {date_key}: {exc}"
                    logger.error(error_msg, exc_info=True)
                    result.errors.append(error_msg)

    async def _compress_weekly(
        self,
        user_id: str,
        now: datetime,
        result: CompressionResult,
    ) -> None:
        """Compress daily summaries older than weekly_age_days into weekly summaries."""
        cutoff = now - timedelta(days=self._config.weekly_age_days)

        # Get daily summaries eligible for weekly compression
        summaries = await self._store.get_summaries(user_id, end=cutoff)
        daily_summaries = [
            s for s in summaries
            if self._is_tier_summary(s, CompressionTier.DAILY)
            and not self._is_already_compressed_summary(s)
        ]

        if not daily_summaries:
            logger.debug("No daily summaries eligible for weekly compression")
            return

        # Group by ISO week
        week_groups = self._group_summaries_by_week(daily_summaries)

        for week_key, week_summaries in week_groups.items():
            if len(week_summaries) < 2:  # Need at least 2 daily summaries
                continue

            try:
                await self._merge_summaries(
                    user_id=user_id,
                    summaries=week_summaries,
                    tier=CompressionTier.WEEKLY,
                )
                result.summaries_created += 1
                result.episodes_compressed += sum(
                    s.episode_count for s in week_summaries
                )
            except Exception as exc:
                error_msg = f"Failed to compress weekly batch {week_key}: {exc}"
                logger.error(error_msg, exc_info=True)
                result.errors.append(error_msg)

    async def _compress_monthly(
        self,
        user_id: str,
        now: datetime,
        result: CompressionResult,
    ) -> None:
        """Compress weekly summaries older than monthly_age_days into monthly summaries."""
        cutoff = now - timedelta(days=self._config.monthly_age_days)

        summaries = await self._store.get_summaries(user_id, end=cutoff)
        weekly_summaries = [
            s for s in summaries
            if self._is_tier_summary(s, CompressionTier.WEEKLY)
            and not self._is_already_compressed_summary(s)
        ]

        if not weekly_summaries:
            logger.debug("No weekly summaries eligible for monthly compression")
            return

        # Group by month
        month_groups = self._group_summaries_by_month(weekly_summaries)

        for month_key, month_summaries in month_groups.items():
            if len(month_summaries) < 2:
                continue

            try:
                await self._merge_summaries(
                    user_id=user_id,
                    summaries=month_summaries,
                    tier=CompressionTier.MONTHLY,
                )
                result.summaries_created += 1
                result.episodes_compressed += sum(
                    s.episode_count for s in month_summaries
                )
            except Exception as exc:
                error_msg = f"Failed to compress monthly batch {month_key}: {exc}"
                logger.error(error_msg, exc_info=True)
                result.errors.append(error_msg)

    async def _compress_batch(
        self,
        user_id: str,
        episodes: list[Episode],
        tier: CompressionTier,
    ) -> EpisodeSummary:
        """Compress a batch of episodes into a summary.

        1. Extract key facts from high-value episodes
        2. Generate a natural language summary
        3. Store the summary
        4. Mark source episodes as compressed
        """
        key_facts = self._extract_key_facts(episodes)

        # Generate summary text
        summary_text = await self._generator.generate_summary(
            episodes=episodes,
            tier=tier,
            key_facts=key_facts,
        )

        # Tag the summary with tier metadata
        summary = build_summary(
            user_id=user_id,
            episodes=episodes,
            summary_text=summary_text,
        )

        # Store compression tier in behavioral_signals for identification
        summary.behavioral_signals["_compression_tier"] = tier.value  # type: ignore[assignment]

        # Store and mark episodes
        stored = await self._store.store_summary(summary)
        await self._store.mark_compressed(
            [ep.id for ep in episodes],
            stored.id,
        )

        logger.info(
            "Created %s summary %s from %d episodes (%d key facts preserved)",
            tier.value,
            stored.id,
            len(episodes),
            len(key_facts),
        )

        return stored

    async def _merge_summaries(
        self,
        user_id: str,
        summaries: list[EpisodeSummary],
        tier: CompressionTier,
    ) -> EpisodeSummary:
        """Merge multiple summaries into a higher-tier summary.

        Aggregates entity mentions, intent distributions, emotions,
        and behavioral signals from all source summaries.
        """
        if not summaries:
            raise ValueError("Cannot merge empty summary list")

        sorted_summaries = sorted(summaries, key=lambda s: s.period_start)

        # Aggregate all source episode IDs
        all_episode_ids: list[str] = []
        total_count = 0
        for s in summaries:
            all_episode_ids.extend(s.source_episode_ids)
            total_count += s.episode_count

        # Merge entity mentions
        merged_entities: Counter[str] = Counter()
        for s in summaries:
            merged_entities.update(s.entity_mentions)

        # Merge intent distributions
        merged_intents: Counter[str] = Counter()
        for s in summaries:
            merged_intents.update(s.intent_distribution)

        # Merge behavioral signals (excluding compression tier metadata)
        merged_signals: Counter[str] = Counter()
        for s in summaries:
            for k, v in s.behavioral_signals.items():
                if not k.startswith("_"):
                    merged_signals[k] += v if isinstance(v, int) else 0

        # Collect dominant emotions (take top 3 across all summaries)
        all_emotions: list[EmotionSnapshot] = []
        for s in summaries:
            all_emotions.extend(s.dominant_emotions)
        emotion_counter: Counter[str] = Counter()
        for em in all_emotions:
            emotion_counter[em.primary] += 1
        top_emotions: list[EmotionSnapshot] = []
        for primary, _ in emotion_counter.most_common(3):
            for em in all_emotions:
                if em.primary == primary:
                    top_emotions.append(em)
                    break

        # Combine summary texts
        key_facts = [s.summary_text for s in summaries if s.summary_text]

        summary_text = await self._generator.generate_summary(
            episodes=[],  # No raw episodes for higher-tier merges
            tier=tier,
            key_facts=key_facts,
        )

        # If generator returns empty (no episodes), build a fallback
        if not summary_text:
            period_start = sorted_summaries[0].period_start
            period_end = sorted_summaries[-1].period_end
            summary_text = (
                f"{tier.value.capitalize()} summary: {total_count} entries "
                f"from {period_start.strftime('%B %d')} to {period_end.strftime('%B %d, %Y')}."
            )
            if merged_entities:
                top_5 = merged_entities.most_common(5)
                summary_text += (
                    f" Top mentions: {', '.join(n for n, _ in top_5)}."
                )

        merged = EpisodeSummary(
            user_id=user_id,
            period_start=sorted_summaries[0].period_start,
            period_end=sorted_summaries[-1].period_end,
            source_episode_ids=all_episode_ids,
            episode_count=total_count,
            summary_text=summary_text,
            dominant_emotions=top_emotions,
            entity_mentions=dict(merged_entities),
            intent_distribution=dict(merged_intents),
            behavioral_signals={
                **dict(merged_signals),
                "_compression_tier": tier.value,
                "_source_summary_ids": ",".join(s.id for s in summaries),
            },
        )

        stored = await self._store.store_summary(merged)

        logger.info(
            "Created %s merged summary %s from %d sub-summaries (%d total episodes)",
            tier.value,
            stored.id,
            len(summaries),
            total_count,
        )

        return stored

    def _extract_key_facts(self, episodes: list[Episode]) -> list[str]:
        """Extract key facts from episodes that should be preserved verbatim.

        Key facts are episodes that:
        - Have high emotional intensity (> threshold)
        - Have many entity references (>= min_entities)
        - Are task/event/reminder intents (actionable)
        - Have behavioral signals (user acted on something)
        """
        key_facts: list[str] = []

        for ep in episodes:
            is_key = False

            # High emotion episodes are important
            if ep.emotion.intensity >= self._config.preserve_high_emotion_threshold:
                is_key = True

            # Entity-rich episodes contain important information
            if len(ep.entities) >= self._config.preserve_min_entities:
                is_key = True

            # Actionable intents
            if ep.intent in self._config.key_fact_intents:
                is_key = True

            # Episodes where user took action
            if ep.behavioral_signal not in (
                BehavioralSignal.NONE,
                BehavioralSignal.SKIPPED,
            ):
                is_key = True

            if is_key and ep.raw_text:
                key_facts.append(ep.raw_text)

        return key_facts

    def _group_by_date(
        self, episodes: list[Episode]
    ) -> dict[str, list[Episode]]:
        """Group episodes by calendar date (UTC)."""
        groups: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            date_key = ep.timestamp.strftime("%Y-%m-%d")
            groups[date_key].append(ep)
        return dict(groups)

    def _group_summaries_by_week(
        self, summaries: list[EpisodeSummary]
    ) -> dict[str, list[EpisodeSummary]]:
        """Group summaries by ISO week."""
        groups: dict[str, list[EpisodeSummary]] = defaultdict(list)
        for s in summaries:
            iso = s.period_start.isocalendar()
            week_key = f"{iso[0]}-W{iso[1]:02d}"
            groups[week_key].append(s)
        return dict(groups)

    def _group_summaries_by_month(
        self, summaries: list[EpisodeSummary]
    ) -> dict[str, list[EpisodeSummary]]:
        """Group summaries by year-month."""
        groups: dict[str, list[EpisodeSummary]] = defaultdict(list)
        for s in summaries:
            month_key = s.period_start.strftime("%Y-%m")
            groups[month_key].append(s)
        return dict(groups)

    def _batch_episodes(
        self, episodes: list[Episode]
    ) -> list[list[Episode]]:
        """Split a list of episodes into batches respecting max_episodes_per_summary."""
        max_size = self._config.max_episodes_per_summary
        if len(episodes) <= max_size:
            return [episodes]

        batches: list[list[Episode]] = []
        for i in range(0, len(episodes), max_size):
            batch = episodes[i : i + max_size]
            if len(batch) >= self._config.min_episodes_per_summary:
                batches.append(batch)
            elif batches:
                # Merge small remainder into last batch
                batches[-1].extend(batch)
        return batches

    def _is_tier_summary(
        self, summary: EpisodeSummary, tier: CompressionTier
    ) -> bool:
        """Check if a summary belongs to a specific compression tier."""
        tier_val = summary.behavioral_signals.get("_compression_tier")
        return tier_val == tier.value

    def _is_already_compressed_summary(
        self, summary: EpisodeSummary
    ) -> bool:
        """Check if a summary has already been merged into a higher-tier summary."""
        # A summary is considered compressed if its ID appears as a source
        # in another summary's _source_summary_ids. We check via the
        # behavioral_signals metadata.
        return "_already_merged" in summary.behavioral_signals

    async def get_compression_stats(
        self, user_id: str
    ) -> dict[str, Any]:
        """Get compression statistics for a user.

        Returns counts of episodes, summaries per tier, and compression ratios.
        """
        total_episodes = await self._store.count(user_id)
        all_summaries = await self._store.get_summaries(user_id)

        tier_counts: dict[str, int] = {
            CompressionTier.DAILY.value: 0,
            CompressionTier.WEEKLY.value: 0,
            CompressionTier.MONTHLY.value: 0,
            "unclassified": 0,
        }

        total_compressed_episodes = 0
        for s in all_summaries:
            tier_val = s.behavioral_signals.get("_compression_tier")
            if tier_val in tier_counts:
                tier_counts[tier_val] += 1
            else:
                tier_counts["unclassified"] += 1
            total_compressed_episodes += s.episode_count

        return {
            "total_episodes": total_episodes,
            "total_summaries": len(all_summaries),
            "summaries_by_tier": tier_counts,
            "total_compressed_episodes": total_compressed_episodes,
            "compression_ratio": (
                total_compressed_episodes / total_episodes
                if total_episodes > 0
                else 0.0
            ),
        }
