"""Neon Postgres implementation of the EpisodicMemoryStore.

Stores episodes, summaries, and entity references in Postgres with pgvector
for semantic search. All queries use parameterized SQL — no string interpolation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from collections.abc import Mapping, Sequence

import asyncpg

from blurt.memory.episodic import (
    BehavioralFilter,
    BehavioralSignal,
    EmotionFilter,
    EmotionSnapshot,
    EntityFilter,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodeSummary,
    EpisodicMemoryStore,
    InputModality,
    IntentFilter,
    SessionFilter,
    TimeRangeFilter,
)

logger = logging.getLogger(__name__)


def _row_to_episode(row: Mapping[str, Any], entity_rows: Sequence[Mapping[str, Any]] | None = None) -> Episode:
    """Convert a database row to an Episode dataclass."""
    entities: list[EntityRef] = []
    if entity_rows:
        for er in entity_rows:
            entities.append(EntityRef(
                name=er["name"],
                entity_type=er["entity_type"],
                entity_id=er["entity_id"],
                confidence=er["confidence"],
            ))

    embedding = None
    if row["embedding"] is not None:
        embedding = list(row["embedding"])

    return Episode(
        id=row["id"],
        user_id=row["user_id"],
        timestamp=row["timestamp"],
        raw_text=row["raw_text"],
        modality=InputModality(row["modality"]),
        intent=row["intent"],
        intent_confidence=row["intent_confidence"],
        emotion=EmotionSnapshot(
            primary=row["emotion_primary"],
            intensity=row["emotion_intensity"],
            valence=row["emotion_valence"],
            arousal=row["emotion_arousal"],
            secondary=row["emotion_secondary"],
        ),
        entities=entities,
        behavioral_signal=BehavioralSignal(row["behavioral_signal"]),
        surfaced_task_id=row["surfaced_task_id"],
        context=EpisodeContext(
            time_of_day=row["context_time_of_day"],
            day_of_week=row["context_day_of_week"],
            session_id=row["context_session_id"],
            preceding_episode_id=row["context_preceding_episode_id"],
            active_task_id=row["context_active_task_id"],
        ),
        is_compressed=row["is_compressed"],
        compressed_into_id=row["compressed_into_id"],
        embedding=embedding,
        source_working_id=row["source_working_id"],
    )


def _row_to_summary(row: Mapping[str, Any]) -> EpisodeSummary:
    """Convert a database row to an EpisodeSummary dataclass."""
    dominant_emotions_raw = row["dominant_emotions"]
    if isinstance(dominant_emotions_raw, str):
        dominant_emotions_raw = json.loads(dominant_emotions_raw)
    dominant_emotions = [
        EmotionSnapshot(
            primary=e.get("primary", "trust"),
            intensity=e.get("intensity", 0.0),
            valence=e.get("valence", 0.0),
            arousal=e.get("arousal", 0.0),
        )
        for e in (dominant_emotions_raw or [])
    ]

    entity_mentions = row["entity_mentions"]
    if isinstance(entity_mentions, str):
        entity_mentions = json.loads(entity_mentions)

    intent_distribution = row["intent_distribution"]
    if isinstance(intent_distribution, str):
        intent_distribution = json.loads(intent_distribution)

    behavioral_signals = row["behavioral_signals"]
    if isinstance(behavioral_signals, str):
        behavioral_signals = json.loads(behavioral_signals)

    embedding = None
    if row["embedding"] is not None:
        embedding = list(row["embedding"])

    return EpisodeSummary(
        id=row["id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        source_episode_ids=list(row["source_episode_ids"] or []),
        episode_count=row["episode_count"],
        summary_text=row["summary_text"],
        dominant_emotions=dominant_emotions,
        entity_mentions=entity_mentions or {},
        intent_distribution=intent_distribution or {},
        behavioral_signals=behavioral_signals or {},
        embedding=embedding,
    )


class PgEpisodicStore(EpisodicMemoryStore):
    """Postgres-backed episodic memory store using asyncpg + pgvector.

    All SQL uses parameterized queries. Embedding similarity search
    uses pgvector's cosine distance operator (<=>).
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, episode: Episode) -> Episode:
        """Store a new episode with its entity references."""
        embedding_val = episode.embedding if episode.embedding else None

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO episodes (
                        id, user_id, timestamp, raw_text, modality,
                        intent, intent_confidence,
                        emotion_primary, emotion_intensity, emotion_valence,
                        emotion_arousal, emotion_secondary,
                        behavioral_signal, surfaced_task_id,
                        context_time_of_day, context_day_of_week,
                        context_session_id, context_preceding_episode_id,
                        context_active_task_id,
                        is_compressed, compressed_into_id,
                        embedding, source_working_id
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7,
                        $8, $9, $10,
                        $11, $12,
                        $13, $14,
                        $15, $16,
                        $17, $18,
                        $19,
                        $20, $21,
                        $22, $23
                    )
                    """,
                    episode.id, episode.user_id, episode.timestamp,
                    episode.raw_text, episode.modality.value,
                    episode.intent, episode.intent_confidence,
                    episode.emotion.primary, episode.emotion.intensity,
                    episode.emotion.valence, episode.emotion.arousal,
                    episode.emotion.secondary,
                    episode.behavioral_signal.value, episode.surfaced_task_id,
                    episode.context.time_of_day, episode.context.day_of_week,
                    episode.context.session_id, episode.context.preceding_episode_id,
                    episode.context.active_task_id,
                    episode.is_compressed, episode.compressed_into_id,
                    embedding_val, episode.source_working_id,
                )

                # Insert entity references
                for entity in episode.entities:
                    await conn.execute(
                        """
                        INSERT INTO episode_entities (episode_id, name, entity_type, entity_id, confidence)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (episode_id, name) DO NOTHING
                        """,
                        episode.id, entity.name, entity.entity_type,
                        entity.entity_id, entity.confidence,
                    )

        return episode

    async def get(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM episodes WHERE id = $1", episode_id
            )
            if row is None:
                return None

            entity_rows = await conn.fetch(
                "SELECT * FROM episode_entities WHERE episode_id = $1",
                episode_id,
            )
            return _row_to_episode(row, entity_rows)

    async def query(
        self,
        user_id: str,
        *,
        time_range: TimeRangeFilter | None = None,
        entity_filter: EntityFilter | None = None,
        emotion_filter: EmotionFilter | None = None,
        intent_filter: IntentFilter | None = None,
        behavioral_filter: BehavioralFilter | None = None,
        session_filter: SessionFilter | None = None,
        limit: int = 50,
        offset: int = 0,
        include_compressed: bool = False,
    ) -> list[Episode]:
        """Query episodes with composable filters. Returns newest first."""
        conditions = ["e.user_id = $1"]
        params: list[Any] = [user_id]
        idx = 2

        if not include_compressed:
            conditions.append("e.is_compressed = FALSE")

        if time_range:
            if time_range.start:
                conditions.append(f"e.timestamp >= ${idx}")
                params.append(time_range.start)
                idx += 1
            if time_range.end:
                conditions.append(f"e.timestamp <= ${idx}")
                params.append(time_range.end)
                idx += 1

        if intent_filter:
            conditions.append(f"e.intent = ${idx}")
            params.append(intent_filter.intent)
            idx += 1

        if emotion_filter:
            if emotion_filter.primary:
                conditions.append(f"e.emotion_primary = ${idx}")
                params.append(emotion_filter.primary)
                idx += 1
            if emotion_filter.min_intensity > 0:
                conditions.append(f"e.emotion_intensity >= ${idx}")
                params.append(emotion_filter.min_intensity)
                idx += 1
            if emotion_filter.valence_range:
                lo, hi = emotion_filter.valence_range
                conditions.append(f"e.emotion_valence >= ${idx}")
                params.append(lo)
                idx += 1
                conditions.append(f"e.emotion_valence <= ${idx}")
                params.append(hi)
                idx += 1

        if behavioral_filter:
            conditions.append(f"e.behavioral_signal = ${idx}")
            params.append(behavioral_filter.signal.value)
            idx += 1

        if session_filter:
            conditions.append(f"e.context_session_id = ${idx}")
            params.append(session_filter.session_id)
            idx += 1

        # Entity filter requires a JOIN
        join_clause = ""
        if entity_filter:
            join_clause = " JOIN episode_entities ee ON e.id = ee.episode_id"
            if entity_filter.entity_name:
                conditions.append(f"lower(ee.name) = ${idx}")
                params.append(entity_filter.entity_name)
                idx += 1
            if entity_filter.entity_id:
                conditions.append(f"ee.entity_id = ${idx}")
                params.append(entity_filter.entity_id)
                idx += 1

        where = " AND ".join(conditions)
        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        sql = f"""
            SELECT DISTINCT e.* FROM episodes e{join_clause}
            WHERE {where}
            ORDER BY e.timestamp DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

            episodes = []
            for row in rows:
                entity_rows = await conn.fetch(
                    "SELECT * FROM episode_entities WHERE episode_id = $1",
                    row["id"],
                )
                episodes.append(_row_to_episode(row, entity_rows))
            return episodes

    async def count(self, user_id: str) -> int:
        """Total episode count for a user."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM episodes WHERE user_id = $1", user_id
            )
            return result or 0

    async def get_session_episodes(self, session_id: str) -> list[Episode]:
        """Get all episodes in a session, ordered chronologically."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM episodes
                WHERE context_session_id = $1
                ORDER BY timestamp ASC
                """,
                session_id,
            )
            episodes = []
            for row in rows:
                entity_rows = await conn.fetch(
                    "SELECT * FROM episode_entities WHERE episode_id = $1",
                    row["id"],
                )
                episodes.append(_row_to_episode(row, entity_rows))
            return episodes

    async def semantic_search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[tuple[Episode, float]]:
        """Search by embedding similarity using pgvector cosine distance."""
        # pgvector <=> returns cosine distance; similarity = 1 - distance
        max_distance = 1.0 - min_similarity

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, (embedding <=> $1::vector) AS distance
                FROM episodes
                WHERE user_id = $2
                  AND is_compressed = FALSE
                  AND embedding IS NOT NULL
                  AND (embedding <=> $1::vector) <= $3
                ORDER BY distance ASC
                LIMIT $4
                """,
                str(query_embedding), user_id, max_distance, limit,
            )

            results: list[tuple[Episode, float]] = []
            for row in rows:
                entity_rows = await conn.fetch(
                    "SELECT * FROM episode_entities WHERE episode_id = $1",
                    row["id"],
                )
                episode = _row_to_episode(row, entity_rows)
                similarity = 1.0 - row["distance"]
                results.append((episode, similarity))
            return results

    async def get_entity_timeline(
        self,
        user_id: str,
        entity_name: str,
        limit: int = 20,
    ) -> list[Episode]:
        """Get episodes mentioning a specific entity, newest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.* FROM episodes e
                JOIN episode_entities ee ON e.id = ee.episode_id
                WHERE e.user_id = $1
                  AND lower(ee.name) = $2
                  AND e.is_compressed = FALSE
                ORDER BY e.timestamp DESC
                LIMIT $3
                """,
                user_id, entity_name.lower(), limit,
            )
            episodes = []
            for row in rows:
                entity_rows = await conn.fetch(
                    "SELECT * FROM episode_entities WHERE episode_id = $1",
                    row["id"],
                )
                episodes.append(_row_to_episode(row, entity_rows))
            return episodes

    async def get_emotion_timeline(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[Episode]:
        """Get episodes in a time range for emotion pattern analysis."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM episodes
                WHERE user_id = $1
                  AND timestamp >= $2
                  AND timestamp <= $3
                  AND is_compressed = FALSE
                ORDER BY timestamp ASC
                """,
                user_id, start, end,
            )
            episodes = []
            for row in rows:
                entity_rows = await conn.fetch(
                    "SELECT * FROM episode_entities WHERE episode_id = $1",
                    row["id"],
                )
                episodes.append(_row_to_episode(row, entity_rows))
            return episodes

    async def mark_compressed(
        self, episode_ids: list[str], summary_id: str
    ) -> int:
        """Mark episodes as compressed into a summary. Returns count updated."""
        if not episode_ids:
            return 0
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE episodes
                SET is_compressed = TRUE, compressed_into_id = $1
                WHERE id = ANY($2) AND is_compressed = FALSE
                """,
                summary_id, episode_ids,
            )
            # asyncpg returns "UPDATE N"
            return int(result.split()[-1])

    async def store_summary(self, summary: EpisodeSummary) -> EpisodeSummary:
        """Store an episode summary."""
        dominant_emotions_json = json.dumps([
            {
                "primary": e.primary,
                "intensity": e.intensity,
                "valence": e.valence,
                "arousal": e.arousal,
            }
            for e in summary.dominant_emotions
        ])
        embedding_val = summary.embedding if summary.embedding else None

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episode_summaries (
                    id, user_id, created_at, period_start, period_end,
                    source_episode_ids, episode_count, summary_text,
                    dominant_emotions, entity_mentions,
                    intent_distribution, behavioral_signals, embedding
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8,
                    $9::jsonb, $10::jsonb,
                    $11::jsonb, $12::jsonb, $13
                )
                """,
                summary.id, summary.user_id, summary.created_at,
                summary.period_start, summary.period_end,
                summary.source_episode_ids, summary.episode_count,
                summary.summary_text,
                dominant_emotions_json,
                json.dumps(summary.entity_mentions),
                json.dumps(summary.intent_distribution),
                json.dumps(summary.behavioral_signals),
                embedding_val,
            )
        return summary

    async def get_summaries(
        self,
        user_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EpisodeSummary]:
        """Retrieve summaries for a time range."""
        conditions = ["user_id = $1"]
        params: list[Any] = [user_id]
        idx = 2

        if start:
            conditions.append(f"period_end >= ${idx}")
            params.append(start)
            idx += 1
        if end:
            conditions.append(f"period_start <= ${idx}")
            params.append(end)
            idx += 1

        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM episode_summaries WHERE {where} ORDER BY period_start DESC",
                *params,
            )
            return [_row_to_summary(row) for row in rows]
