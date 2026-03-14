"""Tests for the user rhythms API endpoint.

Validates:
- GET /api/v1/users/{user_id}/rhythms returns detected patterns with confidence/evidence
- Filtering by rhythm_type, day, time, min_confidence
- GET /current returns rhythms for the current moment
- POST /analyze triggers fresh analysis
- POST /sync-graph creates temporal context nodes in knowledge graph
- GET /heatmap returns weekly heatmap
- Anti-shame: descriptions are neutral, never guilt-inducing
- Knowledge graph integration creates temporal entities and facts
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from blurt.api.rhythms import (
    _build_summary,
    _rhythm_to_response,
    _sync_rhythm_to_graph,
    set_graph_store,
    set_pattern_service,
    set_rhythm_service,
)
from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    FactType,
    LearnedPattern,
    PatternType,
)
from blurt.services.patterns import PatternService
from blurt.services.rhythm import (
    DetectedRhythm,
    RhythmAnalysisResult,
    RhythmDetectionService,
    RhythmType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_id() -> str:
    return f"test-user-{uuid.uuid4().hex[:8]}"


def _make_rhythm(
    rhythm_type: RhythmType = RhythmType.ENERGY_CRASH,
    day: str = "thursday",
    time: str = "afternoon",
    z_score: float = 2.1,
    metric_value: float = 0.15,
    metric_mean: float = 0.5,
    metric_std: float = 0.17,
    observation_count: int = 12,
    confidence: float = 0.72,
    evidence: list[str] | None = None,
) -> DetectedRhythm:
    return DetectedRhythm(
        rhythm_type=rhythm_type,
        day_of_week=day,
        time_of_day=time,
        z_score=z_score,
        metric_value=metric_value,
        metric_mean=metric_mean,
        metric_std=metric_std,
        observation_count=observation_count,
        confidence=confidence,
        evidence_episode_ids=evidence or ["ep-1", "ep-2", "ep-3"],
    )


def _make_analysis_result(
    user_id: str,
    rhythms: list[DetectedRhythm] | None = None,
    total_episodes: int = 50,
) -> RhythmAnalysisResult:
    now = datetime.now(timezone.utc)
    return RhythmAnalysisResult(
        user_id=user_id,
        analysis_period_start=now - timedelta(weeks=4),
        analysis_period_end=now,
        total_episodes_analyzed=total_episodes,
        rhythms=rhythms or [
            _make_rhythm(RhythmType.ENERGY_CRASH, "thursday", "afternoon", confidence=0.72),
            _make_rhythm(RhythmType.CREATIVITY_PEAK, "tuesday", "morning", confidence=0.65),
            _make_rhythm(RhythmType.PRODUCTIVITY_WINDOW, "monday", "morning", confidence=0.80),
            _make_rhythm(RhythmType.MOOD_LOW, "friday", "evening", confidence=0.45),
        ],
        bucket_stats={
            "thursday:afternoon": {
                "observation_count": 12,
                "mean_valence": -0.3,
                "mean_arousal": 0.3,
                "energy_score": 0.15,
                "completion_rate": 0.2,
                "creativity_ratio": 0.1,
            },
            "tuesday:morning": {
                "observation_count": 15,
                "mean_valence": 0.6,
                "mean_arousal": 0.7,
                "energy_score": 0.8,
                "completion_rate": 0.5,
                "creativity_ratio": 0.6,
            },
        },
    )


@pytest.fixture
def mock_rhythm_service(user_id: str) -> RhythmDetectionService:
    """Create a mock rhythm detection service."""
    service = AsyncMock(spec=RhythmDetectionService)
    service.analyze_user_rhythms.return_value = _make_analysis_result(user_id)
    service.get_current_rhythm_context.return_value = {
        "day_of_week": "thursday",
        "time_of_day": "afternoon",
        "active_rhythms": [
            {
                "type": "energy_crash",
                "confidence": 0.72,
                "description": "Energy tends to drop on Thursday afternoon",
                "is_periodic": True,
                "trend": "stable",
            }
        ],
        "bucket_stats": {"observation_count": 12, "mean_valence": -0.3},
        "recommendations": [
            "This is typically a lower-energy time — lighter tasks or a break might feel right"
        ],
    }
    return service


@pytest.fixture
def mock_pattern_service() -> PatternService:
    """Create a mock pattern service."""
    service = AsyncMock(spec=PatternService)
    service.create_pattern.return_value = LearnedPattern(
        user_id="test",
        pattern_type=PatternType.ENERGY_RHYTHM,
        description="Energy tends to drop on Thursday afternoon",
        confidence=0.72,
    )
    return service


@pytest.fixture
def mock_graph_store() -> AsyncMock:
    """Create a mock graph store."""
    store = AsyncMock()
    store.user_id = "test"

    # find_entity_by_name returns None first time (to trigger creation)
    store.find_entity_by_name.return_value = None

    # add_entity creates a new entity
    entity = EntityNode(
        user_id="test",
        name="temporal:thursday:afternoon",
        entity_type=EntityType.TOPIC,
        aliases=["Thursday afternoon"],
        attributes={"temporal_context": True},
    )
    store.add_entity.return_value = entity

    # add_fact returns a fact
    fact = Fact(
        user_id="test",
        content="Energy tends to drop on Thursday afternoon",
        fact_type=FactType.ATTRIBUTE,
        subject_entity_id=entity.id,
        confidence=0.72,
    )
    store.add_fact.return_value = fact

    # add_pattern returns a learned pattern
    store.add_pattern.return_value = LearnedPattern(
        user_id="test",
        pattern_type=PatternType.ENERGY_RHYTHM,
        description="Energy tends to drop on Thursday afternoon",
    )

    return store


@pytest.fixture
def app(
    mock_rhythm_service: RhythmDetectionService,
    mock_pattern_service: PatternService,
    mock_graph_store: AsyncMock,
):
    """Create a test FastAPI app with all dependencies injected."""
    from blurt.api.rhythms import router
    from fastapi import FastAPI

    test_app = FastAPI()
    test_app.include_router(router)

    set_rhythm_service(mock_rhythm_service)
    set_pattern_service(mock_pattern_service)
    set_graph_store(mock_graph_store)

    yield test_app

    # Cleanup
    set_rhythm_service(None)
    set_pattern_service(None)
    set_graph_store(None)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test: GET /api/v1/users/{user_id}/rhythms
# ---------------------------------------------------------------------------


class TestGetUserRhythms:
    """Tests for the main rhythms endpoint."""

    def test_returns_detected_rhythms(self, client: TestClient, user_id: str):
        """All detected rhythms are returned with proper fields."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        assert resp.status_code == 200
        data = resp.json()

        assert data["user_id"] == user_id
        assert data["total_episodes_analyzed"] == 50
        assert len(data["rhythms"]) == 4
        assert data["analysis_period_start"]
        assert data["analysis_period_end"]

    def test_rhythm_has_confidence_and_evidence(self, client: TestClient, user_id: str):
        """Each rhythm has confidence score and supporting evidence."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        data = resp.json()

        rhythm = data["rhythms"][0]
        assert "confidence" in rhythm
        assert 0.0 <= rhythm["confidence"] <= 1.0
        assert "supporting_evidence" in rhythm
        assert isinstance(rhythm["supporting_evidence"], list)
        assert "z_score" in rhythm
        assert rhythm["z_score"] > 0

    def test_rhythm_has_description(self, client: TestClient, user_id: str):
        """Each rhythm has a human-readable description."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        data = resp.json()

        for rhythm in data["rhythms"]:
            assert rhythm["description"]
            assert isinstance(rhythm["description"], str)
            assert len(rhythm["description"]) > 10

    def test_rhythm_has_pattern_type(self, client: TestClient, user_id: str):
        """Each rhythm has a mapped pattern type for knowledge graph storage."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        data = resp.json()

        for rhythm in data["rhythms"]:
            assert "pattern_type" in rhythm

    def test_filter_by_rhythm_type(self, client: TestClient, user_id: str):
        """Filtering by rhythm_type returns only matching rhythms."""
        resp = client.get(
            f"/api/v1/users/{user_id}/rhythms",
            params={"rhythm_type": "energy_crash"},
        )
        data = resp.json()
        assert all(r["rhythm_type"] == "energy_crash" for r in data["rhythms"])

    def test_filter_by_day(self, client: TestClient, user_id: str):
        """Filtering by day returns only rhythms for that day."""
        resp = client.get(
            f"/api/v1/users/{user_id}/rhythms",
            params={"day": "thursday"},
        )
        data = resp.json()
        assert all(r["day_of_week"] == "thursday" for r in data["rhythms"])

    def test_filter_by_time(self, client: TestClient, user_id: str):
        """Filtering by time returns only rhythms for that time."""
        resp = client.get(
            f"/api/v1/users/{user_id}/rhythms",
            params={"time": "morning"},
        )
        data = resp.json()
        assert all(r["time_of_day"] == "morning" for r in data["rhythms"])

    def test_filter_by_min_confidence(self, client: TestClient, user_id: str):
        """Filtering by min_confidence excludes low-confidence rhythms."""
        resp = client.get(
            f"/api/v1/users/{user_id}/rhythms",
            params={"min_confidence": 0.7},
        )
        data = resp.json()
        assert all(r["confidence"] >= 0.7 for r in data["rhythms"])

    def test_summary_counts(self, client: TestClient, user_id: str):
        """Summary includes counts by rhythm category."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        data = resp.json()

        summary = data["summary"]
        assert "energy_crashes" in summary
        assert "creativity_peaks" in summary
        assert "productivity_windows" in summary
        assert summary["energy_crashes"] >= 0

    def test_recommendations_present(self, client: TestClient, user_id: str):
        """Recommendations are included in the response."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        data = resp.json()

        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_anti_shame_descriptions(self, client: TestClient, user_id: str):
        """Descriptions use neutral, shame-free language."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        data = resp.json()

        shame_words = {"failed", "lazy", "overdue", "behind", "missed", "guilt", "streak"}
        for rhythm in data["rhythms"]:
            desc_lower = rhythm["description"].lower()
            for word in shame_words:
                assert word not in desc_lower, (
                    f"Shame word '{word}' found in description: {rhythm['description']}"
                )

    def test_empty_rhythms_when_no_data(
        self, client: TestClient, user_id: str,
        mock_rhythm_service: AsyncMock,
    ):
        """Returns empty list when no rhythms detected (via min_confidence filter)."""
        # Use the filter approach: set confidence above all rhythms
        resp = client.get(
            f"/api/v1/users/{user_id}/rhythms",
            params={"min_confidence": 0.99},
        )
        data = resp.json()

        # All test rhythms have confidence < 0.99, so none pass the filter
        assert data["rhythms"] == []
        assert data["total_episodes_analyzed"] == 50


# ---------------------------------------------------------------------------
# Test: GET /api/v1/users/{user_id}/rhythms/current
# ---------------------------------------------------------------------------


class TestGetCurrentRhythms:
    """Tests for the current-moment rhythm context endpoint."""

    def test_returns_current_context(self, client: TestClient, user_id: str):
        """Returns rhythm context for the current time slot."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms/current")
        assert resp.status_code == 200
        data = resp.json()

        assert "day_of_week" in data
        assert "time_of_day" in data
        assert "active_rhythms" in data
        assert "recommendations" in data

    def test_active_rhythms_have_type_and_confidence(
        self, client: TestClient, user_id: str,
    ):
        """Active rhythms include type and confidence."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms/current")
        data = resp.json()

        for rhythm in data["active_rhythms"]:
            assert "rhythm_type" in rhythm
            assert "confidence" in rhythm
            assert "description" in rhythm

    def test_recommendations_are_shame_free(self, client: TestClient, user_id: str):
        """Current recommendations use shame-free language."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms/current")
        data = resp.json()

        for rec in data["recommendations"]:
            rec_lower = rec.lower()
            assert "overdue" not in rec_lower
            assert "failed" not in rec_lower
            assert "lazy" not in rec_lower


# ---------------------------------------------------------------------------
# Test: POST /api/v1/users/{user_id}/rhythms/analyze
# ---------------------------------------------------------------------------


class TestAnalyzeRhythms:
    """Tests for the fresh analysis trigger endpoint."""

    def test_analyze_returns_rhythms(self, client: TestClient, user_id: str):
        """Analyze returns a full rhythm analysis result."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/analyze",
            json={"lookback_weeks": 4},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["user_id"] == user_id
        assert len(data["rhythms"]) >= 0
        assert "summary" in data

    def test_analyze_custom_lookback(
        self, client: TestClient, user_id: str, mock_rhythm_service: AsyncMock,
    ):
        """Custom lookback_weeks is passed to the service."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/analyze",
            json={"lookback_weeks": 8},
        )
        assert resp.status_code == 200
        # Verify service was called
        mock_rhythm_service.analyze_user_rhythms.assert_called()


# ---------------------------------------------------------------------------
# Test: POST /api/v1/users/{user_id}/rhythms/sync-graph
# ---------------------------------------------------------------------------


class TestSyncRhythmsToGraph:
    """Tests for syncing rhythms into the knowledge graph."""

    def test_sync_creates_patterns(
        self, client: TestClient, user_id: str,
        mock_pattern_service: AsyncMock,
    ):
        """Sync creates LearnedPattern entries for each qualified rhythm."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["patterns_synced"] == 4  # All 4 rhythms
        assert mock_pattern_service.create_pattern.call_count == 4

    def test_sync_filters_by_min_confidence(
        self, client: TestClient, user_id: str,
        mock_pattern_service: AsyncMock,
    ):
        """Only rhythms above min_confidence are synced."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.7},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Only rhythms with confidence >= 0.7 (energy_crash=0.72, productivity=0.80)
        assert data["patterns_synced"] == 2

    def test_sync_creates_temporal_nodes(
        self, client: TestClient, user_id: str,
        mock_graph_store: AsyncMock,
    ):
        """Sync creates temporal context entities in the knowledge graph."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["temporal_nodes_created"] >= 1
        # Verify graph store was called to create entities
        mock_graph_store.add_entity.assert_called()

    def test_sync_creates_facts(
        self, client: TestClient, user_id: str,
        mock_graph_store: AsyncMock,
    ):
        """Sync creates facts describing rhythm observations."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["facts_created"] >= 1
        mock_graph_store.add_fact.assert_called()

    def test_sync_details_contain_pattern_ids(
        self, client: TestClient, user_id: str,
    ):
        """Sync response details include pattern IDs and entity IDs."""
        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.0},
        )
        data = resp.json()

        for detail in data["details"]:
            assert "rhythm_type" in detail
            assert "pattern_id" in detail
            assert "confidence" in detail

    def test_sync_reuses_existing_temporal_entity(
        self, client: TestClient, user_id: str,
        mock_graph_store: AsyncMock,
    ):
        """When a temporal entity already exists, sync reuses it."""
        existing_entity = EntityNode(
            user_id="test",
            name="temporal:thursday:afternoon",
            entity_type=EntityType.TOPIC,
        )
        mock_graph_store.find_entity_by_name.return_value = existing_entity

        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.7},
        )
        assert resp.status_code == 200
        data = resp.json()

        # No new nodes created since entity already exists
        assert data["temporal_nodes_created"] == 0

    def test_sync_without_graph_store(
        self, client: TestClient, user_id: str,
        mock_pattern_service: AsyncMock,
    ):
        """Sync works without a graph store — only creates patterns."""
        set_graph_store(None)

        resp = client.post(
            f"/api/v1/users/{user_id}/rhythms/sync-graph",
            json={"lookback_weeks": 4, "min_confidence": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["patterns_synced"] == 4
        assert data["temporal_nodes_created"] == 0
        assert data["facts_created"] == 0


# ---------------------------------------------------------------------------
# Test: GET /api/v1/users/{user_id}/rhythms/heatmap
# ---------------------------------------------------------------------------


class TestRhythmHeatmap:
    """Tests for the weekly heatmap endpoint."""

    def test_heatmap_returns_28_cells(self, client: TestClient, user_id: str):
        """Heatmap always returns 28 cells (7 days x 4 time periods)."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms/heatmap")
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["cells"]) == 28
        assert data["user_id"] == user_id

    def test_heatmap_cells_have_required_fields(self, client: TestClient, user_id: str):
        """Each heatmap cell has required metric fields."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms/heatmap")
        data = resp.json()

        for cell in data["cells"]:
            assert "day_of_week" in cell
            assert "time_of_day" in cell
            assert "observation_count" in cell
            assert "energy_score" in cell
            assert "active_rhythms" in cell

    def test_heatmap_active_rhythms_populated(self, client: TestClient, user_id: str):
        """Cells with detected rhythms have active_rhythms populated."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms/heatmap")
        data = resp.json()

        # Find the Thursday afternoon cell
        thursday_afternoon = [
            c for c in data["cells"]
            if c["day_of_week"] == "thursday" and c["time_of_day"] == "afternoon"
        ]
        assert len(thursday_afternoon) == 1
        assert "energy_crash" in thursday_afternoon[0]["active_rhythms"]


# ---------------------------------------------------------------------------
# Test: Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal conversion and summary helpers."""

    def test_rhythm_to_response_conversion(self):
        """DetectedRhythm converts to response model correctly."""
        rhythm = _make_rhythm()
        response = _rhythm_to_response(rhythm)

        assert response.rhythm_type == "energy_crash"
        assert response.day_of_week == "thursday"
        assert response.time_of_day == "afternoon"
        assert response.confidence == 0.72
        assert response.observation_count == 12
        assert len(response.supporting_evidence) == 3
        assert response.is_periodic is False
        assert response.trend == "stable"
        assert response.weeks_observed == 0

    def test_rhythm_to_response_with_periodicity(self):
        """DetectedRhythm with periodicity data converts correctly."""
        rhythm = DetectedRhythm(
            rhythm_type=RhythmType.ENERGY_CRASH,
            day_of_week="thursday",
            time_of_day="afternoon",
            z_score=2.1,
            metric_value=0.15,
            metric_mean=0.5,
            metric_std=0.17,
            observation_count=20,
            confidence=0.85,
            is_periodic=True,
            periodicity_strength=0.78,
            trend_direction=-0.15,
            weeks_observed=6,
        )
        response = _rhythm_to_response(rhythm)

        assert response.is_periodic is True
        assert response.periodicity_strength == 0.78
        assert response.trend == "down"
        assert response.weeks_observed == 6

    def test_build_summary_counts_all_types(self):
        """Summary correctly counts all rhythm types."""
        rhythms = [
            _make_rhythm(RhythmType.ENERGY_CRASH),
            _make_rhythm(RhythmType.ENERGY_PEAK),
            _make_rhythm(RhythmType.CREATIVITY_PEAK),
            _make_rhythm(RhythmType.PRODUCTIVITY_WINDOW),
            _make_rhythm(RhythmType.PRODUCTIVITY_DIP),
            _make_rhythm(RhythmType.MOOD_LOW),
            _make_rhythm(RhythmType.MOOD_HIGH),
        ]
        summary = _build_summary(rhythms)

        assert summary.energy_crashes == 1
        assert summary.energy_peaks == 1
        assert summary.creativity_peaks == 1
        assert summary.productivity_windows == 1
        assert summary.productivity_dips == 1
        assert summary.mood_lows == 1
        assert summary.mood_highs == 1

    def test_build_summary_empty(self):
        """Summary handles empty rhythm list."""
        summary = _build_summary([])
        assert summary.energy_crashes == 0
        assert summary.mood_highs == 0


# ---------------------------------------------------------------------------
# Test: Knowledge graph integration
# ---------------------------------------------------------------------------


class TestGraphIntegration:
    """Tests for syncing rhythms into the knowledge graph as temporal nodes."""

    @pytest.mark.asyncio
    async def test_sync_creates_temporal_entity(self, mock_graph_store: AsyncMock):
        """Sync creates a temporal context entity with correct attributes."""
        rhythm = _make_rhythm()
        result = await _sync_rhythm_to_graph(
            mock_graph_store, "user-1", rhythm, "pattern-123",
        )

        mock_graph_store.add_entity.assert_called_once()
        call_kwargs = mock_graph_store.add_entity.call_args
        assert call_kwargs.kwargs["name"] == "temporal:thursday:afternoon"
        assert call_kwargs.kwargs["entity_type"] == EntityType.TOPIC
        assert call_kwargs.kwargs["attributes"]["temporal_context"] is True
        assert call_kwargs.kwargs["attributes"]["day_of_week"] == "thursday"
        assert call_kwargs.kwargs["attributes"]["time_of_day"] == "afternoon"
        assert result["nodes_created"] == 1

    @pytest.mark.asyncio
    async def test_sync_adds_fact_to_entity(self, mock_graph_store: AsyncMock):
        """Sync adds a fact describing the rhythm to the temporal entity."""
        rhythm = _make_rhythm()
        result = await _sync_rhythm_to_graph(
            mock_graph_store, "user-1", rhythm, "pattern-123",
        )

        mock_graph_store.add_fact.assert_called_once()
        call_kwargs = mock_graph_store.add_fact.call_args
        assert "Energy tends to drop" in call_kwargs.kwargs["content"]
        assert "confidence: 0.72" in call_kwargs.kwargs["content"]
        assert call_kwargs.kwargs["fact_type"] == FactType.ATTRIBUTE
        assert call_kwargs.kwargs["confidence"] == 0.72
        assert result["facts_created"] == 1

    @pytest.mark.asyncio
    async def test_sync_adds_pattern_to_graph(self, mock_graph_store: AsyncMock):
        """Sync adds the rhythm as a pattern in the graph store."""
        rhythm = _make_rhythm()
        await _sync_rhythm_to_graph(
            mock_graph_store, "user-1", rhythm, "pattern-123",
        )

        mock_graph_store.add_pattern.assert_called_once()
        call_kwargs = mock_graph_store.add_pattern.call_args
        assert call_kwargs.kwargs["pattern_type"] == PatternType.ENERGY_RHYTHM
        assert call_kwargs.kwargs["description"] == rhythm.description
        assert call_kwargs.kwargs["confidence"] == rhythm.confidence

    @pytest.mark.asyncio
    async def test_sync_reuses_existing_entity(self, mock_graph_store: AsyncMock):
        """When temporal entity exists, it's reused (not duplicated)."""
        existing = EntityNode(
            user_id="user-1",
            name="temporal:thursday:afternoon",
            entity_type=EntityType.TOPIC,
        )
        mock_graph_store.find_entity_by_name.return_value = existing

        rhythm = _make_rhythm()
        result = await _sync_rhythm_to_graph(
            mock_graph_store, "user-1", rhythm, "pattern-123",
        )

        # Entity was not created (already existed)
        mock_graph_store.add_entity.assert_not_called()
        assert result["nodes_created"] == 0
        # But fact was still added
        assert result["facts_created"] == 1
        assert result["entity_id"] == existing.id

    @pytest.mark.asyncio
    async def test_sync_multiple_rhythms_same_slot(self, mock_graph_store: AsyncMock):
        """Multiple rhythms for the same time slot share one temporal entity."""
        mock_graph_store.find_entity_by_name.side_effect = [None, MagicMock(id="existing-id")]

        entity = EntityNode(
            user_id="user-1",
            name="temporal:thursday:afternoon",
            entity_type=EntityType.TOPIC,
        )
        mock_graph_store.add_entity.return_value = entity

        r1 = _make_rhythm(RhythmType.ENERGY_CRASH, "thursday", "afternoon")
        r2 = _make_rhythm(RhythmType.MOOD_LOW, "thursday", "afternoon")

        result1 = await _sync_rhythm_to_graph(mock_graph_store, "user-1", r1, "p1")
        result2 = await _sync_rhythm_to_graph(mock_graph_store, "user-1", r2, "p2")

        # First creates entity, second reuses
        assert result1["nodes_created"] == 1
        assert result2["nodes_created"] == 0

    @pytest.mark.asyncio
    async def test_temporal_entity_has_aliases(self, mock_graph_store: AsyncMock):
        """Temporal entity includes human-readable aliases."""
        rhythm = _make_rhythm(day="tuesday", time="morning")
        await _sync_rhythm_to_graph(
            mock_graph_store, "user-1", rhythm, "pattern-123",
        )

        call_kwargs = mock_graph_store.add_entity.call_args
        aliases = call_kwargs.kwargs["aliases"]
        assert "Tuesday morning" in aliases
        assert "tuesday morning" in aliases


# ---------------------------------------------------------------------------
# Test: Anti-shame design
# ---------------------------------------------------------------------------


class TestAntiShameDesign:
    """Verify anti-shame principles in rhythm responses."""

    def test_no_streak_language(self, client: TestClient, user_id: str):
        """No streak or streak-breaking language in responses."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        text = resp.text.lower()
        assert "streak" not in text
        assert "consecutive" not in text

    def test_no_guilt_language(self, client: TestClient, user_id: str):
        """No guilt-inducing language in responses."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        text = resp.text.lower()
        assert "should have" not in text
        assert "you need to" not in text
        assert "falling behind" not in text

    def test_no_overdue_counters(self, client: TestClient, user_id: str):
        """No overdue counters or pending task pressure."""
        resp = client.get(f"/api/v1/users/{user_id}/rhythms")
        text = resp.text.lower()
        assert "overdue" not in text
        assert "pending" not in text

    def test_descriptions_use_tends_to_language(self):
        """Rhythm descriptions use 'tends to' not 'always' or 'you fail to'."""
        rhythm = _make_rhythm(RhythmType.ENERGY_CRASH)
        response = _rhythm_to_response(rhythm)
        assert "tends to" in response.description.lower()
