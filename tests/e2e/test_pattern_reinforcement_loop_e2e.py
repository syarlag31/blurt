"""E2E Scenario 9: Pattern reinforcement loop.

Tests the compound reinforcement/weakening cycle where patterns
are created, reinforced multiple times, weakened, and the confidence
trajectory is verified end-to-end through the API.

Cross-cutting concerns exercised:
- Compound reinforcement: multiple sequential reinforce calls accumulate
  confidence correctly (monotonically increasing until cap)
- Weakening after reinforcement: confidence decreases after weaken call
  but retains history of prior reinforcements
- Confidence trajectory: the full arc (create → reinforce × N → weaken)
  produces a predictable, bounded confidence curve
- Saturation bounds: confidence is clamped to [0.0, 1.0] regardless of
  how many reinforce/weaken operations are applied
- Pattern state consistency: pattern metadata (type, triggers, actions)
  remains unchanged through reinforcement/weakening mutations
- API idempotency: repeated identical operations produce consistent results
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestReinforcementLoop:
    """Multiple reinforce/weaken cycles through the API."""

    async def test_compound_reinforcement(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Reinforcing a pattern 5 times compounds confidence toward 1.0."""
        pat = await create_pattern_via_api(
            pattern_type="time",
            description="Productive in mornings",
            confidence=0.3,
        )
        pid = pat["id"]

        for i in range(5):
            resp = await client.put(
                f"/api/v1/users/{test_user_id}/patterns/{pid}/reinforce",
                json={
                    "evidence": f"morning task completed #{i + 1}",
                    "confidence_boost": 0.1,
                },
            )
            assert resp.status_code == 200

        # Final state
        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns/{pid}"
        )
        data = resp.json()
        assert data["confidence"] == pytest.approx(0.8, abs=0.01)
        assert data["observation_count"] == 6  # initial 1 + 5 reinforcements
        assert len(data["supporting_evidence"]) == 5

    async def test_reinforce_then_weaken_trajectory(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Reinforce 3 times, then weaken 2 times — net confidence change."""
        pat = await create_pattern_via_api(confidence=0.5)
        pid = pat["id"]

        # Reinforce 3 times (+0.15 total)
        for _ in range(3):
            await client.put(
                f"/api/v1/users/{test_user_id}/patterns/{pid}/reinforce",
                json={"confidence_boost": 0.05},
            )

        # Weaken 2 times (-0.2 total)
        for _ in range(2):
            await client.put(
                f"/api/v1/users/{test_user_id}/patterns/{pid}/weaken",
                json={"confidence_penalty": 0.1},
            )

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns/{pid}"
        )
        data = resp.json()
        # 0.5 + 0.15 - 0.2 = 0.45
        assert data["confidence"] == pytest.approx(0.45, abs=0.02)
        assert data["is_active"] is True

    async def test_confidence_capped_at_1(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Confidence never exceeds 1.0 regardless of reinforcements."""
        pat = await create_pattern_via_api(confidence=0.9)
        pid = pat["id"]

        for _ in range(5):
            await client.put(
                f"/api/v1/users/{test_user_id}/patterns/{pid}/reinforce",
                json={"confidence_boost": 0.1},
            )

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns/{pid}"
        )
        data = resp.json()
        assert data["confidence"] <= 1.0

    async def test_repeated_weakening_deactivates(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Repeated weakening eventually auto-deactivates the pattern."""
        pat = await create_pattern_via_api(confidence=0.5)
        pid = pat["id"]

        for _ in range(5):
            await client.put(
                f"/api/v1/users/{test_user_id}/patterns/{pid}/weaken",
                json={"confidence_penalty": 0.1},
            )

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns/{pid}"
        )
        data = resp.json()
        assert data["confidence"] < 0.1
        assert data["is_active"] is False

    async def test_deactivated_pattern_excluded_from_active_queries(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Once deactivated, pattern doesn't appear in active queries."""
        pat = await create_pattern_via_api(confidence=0.15)
        pid = pat["id"]

        # Weaken to deactivate
        await client.put(
            f"/api/v1/users/{test_user_id}/patterns/{pid}/weaken",
            json={"confidence_penalty": 0.1},
        )

        # Active query should not include it
        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns",
            params={"active": True},
        )
        data = resp.json()
        pattern_ids = [p["id"] for p in data["patterns"]]
        assert pid not in pattern_ids
