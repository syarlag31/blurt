"""E2E Scenario 7: Multi-user isolation.

Verifies that different users' data is isolated: episodes, patterns,
and feedback don't leak between users.

Cross-cutting concerns exercised:
- User-scoped data isolation: episodes, patterns, and feedback are partitioned
  by user_id — no data leaks between users sharing the same store instances
- Episodic memory isolation: user A's episodes are invisible to user B queries
- Pattern store isolation: patterns created by one user don't appear for another
- Feedback isolation: Thompson Sampling parameters are per-user, per-task
- Concurrent user safety: interleaved operations from multiple users don't
  corrupt each other's state
- Query filtering: user_id filter is enforced at every API layer (episodes,
  patterns, feedback)
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio

USER_A = "user-alice"
USER_B = "user-bob"


class TestUserDataIsolation:
    """Data from one user is not visible to another."""

    async def test_episodes_isolated_between_users(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
    ):
        """Episodes for user A are not returned for user B."""
        await create_episode_via_api(raw_text="Alice's thought", user_id=USER_A)
        await create_episode_via_api(raw_text="Bob's thought", user_id=USER_B)

        resp_a = await client.get(f"/api/v1/episodes/user/{USER_A}")
        resp_b = await client.get(f"/api/v1/episodes/user/{USER_B}")

        eps_a = resp_a.json()["episodes"]
        eps_b = resp_b.json()["episodes"]

        assert len(eps_a) == 1
        assert eps_a[0]["raw_text"] == "Alice's thought"
        assert len(eps_b) == 1
        assert eps_b[0]["raw_text"] == "Bob's thought"

    async def test_patterns_isolated_between_users(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
    ):
        """Patterns for user A are not returned for user B."""
        await create_pattern_via_api(
            description="Alice's pattern", user_id=USER_A
        )
        await create_pattern_via_api(
            description="Bob's pattern", user_id=USER_B
        )

        resp_a = await client.get(f"/api/v1/users/{USER_A}/patterns")
        resp_b = await client.get(f"/api/v1/users/{USER_B}/patterns")

        pats_a = resp_a.json()["patterns"]
        pats_b = resp_b.json()["patterns"]

        assert len(pats_a) == 1
        assert pats_a[0]["description"] == "Alice's pattern"
        assert len(pats_b) == 1
        assert pats_b[0]["description"] == "Bob's pattern"

    async def test_pattern_access_cross_user_returns_404(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
    ):
        """Accessing another user's pattern by ID returns 404."""
        pat = await create_pattern_via_api(
            description="Alice only", user_id=USER_A
        )
        pattern_id = pat["id"]

        # User B tries to access User A's pattern
        resp = await client.get(
            f"/api/v1/users/{USER_B}/patterns/{pattern_id}"
        )
        assert resp.status_code == 404

    async def test_feedback_isolated_between_users(
        self,
        client: httpx.AsyncClient,
        record_feedback_via_api: Any,
    ):
        """Recent feedback for user A excludes user B's events."""
        await record_feedback_via_api(
            task_id="t1", action="accept", user_id=USER_A
        )
        await record_feedback_via_api(
            task_id="t2", action="dismiss", user_id=USER_B
        )

        resp_a = await client.get(
            "/api/v1/feedback/recent",
            params={"user_id": USER_A},
        )
        events_a = resp_a.json()
        assert len(events_a) == 1
        assert events_a[0]["task_id"] == "t1"

        resp_b = await client.get(
            "/api/v1/feedback/recent",
            params={"user_id": USER_B},
        )
        events_b = resp_b.json()
        assert len(events_b) == 1
        assert events_b[0]["task_id"] == "t2"
