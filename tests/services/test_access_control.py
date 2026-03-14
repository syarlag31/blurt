"""Tests for tier-based access control for QUESTION intent.

Validates:
- Tier capability definitions are correct
- Query type gating works per tier
- Question type auto-detection from natural language
- Free-tier response formatting (no source episodes, limited results)
- Premium-tier response formatting (full recall, confidence scores)
- Anti-shame upgrade hints (no guilt language)
- Graceful fallback when premium queries hit free tier
"""

from __future__ import annotations

import pytest

from blurt.services.access_control import (
    FREE_QUERY_TYPES,
    PREMIUM_QUERY_TYPES,
    QuestionQueryType,
    UserTier,
    classify_question_type,
    format_free_tier_response,
    format_premium_tier_response,
    gate_query_for_tier,
    get_capabilities,
    is_query_allowed,
)


# ---------------------------------------------------------------------------
# Tier capabilities
# ---------------------------------------------------------------------------


class TestTierCapabilities:
    def test_free_tier_no_semantic_search(self):
        caps = get_capabilities(UserTier.FREE)
        assert not caps.semantic_search
        assert not caps.graph_traversal
        assert not caps.neighborhood_search
        assert not caps.episodic_recall
        assert not caps.pattern_access

    def test_free_tier_limits(self):
        caps = get_capabilities(UserTier.FREE)
        assert caps.max_results == 10
        assert caps.max_graph_hops == 0
        assert caps.max_history_days == 30

    def test_free_tier_no_premium_response_fields(self):
        caps = get_capabilities(UserTier.FREE)
        assert not caps.include_source_episodes
        assert not caps.include_confidence_scores
        assert not caps.include_relationship_context

    def test_premium_tier_full_access(self):
        caps = get_capabilities(UserTier.PREMIUM)
        assert caps.semantic_search
        assert caps.graph_traversal
        assert caps.neighborhood_search
        assert caps.episodic_recall
        assert caps.pattern_access

    def test_premium_tier_generous_limits(self):
        caps = get_capabilities(UserTier.PREMIUM)
        assert caps.max_results == 100
        assert caps.max_graph_hops == 3
        assert caps.max_history_days >= 365

    def test_premium_tier_full_response_fields(self):
        caps = get_capabilities(UserTier.PREMIUM)
        assert caps.include_source_episodes
        assert caps.include_confidence_scores
        assert caps.include_relationship_context

    def test_team_tier_superset_of_premium(self):
        premium = get_capabilities(UserTier.PREMIUM)
        team = get_capabilities(UserTier.TEAM)
        assert team.max_results >= premium.max_results
        assert team.max_graph_hops >= premium.max_graph_hops
        assert team.semantic_search
        assert team.graph_traversal


# ---------------------------------------------------------------------------
# Query type gating
# ---------------------------------------------------------------------------


class TestQueryGating:
    """Test that query types are properly gated by tier."""

    @pytest.mark.parametrize("query_type", list(FREE_QUERY_TYPES))
    def test_free_tier_allows_structured_queries(self, query_type):
        assert is_query_allowed(query_type, UserTier.FREE)

    @pytest.mark.parametrize("query_type", [
        QuestionQueryType.SEMANTIC_RECALL,
        QuestionQueryType.GRAPH_QUERY,
        QuestionQueryType.TEMPORAL_RECALL,
        QuestionQueryType.PATTERN_QUERY,
        QuestionQueryType.NEIGHBORHOOD,
    ])
    def test_free_tier_blocks_premium_queries(self, query_type):
        assert not is_query_allowed(query_type, UserTier.FREE)

    @pytest.mark.parametrize("query_type", list(PREMIUM_QUERY_TYPES))
    def test_premium_tier_allows_all_queries(self, query_type):
        assert is_query_allowed(query_type, UserTier.PREMIUM)

    @pytest.mark.parametrize("query_type", list(PREMIUM_QUERY_TYPES))
    def test_team_tier_allows_all_queries(self, query_type):
        assert is_query_allowed(query_type, UserTier.TEAM)


class TestGateQueryForTier:
    """Test the gate_query_for_tier function with fallbacks."""

    def test_allowed_query_returns_true(self):
        allowed, fallback, msg = gate_query_for_tier(
            QuestionQueryType.ENTITY_LOOKUP, UserTier.FREE
        )
        assert allowed is True
        assert fallback is None
        assert msg is None

    def test_gated_query_returns_fallback(self):
        allowed, fallback, msg = gate_query_for_tier(
            QuestionQueryType.SEMANTIC_RECALL, UserTier.FREE
        )
        assert allowed is False
        assert fallback == QuestionQueryType.FACT_LOOKUP
        assert msg is not None

    def test_graph_query_falls_back_to_entity_lookup(self):
        _, fallback, _ = gate_query_for_tier(
            QuestionQueryType.GRAPH_QUERY, UserTier.FREE
        )
        assert fallback == QuestionQueryType.ENTITY_LOOKUP

    def test_temporal_recall_falls_back_to_recent_facts(self):
        _, fallback, _ = gate_query_for_tier(
            QuestionQueryType.TEMPORAL_RECALL, UserTier.FREE
        )
        assert fallback == QuestionQueryType.RECENT_FACTS

    def test_pattern_query_falls_back_to_recent_facts(self):
        _, fallback, _ = gate_query_for_tier(
            QuestionQueryType.PATTERN_QUERY, UserTier.FREE
        )
        assert fallback == QuestionQueryType.RECENT_FACTS

    def test_neighborhood_falls_back_to_entity_lookup(self):
        _, fallback, _ = gate_query_for_tier(
            QuestionQueryType.NEIGHBORHOOD, UserTier.FREE
        )
        assert fallback == QuestionQueryType.ENTITY_LOOKUP

    def test_premium_never_gated(self):
        for qt in QuestionQueryType:
            allowed, _, _ = gate_query_for_tier(qt, UserTier.PREMIUM)
            assert allowed is True

    def test_upgrade_message_anti_shame(self):
        """Upgrade messages must never contain shame language."""
        shame_words = [
            "overdue", "behind", "missed", "failed", "lazy",
            "you should", "you must", "hurry", "running out",
            "deadline", "penalty", "streak",
        ]
        for qt in QuestionQueryType:
            allowed, _, msg = gate_query_for_tier(qt, UserTier.FREE)
            if msg:
                msg_lower = msg.lower()
                for word in shame_words:
                    assert word not in msg_lower, (
                        f"Shame word '{word}' found in upgrade message: {msg}"
                    )


# ---------------------------------------------------------------------------
# Question type classification
# ---------------------------------------------------------------------------


class TestClassifyQuestionType:
    """Test auto-detection of question types from natural language."""

    def test_entity_lookup_who_is(self):
        assert classify_question_type("Who is Sarah?") == QuestionQueryType.ENTITY_LOOKUP

    def test_fact_lookup_what_is(self):
        assert classify_question_type("What is Sarah's role?") == QuestionQueryType.FACT_LOOKUP

    def test_count_query(self):
        assert classify_question_type("How many tasks do I have?") == QuestionQueryType.COUNT_QUERY

    def test_semantic_recall(self):
        assert classify_question_type("What did I say about the project?") == QuestionQueryType.SEMANTIC_RECALL

    def test_temporal_recall(self):
        assert classify_question_type("What was I thinking about last week?") == QuestionQueryType.TEMPORAL_RECALL

    def test_pattern_query(self):
        assert classify_question_type("When am I most productive?") == QuestionQueryType.PATTERN_QUERY

    def test_graph_query(self):
        assert classify_question_type("How are Sarah and the project connected?") == QuestionQueryType.GRAPH_QUERY

    def test_neighborhood_query(self):
        assert classify_question_type("Tell me about the Q2 project") == QuestionQueryType.NEIGHBORHOOD

    def test_short_query_defaults_to_entity(self):
        assert classify_question_type("Sarah") == QuestionQueryType.ENTITY_LOOKUP

    def test_did_i_ever_is_recall(self):
        assert classify_question_type("Did I ever finish that book?") == QuestionQueryType.SEMANTIC_RECALL

    def test_habit_is_pattern(self):
        assert classify_question_type("What do I tend to do in the mornings?") == QuestionQueryType.PATTERN_QUERY


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------


class TestFreeTierResponseFormatting:
    """Test that free-tier responses are properly limited."""

    def _sample_results(self, n: int) -> list[dict]:
        return [{"content": f"result-{i}", "name": f"Entity-{i}"} for i in range(n)]

    def test_limits_results_to_max(self):
        caps = get_capabilities(UserTier.FREE)
        results = self._sample_results(20)
        response = format_free_tier_response(
            results, "test query", QuestionQueryType.ENTITY_LOOKUP, 20, caps
        )
        assert response.result_count <= caps.max_results
        assert response.truncated is True
        assert response.total_available == 20

    def test_no_source_episodes(self):
        caps = get_capabilities(UserTier.FREE)
        response = format_free_tier_response(
            self._sample_results(3), "test", QuestionQueryType.ENTITY_LOOKUP, 3, caps
        )
        assert response.source_episodes == []
        assert response.confidence_scores == []
        assert response.relationship_context == []

    def test_tier_is_free(self):
        caps = get_capabilities(UserTier.FREE)
        response = format_free_tier_response(
            [], "test", QuestionQueryType.ENTITY_LOOKUP, 0, caps
        )
        assert response.tier == UserTier.FREE

    def test_not_truncated_when_under_limit(self):
        caps = get_capabilities(UserTier.FREE)
        response = format_free_tier_response(
            self._sample_results(5), "test", QuestionQueryType.ENTITY_LOOKUP, 5, caps
        )
        assert response.truncated is False
        assert response.upgrade_hint is None

    def test_upgrade_hint_when_truncated(self):
        caps = get_capabilities(UserTier.FREE)
        response = format_free_tier_response(
            self._sample_results(20), "test", QuestionQueryType.ENTITY_LOOKUP, 20, caps
        )
        assert response.upgrade_hint is not None
        assert "Premium" in response.upgrade_hint

    def test_empty_results_have_helpful_summary(self):
        caps = get_capabilities(UserTier.FREE)
        response = format_free_tier_response(
            [], "test", QuestionQueryType.ENTITY_LOOKUP, 0, caps
        )
        assert response.answer_summary != ""
        assert "don't have" in response.answer_summary.lower() or "no" in response.answer_summary.lower()

    def test_entity_lookup_summary(self):
        caps = get_capabilities(UserTier.FREE)
        results = [{"name": "Sarah", "content": "person: Sarah"}]
        response = format_free_tier_response(
            results, "Who is Sarah?", QuestionQueryType.ENTITY_LOOKUP, 1, caps
        )
        assert "Sarah" in response.answer_summary

    def test_count_query_summary(self):
        caps = get_capabilities(UserTier.FREE)
        results = [{"content": "5 items"}]
        response = format_free_tier_response(
            results, "How many?", QuestionQueryType.COUNT_QUERY, 5, caps
        )
        assert "5" in response.answer_summary


class TestPremiumTierResponseFormatting:
    """Test that premium-tier responses include full detail."""

    def _sample_results(self, n: int) -> list[dict]:
        return [{"content": f"result-{i}"} for i in range(n)]

    def test_includes_source_episodes(self):
        caps = get_capabilities(UserTier.PREMIUM)
        episodes = ["ep-1", "ep-2"]
        response = format_premium_tier_response(
            self._sample_results(3), "test", QuestionQueryType.SEMANTIC_RECALL,
            3, caps, source_episodes=episodes,
        )
        assert response.source_episodes == episodes

    def test_includes_confidence_scores(self):
        caps = get_capabilities(UserTier.PREMIUM)
        scores = [0.95, 0.87, 0.72]
        response = format_premium_tier_response(
            self._sample_results(3), "test", QuestionQueryType.SEMANTIC_RECALL,
            3, caps, confidence_scores=scores,
        )
        assert response.confidence_scores == scores

    def test_includes_relationship_context(self):
        caps = get_capabilities(UserTier.PREMIUM)
        ctx = [{"type": "works_with", "strength": 5.0}]
        response = format_premium_tier_response(
            self._sample_results(1), "test", QuestionQueryType.GRAPH_QUERY,
            1, caps, relationship_context=ctx,
        )
        assert response.relationship_context == ctx

    def test_tier_is_premium(self):
        caps = get_capabilities(UserTier.PREMIUM)
        response = format_premium_tier_response(
            [], "test", QuestionQueryType.SEMANTIC_RECALL, 0, caps,
        )
        assert response.tier == UserTier.PREMIUM

    def test_no_upgrade_hint(self):
        caps = get_capabilities(UserTier.PREMIUM)
        response = format_premium_tier_response(
            self._sample_results(3), "test", QuestionQueryType.SEMANTIC_RECALL,
            3, caps,
        )
        assert response.upgrade_hint is None

    def test_generous_result_limit(self):
        caps = get_capabilities(UserTier.PREMIUM)
        results = self._sample_results(50)
        response = format_premium_tier_response(
            results, "test", QuestionQueryType.SEMANTIC_RECALL, 50, caps,
        )
        assert response.result_count == 50

    def test_semantic_recall_summary_mentions_relevance(self):
        caps = get_capabilities(UserTier.PREMIUM)
        scores = [0.9, 0.8]
        response = format_premium_tier_response(
            self._sample_results(2), "test", QuestionQueryType.SEMANTIC_RECALL,
            2, caps, confidence_scores=scores,
        )
        assert "relevance" in response.answer_summary.lower() or "memories" in response.answer_summary.lower()

    def test_pattern_query_summary(self):
        caps = get_capabilities(UserTier.PREMIUM)
        response = format_premium_tier_response(
            self._sample_results(2), "test", QuestionQueryType.PATTERN_QUERY,
            2, caps,
        )
        assert "pattern" in response.answer_summary.lower()


# ---------------------------------------------------------------------------
# Anti-shame validation
# ---------------------------------------------------------------------------


class TestAntiShameDesign:
    """Ensure all user-facing messages follow anti-shame design principles."""

    SHAME_WORDS = [
        "overdue", "behind", "missed", "failed", "lazy", "guilty",
        "you should have", "you must", "hurry", "running out",
        "deadline", "penalty", "streak", "slacking", "falling behind",
    ]

    def _check_no_shame(self, text: str):
        text_lower = text.lower()
        for word in self.SHAME_WORDS:
            assert word not in text_lower, f"Shame word '{word}' in: {text}"

    def test_free_tier_empty_summary_no_shame(self):
        caps = get_capabilities(UserTier.FREE)
        response = format_free_tier_response(
            [], "test", QuestionQueryType.ENTITY_LOOKUP, 0, caps
        )
        self._check_no_shame(response.answer_summary)

    def test_premium_empty_summary_no_shame(self):
        caps = get_capabilities(UserTier.PREMIUM)
        response = format_premium_tier_response(
            [], "test", QuestionQueryType.SEMANTIC_RECALL, 0, caps
        )
        self._check_no_shame(response.answer_summary)

    def test_upgrade_hints_no_shame(self):
        caps = get_capabilities(UserTier.FREE)
        results = [{"content": f"r-{i}"} for i in range(20)]
        response = format_free_tier_response(
            results, "test", QuestionQueryType.ENTITY_LOOKUP, 20, caps
        )
        if response.upgrade_hint:
            self._check_no_shame(response.upgrade_hint)

    def test_all_gate_messages_no_shame(self):
        for qt in QuestionQueryType:
            _, _, msg = gate_query_for_tier(qt, UserTier.FREE)
            if msg:
                self._check_no_shame(msg)
