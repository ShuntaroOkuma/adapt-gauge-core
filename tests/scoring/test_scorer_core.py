"""
Tests for the core scorer dispatcher (scorer.py)

Verifies:
- Standard scoring methods work (exact_match, contains, f1)
- llm_judge works with a mock grader
- custom:* raises ValueError in adapt-gauge-core
"""

import pytest
from unittest.mock import MagicMock

from adapt_gauge_core.scoring.scorer import score, _FALLBACK_SCORERS
from adapt_gauge_core.domain.value_objects import ScoringResult, ModelResponse


class TestStandardScoring:
    """Standard scoring methods"""

    def test_exact_match_hit(self):
        result = score("hello world", "Hello World", "exact_match")
        assert result.score == 1.0

    def test_exact_match_miss(self):
        result = score("hello", "world", "exact_match")
        assert result.score == 0.0

    def test_contains_hit(self):
        result = score("hello", "say hello world", "contains")
        assert result.score == 1.0

    def test_contains_miss(self):
        result = score("goodbye", "hello world", "contains")
        assert result.score == 0.0

    def test_f1_identical(self):
        result = score("the quick brown fox", "the quick brown fox", "f1")
        assert result.score == 1.0

    def test_f1_partial(self):
        result = score("the quick brown fox", "the quick fox", "f1")
        assert 0.0 < result.score < 1.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown scoring method"):
            score("a", "b", "nonexistent")

    def test_dict_expected_raises_for_standard(self):
        with pytest.raises(ValueError, match="Standard scoring requires"):
            score({"key": "val"}, "actual", "exact_match")


class TestCustomScoringUnavailable:
    """custom:* scoring is not available in adapt-gauge-core"""

    def test_custom_raises_value_error(self):
        with pytest.raises(ValueError, match="Custom scoring is not available"):
            score({"key": "val"}, "actual", "custom:route")

    def test_custom_any_type_raises(self):
        with pytest.raises(ValueError, match="Custom scoring is not available"):
            score("expected", "actual", "custom:resilience")


class TestLLMJudgeScoring:
    """llm_judge scoring with mock grader"""

    def test_llm_judge_requires_grader(self):
        with pytest.raises(ValueError, match="grader_client is required"):
            score("expected", "actual", "llm_judge")

    def test_llm_judge_with_mock_grader(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = ModelResponse(
            output='{"score": 0.8, "reason": "Good answer"}',
            latency_ms=100,
            model_name="mock-grader",
        )

        result = score(
            "expected answer",
            "actual answer",
            "llm_judge",
            grader_client=mock_client,
        )
        assert result.score == 0.8
        assert result.reason == "Good answer"

    def test_llm_judge_fallback_on_failure(self):
        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("API error")

        result = score(
            "the quick brown fox",
            "the quick brown fox",
            "llm_judge",
            grader_client=mock_client,
            fallback_method="f1",
        )
        # Should fall back to f1 scoring
        assert result.score == 1.0
        assert "fallback" in result.reason.lower()


class TestFallbackScorers:
    """_FALLBACK_SCORERS registry"""

    def test_contains_expected_methods(self):
        assert "exact_match" in _FALLBACK_SCORERS
        assert "contains" in _FALLBACK_SCORERS
        assert "f1" in _FALLBACK_SCORERS
