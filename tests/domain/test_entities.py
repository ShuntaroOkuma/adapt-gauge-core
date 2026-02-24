"""Tests for domain entities and value objects"""

import pytest

from adapt_gauge_core.domain.entities import (
    EvaluationResult,
    AcquisitionMetrics,
    HealthCheckResult,
)
from adapt_gauge_core.domain.value_objects import (
    CostMetrics,
    ScoringResult,
    ModelResponse,
)


class TestEvaluationResult:
    def test_construction(self):
        result = EvaluationResult(
            run_id="20260101_120000",
            task_id="test_task",
            category="test_category",
            model_name="gemini-2.5-flash",
            shot_count=0,
            input="test input",
            expected_output="expected",
            actual_output="actual",
            score=0.8,
            scoring_method="exact_match",
            latency_ms=100,
            timestamp="2026-01-01T12:00:00",
        )
        assert result.run_id == "20260101_120000"
        assert result.score == 0.8
        assert result.input_tokens == 0  # default
        assert result.output_tokens == 0  # default

    def test_with_token_counts(self):
        result = EvaluationResult(
            run_id="r1",
            task_id="t1",
            category="c1",
            model_name="m1",
            shot_count=0,
            input="i",
            expected_output="e",
            actual_output="a",
            score=1.0,
            scoring_method="f1",
            latency_ms=50,
            timestamp="ts",
            input_tokens=100,
            output_tokens=200,
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 200


class TestAcquisitionMetrics:
    def test_construction(self):
        metrics = AcquisitionMetrics(
            improvement_rate=0.05,
            threshold_shots=4,
            learning_curve_auc=0.75,
        )
        assert metrics.improvement_rate == 0.05
        assert metrics.threshold_shots == 4
        assert metrics.learning_curve_auc == 0.75


class TestHealthCheckResult:
    def test_success(self):
        result = HealthCheckResult(
            model_name="gemini-2.5-flash",
            success=True,
            latency_ms=150,
            error=None,
        )
        assert result.success is True
        assert result.latency_ms == 150
        assert result.error is None

    def test_failure(self):
        result = HealthCheckResult(
            model_name="gemini-2.5-flash",
            success=False,
            latency_ms=None,
            error="Connection refused",
        )
        assert result.success is False
        assert result.latency_ms is None
        assert result.error == "Connection refused"


class TestCostMetrics:
    def test_construction(self):
        metrics = CostMetrics(
            input_tokens=1000,
            output_tokens=500,
            latency_ms=200,
        )
        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.latency_ms == 200
        assert metrics.input_price_per_m == 0.0  # default

    def test_negative_input_tokens_raises(self):
        with pytest.raises(ValueError, match="input_tokens must be non-negative"):
            CostMetrics(input_tokens=-1, output_tokens=0, latency_ms=0)

    def test_negative_output_tokens_raises(self):
        with pytest.raises(ValueError, match="output_tokens must be non-negative"):
            CostMetrics(input_tokens=0, output_tokens=-1, latency_ms=0)

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            CostMetrics(input_tokens=0, output_tokens=0, latency_ms=-1)


class TestScoringResult:
    def test_with_score_only(self):
        result = ScoringResult(score=0.8)
        assert result.score == 0.8
        assert result.reason is None

    def test_with_reason(self):
        result = ScoringResult(score=1.0, reason="Perfect match")
        assert result.score == 1.0
        assert result.reason == "Perfect match"


class TestModelResponse:
    def test_construction(self):
        response = ModelResponse(
            output="Hello",
            latency_ms=100,
            model_name="gemini-2.5-flash",
        )
        assert response.output == "Hello"
        assert response.latency_ms == 100
        assert response.input_tokens == 0  # default
        assert response.output_tokens == 0  # default

    def test_with_tokens(self):
        response = ModelResponse(
            output="Hi",
            latency_ms=50,
            model_name="claude-sonnet",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.input_tokens == 10
        assert response.output_tokens == 5
