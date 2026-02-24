"""
Tests for efficiency_calc.py
"""

import pytest
from adapt_gauge_core.efficiency_calc import (
    CostMetrics,
    calculate_total_cost,
    calculate_economy_score,
    improvement_rate,
    threshold_shots,
    learning_curve_auc,
    calculate_all_metrics,
)


class TestCostMetrics:
    """Tests for CostMetrics dataclass"""

    def test_create_with_defaults(self):
        """Create instance with default values"""
        metrics = CostMetrics(
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1500
        )
        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.latency_ms == 1500
        assert metrics.input_price_per_m == 0.0
        assert metrics.output_price_per_m == 0.0
        assert metrics.time_price_per_sec == 0.0

    def test_create_with_all_values(self):
        """Create instance with all values specified"""
        metrics = CostMetrics(
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1500,
            input_price_per_m=3.0,
            output_price_per_m=15.0,
            time_price_per_sec=0.001
        )
        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.latency_ms == 1500
        assert metrics.input_price_per_m == 3.0
        assert metrics.output_price_per_m == 15.0
        assert metrics.time_price_per_sec == 0.001

    def test_negative_input_tokens_raises_error(self):
        """Should raise error for negative input tokens"""
        with pytest.raises(ValueError, match="input_tokens must be non-negative"):
            CostMetrics(input_tokens=-1, output_tokens=500, latency_ms=1500)

    def test_negative_output_tokens_raises_error(self):
        """Should raise error for negative output tokens"""
        with pytest.raises(ValueError, match="output_tokens must be non-negative"):
            CostMetrics(input_tokens=1000, output_tokens=-1, latency_ms=1500)

    def test_negative_latency_raises_error(self):
        """Should raise error for negative latency"""
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            CostMetrics(input_tokens=1000, output_tokens=500, latency_ms=-1)


class TestCalculateTotalCost:
    """Tests for calculate_total_cost function"""

    def test_zero_cost_with_defaults(self):
        """Cost should be 0 with default prices (0)"""
        metrics = CostMetrics(
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1500
        )
        assert calculate_total_cost(metrics) == 0.0

    def test_token_cost_calculation(self):
        """Token cost calculation"""
        # 1M input tokens * $3 = $3
        # 1M output tokens * $15 = $15
        # Total = $18
        metrics = CostMetrics(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            latency_ms=0,
            input_price_per_m=3.0,
            output_price_per_m=15.0
        )
        assert calculate_total_cost(metrics) == 18.0

    def test_time_cost_calculation(self):
        """Time cost calculation"""
        # 1000ms = 1s, 1s * $0.001 = $0.001
        metrics = CostMetrics(
            input_tokens=0,
            output_tokens=0,
            latency_ms=1000,
            time_price_per_sec=0.001
        )
        assert calculate_total_cost(metrics) == 0.001

    def test_combined_cost_calculation(self):
        """Token cost + time cost calculation"""
        # 1000 input tokens * $3 / 1M = $0.003
        # 500 output tokens * $15 / 1M = $0.0075
        # 2000ms = 2s * $0.01 = $0.02
        # Total = $0.003 + $0.0075 + $0.02 = $0.0305
        metrics = CostMetrics(
            input_tokens=1000,
            output_tokens=500,
            latency_ms=2000,
            input_price_per_m=3.0,
            output_price_per_m=15.0,
            time_price_per_sec=0.01
        )
        expected = (1000 / 1_000_000 * 3.0) + (500 / 1_000_000 * 15.0) + (2000 / 1000 * 0.01)
        assert calculate_total_cost(metrics) == pytest.approx(expected)


class TestCostEfficiency:
    """Tests for calculate_economy_score function"""

    def test_normal_calculation(self):
        """Normal calculation"""
        # Accuracy 0.8 / cost $0.01 = 80
        assert calculate_economy_score(0.8, 0.01) == 80.0

    def test_zero_cost_returns_zero(self):
        """Should return 0 when cost is 0"""
        assert calculate_economy_score(0.8, 0.0) == 0.0

    def test_negative_cost_returns_zero(self):
        """Should return 0 when cost is negative"""
        assert calculate_economy_score(0.8, -0.01) == 0.0

    def test_perfect_accuracy(self):
        """100% accuracy case"""
        assert calculate_economy_score(1.0, 0.1) == 10.0

    def test_zero_accuracy(self):
        """0% accuracy case"""
        assert calculate_economy_score(0.0, 0.1) == 0.0


class TestImprovementRate:
    """Tests for improvement_rate function"""

    def test_normal_calculation(self):
        """Normal calculation"""
        scores = {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.85, 8: 0.95}
        # (0.95 - 0.3) / 8 = 0.08125
        assert improvement_rate(scores) == pytest.approx(0.08125)

    def test_no_improvement(self):
        """No improvement case"""
        scores = {0: 0.5, 1: 0.5, 2: 0.5, 4: 0.5, 8: 0.5}
        assert improvement_rate(scores) == 0.0

    def test_negative_improvement(self):
        """Degradation case (negative improvement rate)"""
        scores = {0: 0.8, 1: 0.7, 2: 0.6, 4: 0.5, 8: 0.4}
        assert improvement_rate(scores) == pytest.approx(-0.05)


class TestThresholdShots:
    """Tests for threshold_shots function"""

    def test_threshold_reached_at_zero(self):
        """Threshold reached at 0-shot"""
        scores = {0: 0.9, 1: 0.92, 2: 0.95, 4: 0.97, 8: 0.99}
        assert threshold_shots(scores, 0.8) == 0

    def test_threshold_reached_at_two(self):
        """Threshold reached at 2-shot"""
        scores = {0: 0.3, 1: 0.5, 2: 0.85, 4: 0.9, 8: 0.95}
        assert threshold_shots(scores, 0.8) == 2

    def test_threshold_not_reached(self):
        """Threshold not reached"""
        scores = {0: 0.3, 1: 0.4, 2: 0.5, 4: 0.6, 8: 0.7}
        assert threshold_shots(scores, 0.8) == -1

    def test_custom_threshold(self):
        """Custom threshold value"""
        scores = {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.85, 8: 0.95}
        assert threshold_shots(scores, 0.9) == 8


class TestLearningCurveAuc:
    """Tests for learning_curve_auc function"""

    def test_perfect_score(self):
        """100% score at all shot counts"""
        scores = {0: 1.0, 1: 1.0, 2: 1.0, 4: 1.0, 8: 1.0}
        # All intervals at 1.0 so AUC=1.0
        assert learning_curve_auc(scores) == 1.0

    def test_zero_score(self):
        """0% score at all shot counts"""
        scores = {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0}
        assert learning_curve_auc(scores) == 0.0

    def test_linear_improvement(self):
        """Linear improvement case"""
        # e.g. 0, 0.125, 0.25, 0.5, 1.0 (linear over x-axis 0,1,2,4,8)
        scores = {0: 0.0, 1: 0.125, 2: 0.25, 4: 0.5, 8: 1.0}
        # Trapezoidal rule:
        # [0,1]: (0 + 0.125)/2 * 1 = 0.0625
        # [1,2]: (0.125 + 0.25)/2 * 1 = 0.1875
        # [2,4]: (0.25 + 0.5)/2 * 2 = 0.75
        # [4,8]: (0.5 + 1.0)/2 * 4 = 3.0
        # Total = 4.0, normalized = 4.0 / 8 = 0.5
        assert learning_curve_auc(scores) == pytest.approx(0.5)

    def test_missing_shot_raises_error(self):
        """Should raise error when required shot count is missing"""
        scores = {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.85}  # 8-shot missing
        with pytest.raises(ValueError, match="Score for shot count 8 is missing"):
            learning_curve_auc(scores)


class TestCalculateAllMetrics:
    """Tests for calculate_all_metrics function"""

    def test_all_metrics_returned(self):
        """All metrics should be returned"""
        scores = {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.85, 8: 0.95}
        result = calculate_all_metrics(scores)

        assert "improvement_rate" in result
        assert "threshold_shots" in result
        assert "learning_curve_auc" in result

    def test_metrics_values(self):
        """Each metric value should be correct"""
        scores = {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.85, 8: 0.95}
        result = calculate_all_metrics(scores, threshold=0.8)

        assert result["improvement_rate"] == pytest.approx(0.08125)
        assert result["threshold_shots"] == 4  # 0.85 >= 0.8
        # AUC calculation verified in separate test
        assert result["learning_curve_auc"] > 0
