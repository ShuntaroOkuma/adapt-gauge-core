"""
Tests for use_cases.aei
"""

import pandas as pd
import pytest

from adapt_gauge_core.use_cases.aei import (
    compute_aei,
    detect_negative_learning,
    detect_peak_regression,
    detect_mid_curve_dip,
)


class TestComputeAEI:
    """Tests for compute_aei function"""

    def test_6axis_dataframe(self):
        """AEI should be calculated when all 6 axes are present"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "axis_Acquisition": 0.8,
                "axis_Resilience_Noise": 0.9,
                "axis_Resilience_Detect": 0.7,
                "axis_Efficiency": 0.6,
                "axis_Agency": 0.5,
                "axis_Fidelity": 0.85,
            },
            {
                "model_name": "model-b",
                "axis_Acquisition": 0.5,
                "axis_Resilience_Noise": 0.6,
                "axis_Resilience_Detect": 0.4,
                "axis_Efficiency": 0.3,
                "axis_Agency": 0.2,
                "axis_Fidelity": 0.55,
            },
        ])

        result = compute_aei(df)
        assert result is not None
        assert len(result) == 2
        # model-a should be ranked higher (sorted descending by AEI)
        assert result.iloc[0]["model_name"] == "model-a"
        assert result.iloc[1]["model_name"] == "model-b"
        # AEI is the mean of 6 axis scores
        expected_aei_a = (0.8 + 0.9 + 0.7 + 0.6 + 0.5 + 0.85) / 6
        assert result.iloc[0]["AEI"] == pytest.approx(expected_aei_a)

    def test_3axis_partial(self):
        """AEI should be calculated with only 3 axes present"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "axis_Acquisition": 0.8,
                "axis_Resilience_Noise": 0.9,
                "axis_Fidelity": 0.85,
            },
        ])

        result = compute_aei(df)
        assert result is not None
        assert len(result) == 1
        expected_aei = (0.8 + 0.9 + 0.85) / 3
        assert result.iloc[0]["AEI"] == pytest.approx(expected_aei)

    def test_no_axis_data_returns_none(self):
        """Should return None when no axis data is present"""
        df = pd.DataFrame([
            {"model_name": "model-a", "score_0shot": 0.5},
        ])

        result = compute_aei(df)
        assert result is None

    def test_all_nan_axis_returns_none(self):
        """Should return None when all axes are NaN"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "axis_Acquisition": None,
                "axis_Resilience_Noise": None,
                "axis_Fidelity": None,
            },
        ])

        result = compute_aei(df)
        assert result is None

    def test_sorted_descending(self):
        """Results should be sorted by AEI in descending order"""
        df = pd.DataFrame([
            {"model_name": "low", "axis_Acquisition": 0.1, "axis_Fidelity": 0.2},
            {"model_name": "high", "axis_Acquisition": 0.9, "axis_Fidelity": 0.8},
            {"model_name": "mid", "axis_Acquisition": 0.5, "axis_Fidelity": 0.5},
        ])

        result = compute_aei(df)
        assert result is not None
        assert list(result["model_name"]) == ["high", "mid", "low"]


class TestDetectNegativeLearning:
    """Tests for detect_negative_learning function"""

    def test_detects_degradation(self):
        """Should detect >= 10% performance degradation"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.5,  # 37.5% drop > 10%
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "model-a"
        assert alerts[0]["task_id"] == "task1"
        assert alerts[0]["drop_pct"] == pytest.approx(37.5)
        assert alerts[0]["type"] == "negative_learning"

    def test_detects_small_degradation(self):
        """Should detect 10-20% degradation (previously missed)"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.7,  # 12.5% drop, now detected
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1
        assert alerts[0]["drop_pct"] == pytest.approx(12.5)

    def test_severity_collapse(self):
        """Should classify >= 50% drop as collapse"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.3,  # 62.5% drop
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "collapse"

    def test_severity_degradation(self):
        """Should classify 10-50% drop as degradation"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.6,  # 25% drop
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "degradation"

    def test_no_detection_when_improvement(self):
        """Should not detect when performance improves"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.5,
                "score_8shot": 0.9,  # improved
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 0

    def test_no_detection_small_drop(self):
        """Should not detect drops below 10%"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.75,  # 6.25% drop < 10%
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 0

    def test_skip_low_baseline(self):
        """Should skip when 0-shot score is <= 0.05"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.03,
                "score_8shot": 0.01,
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 0

    def test_custom_label_fn(self):
        """label_fn callback should be called correctly"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.4,
            },
        ])

        label_fn = lambda tid, cat: f"[{cat}] {tid}"
        alerts = detect_negative_learning(df, label_fn=label_fn)
        assert len(alerts) == 1
        assert alerts[0]["task_label"] == "[cat1] task1"

    def test_default_label_fn_returns_task_id(self):
        """Default label_fn should return task_id as-is"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.4,
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1
        assert alerts[0]["task_label"] == "task1"

    def test_fallback_to_4shot(self):
        """Should fallback to score_4shot when score_8shot is not available"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_4shot": 0.3,
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1

    def test_sorted_by_drop_pct(self):
        """Results should be sorted by drop_pct in descending order"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.5,  # 37.5%
            },
            {
                "model_name": "model-b",
                "task_id": "task2",
                "category": "cat1",
                "score_0shot": 0.9,
                "score_8shot": 0.2,  # 77.8%
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 2
        assert alerts[0]["model"] == "model-b"
        assert alerts[1]["model"] == "model-a"


class TestDetectPeakRegression:
    """Tests for detect_peak_regression function"""

    def test_detects_learned_then_forgot(self):
        """Should detect when model learned (peak > 0-shot) then regressed"""
        # gemini-3-flash / custom_route pattern: 0.333 -> 0.636 -> 0.333
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.333,
                "score_1shot": 0.267,
                "score_2shot": 0.267,
                "score_4shot": 0.636,
                "score_8shot": 0.333,
            },
        ])

        alerts = detect_peak_regression(df)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "model-a"
        assert alerts[0]["type"] == "peak_regression"
        assert alerts[0]["score_peak"] == pytest.approx(0.636)
        assert alerts[0]["peak_shot"] == 4
        assert alerts[0]["score_final"] == pytest.approx(0.333)
        # drop from peak: (0.636 - 0.333) / 0.636 ~= 47.6%
        assert alerts[0]["drop_pct"] == pytest.approx(47.6, abs=0.1)

    def test_no_detection_when_final_is_peak(self):
        """Should not detect when final shot IS the peak"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.2,
                "score_1shot": 0.3,
                "score_2shot": 0.5,
                "score_4shot": 0.7,
                "score_8shot": 0.9,  # monotonic increase
            },
        ])

        alerts = detect_peak_regression(df)
        assert len(alerts) == 0

    def test_no_detection_when_no_learning(self):
        """Should not detect when peak is not 10% above 0-shot (no learning evidence)"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.80,
                "score_1shot": 0.82,  # only 2.5% above 0-shot
                "score_2shot": 0.81,
                "score_4shot": 0.79,
                "score_8shot": 0.60,
            },
        ])

        alerts = detect_peak_regression(df)
        assert len(alerts) == 0

    def test_no_detection_small_regression_from_peak(self):
        """Should not detect when regression from peak is < 20%"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.5,
                "score_1shot": 0.7,
                "score_2shot": 0.8,
                "score_4shot": 0.9,
                "score_8shot": 0.85,  # only 5.6% drop from peak
            },
        ])

        alerts = detect_peak_regression(df)
        assert len(alerts) == 0

    def test_skip_low_baseline(self):
        """Should skip when 0-shot score is <= 0.05"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.02,
                "score_1shot": 0.5,
                "score_2shot": 0.6,
                "score_4shot": 0.7,
                "score_8shot": 0.3,
            },
        ])

        alerts = detect_peak_regression(df)
        assert len(alerts) == 0

    def test_custom_label_fn(self):
        """label_fn callback should be called correctly"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.3,
                "score_1shot": 0.8,
                "score_2shot": 0.7,
                "score_4shot": 0.6,
                "score_8shot": 0.4,
            },
        ])

        label_fn = lambda tid, cat: f"[{cat}] {tid}"
        alerts = detect_peak_regression(df, label_fn=label_fn)
        assert len(alerts) == 1
        assert alerts[0]["task_label"] == "[cat1] task1"

    def test_sorted_by_drop_pct(self):
        """Results should be sorted by drop_pct descending"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.3,
                "score_1shot": 0.8,
                "score_2shot": 0.7,
                "score_4shot": 0.6,
                "score_8shot": 0.5,  # 37.5% from peak
            },
            {
                "model_name": "model-b",
                "task_id": "task2",
                "category": "cat1",
                "score_0shot": 0.3,
                "score_1shot": 0.9,
                "score_2shot": 0.5,
                "score_4shot": 0.4,
                "score_8shot": 0.35,  # 61.1% from peak
            },
        ])

        alerts = detect_peak_regression(df)
        assert len(alerts) == 2
        assert alerts[0]["model"] == "model-b"
        assert alerts[1]["model"] == "model-a"

    def test_handles_missing_shot_columns(self):
        """Should work with only 0shot and 4shot columns"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.3,
                "score_4shot": 0.8,
            },
        ])

        # No 8-shot column, should fallback to 4-shot as final
        alerts = detect_peak_regression(df)
        assert len(alerts) == 0  # peak IS the final


class TestDetectMidCurveDip:
    """Tests for detect_mid_curve_dip function"""

    def test_detects_sharp_drop(self):
        """Should detect >= 30% drop between adjacent shots"""
        # gemini-3-pro / classification: 0.60 -> 0.40 (33% drop)
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.60,
                "score_1shot": 0.40,
                "score_2shot": 0.40,
                "score_4shot": 0.40,
                "score_8shot": 0.60,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "model-a"
        assert alerts[0]["type"] == "mid_curve_dip"
        assert alerts[0]["from_shot"] == 0
        assert alerts[0]["to_shot"] == 1
        assert alerts[0]["score_from"] == pytest.approx(0.60)
        assert alerts[0]["score_to"] == pytest.approx(0.40)
        assert alerts[0]["drop_pct"] == pytest.approx(33.3, abs=0.1)

    def test_detects_mid_curve_drop_with_recovery(self):
        """Should detect mid-curve dip even when score recovers later"""
        # gemini-2.5-flash / summarization: 0.607 -> 0.282 (53.5% drop at 4-shot)
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.102,
                "score_1shot": 0.607,
                "score_2shot": 0.552,
                "score_4shot": 0.282,
                "score_8shot": 0.683,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) >= 1
        # The largest dip is 1-shot(0.607) -> 4-shot(0.282), but we check adjacent pairs
        # Actually 2-shot(0.552) -> 4-shot(0.282) = 48.9% drop is the adjacent pair
        largest = alerts[0]
        assert largest["drop_pct"] == pytest.approx(48.9, abs=0.5)

    def test_no_detection_monotonic_increase(self):
        """Should not detect anything for monotonically increasing scores"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.2,
                "score_1shot": 0.4,
                "score_2shot": 0.6,
                "score_4shot": 0.8,
                "score_8shot": 0.9,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) == 0

    def test_no_detection_small_dip(self):
        """Should not detect dips below 30%"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.5,
                "score_1shot": 0.7,
                "score_2shot": 0.6,  # 14.3% dip, below threshold
                "score_4shot": 0.8,
                "score_8shot": 0.9,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) == 0

    def test_skip_low_source_score(self):
        """Should skip when source score is <= 0.05"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.03,
                "score_1shot": 0.0,
                "score_2shot": 0.5,
                "score_4shot": 0.7,
                "score_8shot": 0.8,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) == 0

    def test_custom_label_fn(self):
        """label_fn callback should be called correctly"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.60,
                "score_1shot": 0.30,
                "score_2shot": 0.60,
                "score_4shot": 0.70,
                "score_8shot": 0.80,
            },
        ])

        label_fn = lambda tid, cat: f"[{cat}] {tid}"
        alerts = detect_mid_curve_dip(df, label_fn=label_fn)
        assert len(alerts) == 1
        assert alerts[0]["task_label"] == "[cat1] task1"

    def test_sorted_by_drop_pct(self):
        """Results should be sorted by drop_pct descending"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.60,
                "score_1shot": 0.30,  # 50% dip
                "score_2shot": 0.80,
                "score_4shot": 0.85,
                "score_8shot": 0.90,
            },
            {
                "model_name": "model-b",
                "task_id": "task2",
                "category": "cat1",
                "score_0shot": 0.80,
                "score_1shot": 0.20,  # 75% dip
                "score_2shot": 0.90,
                "score_4shot": 0.95,
                "score_8shot": 0.95,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) == 2
        assert alerts[0]["model"] == "model-b"
        assert alerts[1]["model"] == "model-a"

    def test_reports_one_alert_per_model_task(self):
        """Should report only the largest dip per model-task pair"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.60,
                "score_1shot": 0.30,  # 50% dip
                "score_2shot": 0.80,
                "score_4shot": 0.40,  # another 50% dip
                "score_8shot": 0.90,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        # Only the largest dip should be reported per model-task
        assert len(alerts) == 1

    def test_handles_missing_shot_columns(self):
        """Should work with only available shot columns"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.60,
                "score_4shot": 0.30,
            },
        ])

        alerts = detect_mid_curve_dip(df)
        assert len(alerts) == 1
        assert alerts[0]["drop_pct"] == pytest.approx(50.0)
