"""
src.use_cases.aei のテスト
"""

import pandas as pd
import pytest

from adapt_gauge_core.use_cases.aei import compute_aei, detect_negative_learning


class TestComputeAEI:
    """compute_aei関数のテスト"""

    def test_6axis_dataframe(self):
        """6軸全てのデータがある場合、AEIが算出される"""
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
        """3軸のみのデータでもAEIが算出される"""
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
        """軸データがない場合はNoneを返す"""
        df = pd.DataFrame([
            {"model_name": "model-a", "score_0shot": 0.5},
        ])

        result = compute_aei(df)
        assert result is None

    def test_all_nan_axis_returns_none(self):
        """全軸がNaNの場合もNoneを返す"""
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
        """結果はAEI降順でソートされる"""
        df = pd.DataFrame([
            {"model_name": "low", "axis_Acquisition": 0.1, "axis_Fidelity": 0.2},
            {"model_name": "high", "axis_Acquisition": 0.9, "axis_Fidelity": 0.8},
            {"model_name": "mid", "axis_Acquisition": 0.5, "axis_Fidelity": 0.5},
        ])

        result = compute_aei(df)
        assert result is not None
        assert list(result["model_name"]) == ["high", "mid", "low"]


class TestDetectNegativeLearning:
    """detect_negative_learning関数のテスト"""

    def test_detects_degradation(self):
        """20%以上の性能劣化を検出"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.5,  # 37.5% drop > 20%
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "model-a"
        assert alerts[0]["task_id"] == "task1"
        assert alerts[0]["drop_pct"] == pytest.approx(37.5)

    def test_no_detection_when_improvement(self):
        """改善の場合は検出しない"""
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
        """20%未満の劣化は検出しない"""
        df = pd.DataFrame([
            {
                "model_name": "model-a",
                "task_id": "task1",
                "category": "cat1",
                "score_0shot": 0.8,
                "score_8shot": 0.7,  # 12.5% drop < 20%
            },
        ])

        alerts = detect_negative_learning(df)
        assert len(alerts) == 0

    def test_skip_low_baseline(self):
        """0-shotスコアが0.05以下の場合はスキップ"""
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
        """label_fnコールバックが正しく呼ばれる"""
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
        """デフォルトlabel_fnはtask_idをそのまま返す"""
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
        """score_8shotがない場合はscore_4shotにフォールバック"""
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
        """結果は劣化率降順でソートされる"""
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
