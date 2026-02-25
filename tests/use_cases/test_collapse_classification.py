"""
Tests for collapse pattern classification and resilience score.
"""

import pandas as pd
import pytest

from adapt_gauge_core.use_cases.aei import (
    classify_collapse_pattern,
    calculate_resilience_score,
)


def _make_row(model: str, task: str, scores: dict[int, float]) -> dict:
    """Helper to build a summary DataFrame row."""
    row = {"model_name": model, "task_id": task, "category": "test"}
    for shot, score in scores.items():
        row[f"score_{shot}shot"] = score
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestClassifyCollapsePattern:
    """Tests for classify_collapse_pattern function."""

    def test_stable_monotonic_increase(self):
        """Monotonically increasing scores should be classified as stable."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.2, 1: 0.4, 2: 0.6, 4: 0.7, 8: 0.8}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "stable"

    def test_stable_flat(self):
        """Flat scores (no change) should be classified as stable."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.5, 1: 0.5, 2: 0.5, 4: 0.5, 8: 0.5}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "stable"

    def test_immediate_collapse(self):
        """Score that drops immediately from 0-shot and stays low."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.8, 1: 0.3, 2: 0.2, 4: 0.2, 8: 0.1}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "immediate_collapse"

    def test_gradual_decline(self):
        """Score that gradually decreases across shot counts."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.8, 1: 0.7, 2: 0.6, 4: 0.5, 8: 0.4}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "gradual_decline"

    def test_peak_regression(self):
        """Score that peaks at intermediate shot then drops back."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.8, 4: 0.7, 8: 0.3}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "peak_regression"

    def test_multiple_rows(self):
        """Should classify each model-task combination independently."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.2, 1: 0.4, 2: 0.6, 4: 0.7, 8: 0.8}),
            _make_row("model-b", "task-1", {0: 0.8, 1: 0.3, 2: 0.2, 4: 0.2, 8: 0.1}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 2
        patterns = {r["model"]: r["pattern"] for r in result}
        assert patterns["model-a"] == "stable"
        assert patterns["model-b"] == "immediate_collapse"

    def test_result_contains_required_fields(self):
        """Result dicts must contain model, task_id, pattern, and scores."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.5, 1: 0.6, 2: 0.7, 4: 0.8, 8: 0.9}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        entry = result[0]
        assert "model" in entry
        assert "task_id" in entry
        assert "pattern" in entry
        assert "scores" in entry
        assert entry["model"] == "model-a"
        assert entry["task_id"] == "task-1"

    def test_low_baseline_classified_as_stable(self):
        """Very low 0-shot baseline (<=0.05) should default to stable."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.02, 1: 0.01, 2: 0.0, 4: 0.0, 8: 0.0}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "stable"

    def test_small_decline_is_stable(self):
        """Minor fluctuations (<10% overall drop) should be stable."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.8, 1: 0.78, 2: 0.79, 4: 0.77, 8: 0.76}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "stable"

    def test_missing_intermediate_shots(self):
        """Should work even with missing intermediate shot columns."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.5, 8: 0.9}),
        ])
        result = classify_collapse_pattern(df)
        assert len(result) == 1
        assert result[0]["pattern"] == "stable"

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame(columns=["model_name", "task_id", "score_0shot", "score_8shot"])
        result = classify_collapse_pattern(df)
        assert result == []


class TestCalculateResilienceScore:
    """Tests for calculate_resilience_score function."""

    def test_all_stable_returns_1(self):
        """Model with all stable patterns should score 1.0."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.6, 4: 0.7, 8: 0.8}),
            _make_row("model-a", "task-2", {0: 0.4, 1: 0.5, 2: 0.6, 4: 0.7, 8: 0.8}),
        ])
        result = calculate_resilience_score(df)
        assert "model-a" in result
        assert result["model-a"] == pytest.approx(1.0)

    def test_all_collapse_returns_low(self):
        """Model with all collapses should score near 0."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.8, 1: 0.2, 2: 0.1, 4: 0.05, 8: 0.0}),
            _make_row("model-a", "task-2", {0: 0.9, 1: 0.3, 2: 0.1, 4: 0.05, 8: 0.0}),
        ])
        result = calculate_resilience_score(df)
        assert result["model-a"] < 0.3

    def test_mixed_patterns(self):
        """Model with mix of stable and collapse should score between 0 and 1."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.6, 4: 0.7, 8: 0.8}),
            _make_row("model-a", "task-2", {0: 0.8, 1: 0.3, 2: 0.2, 4: 0.1, 8: 0.05}),
        ])
        result = calculate_resilience_score(df)
        assert 0.0 < result["model-a"] < 1.0

    def test_multiple_models(self):
        """Should return scores for each model independently."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.8, 8: 0.9}),
            _make_row("model-b", "task-1", {0: 0.8, 1: 0.2, 2: 0.1, 4: 0.05, 8: 0.0}),
        ])
        result = calculate_resilience_score(df)
        assert result["model-a"] > result["model-b"]

    def test_score_range_0_to_1(self):
        """All scores should be in [0, 1] range."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.5, 1: 0.6, 2: 0.7, 4: 0.8, 8: 0.9}),
            _make_row("model-b", "task-1", {0: 0.9, 1: 0.1, 2: 0.05, 4: 0.0, 8: 0.0}),
        ])
        result = calculate_resilience_score(df)
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_peak_regression_penalizes_from_peak(self):
        """Peak regression penalty should use peak-to-final drop, not 0-shot-to-final."""
        # 0-shot=0.3, peaks at 0.8, drops to 0.3 -> same as 0-shot but
        # peak regression should still penalize (peak drop = 62.5%)
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.8, 4: 0.7, 8: 0.3}),
        ])
        result = calculate_resilience_score(df)
        assert result["model-a"] < 1.0

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty dict."""
        df = pd.DataFrame(columns=["model_name", "task_id", "score_0shot", "score_8shot"])
        result = calculate_resilience_score(df)
        assert result == {}


class TestRunnerIntegration:
    """Tests that classification & resilience integrate into summary_df correctly."""

    def test_summary_df_gets_collapse_pattern_column(self):
        """classify_collapse_pattern results should map into a DataFrame column."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.2, 1: 0.4, 2: 0.6, 4: 0.7, 8: 0.8}),
            _make_row("model-b", "task-1", {0: 0.8, 1: 0.3, 2: 0.2, 4: 0.2, 8: 0.1}),
        ])
        classifications = classify_collapse_pattern(df)
        pattern_map = {
            (c["model"], c["task_id"]): c["pattern"]
            for c in classifications
        }
        df["collapse_pattern"] = df.apply(
            lambda r: pattern_map.get((r["model_name"], r.get("task_id", "")), ""),
            axis=1,
        )

        assert "collapse_pattern" in df.columns
        assert df.loc[df["model_name"] == "model-a", "collapse_pattern"].iloc[0] == "stable"
        assert df.loc[df["model_name"] == "model-b", "collapse_pattern"].iloc[0] == "immediate_collapse"

    def test_summary_df_gets_resilience_score_column(self):
        """calculate_resilience_score results should map into a DataFrame column."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.6, 4: 0.7, 8: 0.8}),
            _make_row("model-b", "task-1", {0: 0.8, 1: 0.3, 2: 0.2, 4: 0.2, 8: 0.1}),
        ])
        resilience_scores = calculate_resilience_score(df)
        df["resilience_score"] = df["model_name"].map(resilience_scores)

        assert "resilience_score" in df.columns
        assert df.loc[df["model_name"] == "model-a", "resilience_score"].iloc[0] == pytest.approx(1.0)
        assert df.loc[df["model_name"] == "model-b", "resilience_score"].iloc[0] < 0.3

    def test_columns_survive_csv_roundtrip(self, tmp_path):
        """New columns should persist through CSV save/load cycle."""
        df = _make_df([
            _make_row("model-a", "task-1", {0: 0.3, 1: 0.5, 2: 0.8, 4: 0.7, 8: 0.3}),
        ])
        classifications = classify_collapse_pattern(df)
        resilience_scores = calculate_resilience_score(df)

        pattern_map = {(c["model"], c["task_id"]): c["pattern"] for c in classifications}
        df["collapse_pattern"] = df.apply(
            lambda r: pattern_map.get((r["model_name"], r.get("task_id", "")), ""),
            axis=1,
        )
        df["resilience_score"] = df["model_name"].map(resilience_scores)

        csv_path = tmp_path / "summary.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)

        assert loaded["collapse_pattern"].iloc[0] == "peak_regression"
        assert loaded["resilience_score"].iloc[0] == pytest.approx(
            resilience_scores["model-a"], abs=1e-6
        )
