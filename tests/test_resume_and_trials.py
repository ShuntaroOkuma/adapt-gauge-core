"""
Tests for resume capability, multi-trial aggregation, and per-evaluation error handling.
"""

import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from adapt_gauge_core.domain.constants import SHOT_SCHEDULE
from adapt_gauge_core.domain.entities import EvaluationResult
from adapt_gauge_core.domain.value_objects import ModelResponse
from adapt_gauge_core.harness_config import HarnessConfig, TrialConfig, ReliabilityConfig
from adapt_gauge_core.runner import _build_skip_set, _save_raw_results
from adapt_gauge_core.task_loader import Task, TestCase, Example
from adapt_gauge_core.use_cases.evaluation import run_single_evaluation, aggregate_results


class TestTrialId:
    """Test trial_id in EvaluationResult"""

    def test_default_trial_id(self):
        result = EvaluationResult(
            run_id="r1", task_id="t1", category="c1", model_name="m1",
            shot_count=0, input="i", expected_output="e", actual_output="a",
            score=1.0, scoring_method="f1", latency_ms=50, timestamp="ts",
        )
        assert result.trial_id == 1

    def test_explicit_trial_id(self):
        result = EvaluationResult(
            run_id="r1", task_id="t1", category="c1", model_name="m1",
            shot_count=0, input="i", expected_output="e", actual_output="a",
            score=1.0, scoring_method="f1", latency_ms=50, timestamp="ts",
            trial_id=3,
        )
        assert result.trial_id == 3

    def test_trial_id_in_asdict(self):
        result = EvaluationResult(
            run_id="r1", task_id="t1", category="c1", model_name="m1",
            shot_count=0, input="i", expected_output="e", actual_output="a",
            score=1.0, scoring_method="f1", latency_ms=50, timestamp="ts",
            trial_id=2,
        )
        d = asdict(result)
        assert d["trial_id"] == 2

    def test_run_single_evaluation_passes_trial_id(self):
        task = Task(
            task_id="t1", category="c1", description="d", difficulty="easy",
            examples=[Example(input="a", output="b")],
            test_cases=[TestCase(input="hello", expected_output="hello", scoring_method="exact_match")],
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = ModelResponse(
            output="hello", latency_ms=50, model_name="mock",
        )

        result = run_single_evaluation(
            task=task, test_case=task.test_cases[0], model_client=mock_client,
            shot_count=0, run_id="r1", trial_id=5,
        )
        assert result.trial_id == 5


class TestMultiTrialAggregation:
    """Test multi-trial aggregation in aggregate_results()"""

    def _make_multi_trial_results(self, num_trials=3):
        results = []
        for trial_id in range(1, num_trials + 1):
            for shot in SHOT_SCHEDULE:
                score_val = min(1.0, 0.2 + shot * 0.1 + trial_id * 0.05)
                results.append(EvaluationResult(
                    run_id="test_run", task_id="test_task", category="test",
                    model_name="mock-model", shot_count=shot,
                    input="test input", expected_output="expected",
                    actual_output="actual", score=score_val,
                    scoring_method="f1", latency_ms=50,
                    timestamp="2024-01-01T00:00:00",
                    trial_id=trial_id,
                ))
        return results

    def test_multi_trial_produces_single_summary_row(self):
        results = self._make_multi_trial_results(num_trials=3)
        config = HarnessConfig(
            trials=TrialConfig(num_trials=3, aggregation="mean"),
            reliability=ReliabilityConfig(calculate_pass_at_k=True, k_values=[1, 3]),
        )
        summary = aggregate_results(results, config=config)

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["num_trials"] == 3

    def test_multi_trial_has_reliability_metrics(self):
        results = self._make_multi_trial_results(num_trials=3)
        config = HarnessConfig(
            trials=TrialConfig(num_trials=3, aggregation="mean"),
            reliability=ReliabilityConfig(calculate_pass_at_k=True, k_values=[1, 3]),
        )
        summary = aggregate_results(results, config=config)

        row = summary.iloc[0]
        assert "score_variance" in row
        assert "pass_at_1" in summary.columns
        assert "pass_all_1" in summary.columns
        assert "pass_at_3" in summary.columns
        assert "pass_all_3" in summary.columns

    def test_single_trial_no_pass_at_k(self):
        results = self._make_multi_trial_results(num_trials=1)
        config = HarnessConfig(
            trials=TrialConfig(num_trials=1),
            reliability=ReliabilityConfig(calculate_pass_at_k=True, k_values=[1, 3]),
        )
        summary = aggregate_results(results, config=config)

        row = summary.iloc[0]
        assert row["num_trials"] == 1
        assert row["score_variance"] == 0.0
        assert "pass_at_1" not in summary.columns

    def test_median_aggregation(self):
        results = self._make_multi_trial_results(num_trials=3)
        config = HarnessConfig(
            trials=TrialConfig(num_trials=3, aggregation="median"),
            reliability=ReliabilityConfig(calculate_pass_at_k=False),
        )
        summary = aggregate_results(results, config=config)

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["num_trials"] == 3

    def test_backward_compat_no_trial_id_column(self):
        """Results without trial_id should default to 1."""
        results = []
        for shot in SHOT_SCHEDULE:
            results.append(EvaluationResult(
                run_id="test_run", task_id="test_task", category="test",
                model_name="mock-model", shot_count=shot,
                input="test", expected_output="exp", actual_output="act",
                score=0.5, scoring_method="f1", latency_ms=50,
                timestamp="ts",
            ))
        summary = aggregate_results(results)
        assert len(summary) == 1
        assert summary.iloc[0]["num_trials"] == 1


class TestExampleSelectionAggregation:
    """Test that aggregate_results groups by example_selection when present."""

    def _make_results_with_selection(self, selection_methods=("fixed", "tfidf")):
        results = []
        for sel in selection_methods:
            for shot in SHOT_SCHEDULE:
                # Give tfidf slightly different scores so we can verify separation
                bonus = 0.05 if sel == "tfidf" else 0.0
                score_val = min(1.0, 0.3 + shot * 0.08 + bonus)
                results.append(EvaluationResult(
                    run_id="test_run", task_id="test_task", category="test",
                    model_name="mock-model", shot_count=shot,
                    input="test input", expected_output="expected",
                    actual_output="actual", score=score_val,
                    scoring_method="f1", latency_ms=50,
                    timestamp="2024-01-01T00:00:00",
                    example_selection=sel,
                ))
        return results

    def test_two_selection_methods_produce_two_rows(self):
        results = self._make_results_with_selection(["fixed", "tfidf"])
        summary = aggregate_results(results)

        assert len(summary) == 2
        assert set(summary["example_selection"]) == {"fixed", "tfidf"}

    def test_single_selection_produces_one_row(self):
        results = self._make_results_with_selection(["fixed"])
        summary = aggregate_results(results)

        assert len(summary) == 1
        assert summary.iloc[0]["example_selection"] == "fixed"

    def test_scores_differ_between_selection_methods(self):
        results = self._make_results_with_selection(["fixed", "tfidf"])
        summary = aggregate_results(results)

        fixed_row = summary[summary["example_selection"] == "fixed"].iloc[0]
        tfidf_row = summary[summary["example_selection"] == "tfidf"].iloc[0]

        # tfidf has +0.05 bonus, so its 0-shot score should be higher
        assert tfidf_row["score_0shot"] > fixed_row["score_0shot"]

    def test_default_example_selection_is_fixed(self):
        """Results without explicit example_selection should default to 'fixed'."""
        results = []
        for shot in SHOT_SCHEDULE:
            results.append(EvaluationResult(
                run_id="test_run", task_id="test_task", category="test",
                model_name="mock-model", shot_count=shot,
                input="test", expected_output="exp", actual_output="act",
                score=0.5, scoring_method="f1", latency_ms=50,
                timestamp="ts",
            ))
        summary = aggregate_results(results)
        assert len(summary) == 1
        assert summary.iloc[0]["example_selection"] == "fixed"

    def test_example_selection_column_always_present(self):
        results = self._make_results_with_selection(["fixed"])
        summary = aggregate_results(results)
        assert "example_selection" in summary.columns


class TestResumeCapability:
    """Test resume via _build_skip_set"""

    def test_build_skip_set_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw_results_test.csv"

            data = [{
                "run_id": "test", "task_id": "t1", "category": "c1",
                "model_name": "m1", "shot_count": 0, "trial_id": 1,
                "input": "hello", "expected_output": "world",
                "actual_output": "world", "score": 1.0,
                "scoring_method": "exact_match", "latency_ms": 50,
                "timestamp": "ts", "input_tokens": 10, "output_tokens": 5,
            }]
            pd.DataFrame(data).to_csv(raw_path, index=False)

            existing, skip_set = _build_skip_set(raw_path)

            assert len(existing) == 1
            assert len(skip_set) == 1

            expected_key = (
                "t1", "m1", 0, 1,
                hashlib.sha256("hello".encode("utf-8")).hexdigest(),
                "fixed",
            )
            assert expected_key in skip_set

    def test_build_skip_set_nonexistent_file(self):
        existing, skip_set = _build_skip_set(Path("/nonexistent/file.csv"))
        assert existing == []
        assert skip_set == set()

    def test_build_skip_set_empty_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "empty.csv"
            pd.DataFrame().to_csv(raw_path, index=False)

            existing, skip_set = _build_skip_set(raw_path)
            assert existing == []
            assert skip_set == set()


class TestIntermediateSave:
    """Test intermediate result saving"""

    def test_save_raw_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "results" / "raw.csv"

            results = [{"run_id": "r1", "score": 1.0, "task_id": "t1"}]
            _save_raw_results(results, raw_path)

            assert raw_path.exists()
            loaded = pd.read_csv(raw_path)
            assert len(loaded) == 1
            assert loaded.iloc[0]["score"] == 1.0

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "a" / "b" / "c" / "raw.csv"

            _save_raw_results([{"x": 1}], raw_path)
            assert raw_path.exists()
