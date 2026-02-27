"""
Integration test for the CLI pipeline (using mocks for model clients).

Verifies the full evaluation pipeline works end-to-end:
1. Load task pack
2. Run evaluations (mocked model clients)
3. Aggregate results
4. Detect few-shot collapse
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from adapt_gauge_core.domain.constants import SHOT_SCHEDULE
from adapt_gauge_core.domain.value_objects import ModelResponse, ScoringResult
from adapt_gauge_core.task_loader import load_task_pack, Task, TestCase, Example
from adapt_gauge_core.use_cases.evaluation import run_single_evaluation, aggregate_results
from adapt_gauge_core.use_cases.aei import detect_few_shot_collapse
from adapt_gauge_core.use_cases.health_check import get_llm_judge_tasks


class TestTaskPackLoading:
    """Test loading the core demo pack"""

    def test_load_core_demo_pack(self):
        pack = load_task_pack("tasks/task_pack_core_demo.json")
        assert pack.pack_id == "task_pack_core_demo"
        assert pack.pack_name == "Core Demo Pack"
        assert len(pack.tasks) == 4

    def test_core_demo_scoring_methods(self):
        pack = load_task_pack("tasks/task_pack_core_demo.json")
        scoring_methods = set()
        for task in pack.tasks:
            for tc in task.test_cases:
                scoring_methods.add(tc.scoring_method)
        assert scoring_methods == {"f1", "exact_match", "contains", "llm_judge"}

    def test_llm_judge_tasks_detected(self):
        pack = load_task_pack("tasks/task_pack_core_demo.json")
        llm_tasks = get_llm_judge_tasks(pack)
        assert "custom_route_001" in llm_tasks


class TestSingleEvaluation:
    """Test running a single evaluation with mock client"""

    def _make_task(self):
        return Task(
            task_id="test_task",
            category="test",
            description="Test task",
            difficulty="easy",
            examples=[Example(input="a", output="b")],
            test_cases=[
                TestCase(input="hello", expected_output="hello", scoring_method="exact_match")
            ],
        )

    def test_run_single_evaluation_exact_match(self):
        task = self._make_task()
        mock_client = MagicMock()
        mock_client.generate.return_value = ModelResponse(
            output="hello",
            latency_ms=50,
            model_name="mock-model",
            input_tokens=10,
            output_tokens=5,
        )

        result = run_single_evaluation(
            task=task,
            test_case=task.test_cases[0],
            model_client=mock_client,
            shot_count=0,
            run_id="test_run",
        )

        assert result.score == 1.0
        assert result.model_name == "mock-model"
        assert result.task_id == "test_task"
        assert result.shot_count == 0


class TestAggregation:
    """Test result aggregation"""

    def _make_mock_results(self):
        from adapt_gauge_core.domain.entities import EvaluationResult

        results = []
        for shot in SHOT_SCHEDULE:
            # Score increases with shot count
            score_val = min(1.0, 0.2 + shot * 0.1)
            results.append(EvaluationResult(
                run_id="test_run",
                task_id="test_task",
                category="test",
                model_name="mock-model",
                shot_count=shot,
                input="test input",
                expected_output="expected",
                actual_output="actual",
                score=score_val,
                scoring_method="f1",
                latency_ms=50,
                timestamp="2024-01-01T00:00:00",
                input_tokens=10,
                output_tokens=5,
            ))
        return results

    def test_aggregate_produces_summary(self):
        results = self._make_mock_results()
        summary = aggregate_results(results)

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["task_id"] == "test_task"
        assert row["model_name"] == "mock-model"
        assert "improvement_rate" in row
        assert "threshold_shots" in row
        assert "learning_curve_auc" in row

    def test_no_axis_scores_in_core(self):
        """Core aggregation should NOT include axis_* columns"""
        results = self._make_mock_results()
        summary = aggregate_results(results)
        axis_cols = [c for c in summary.columns if c.startswith("axis_")]
        assert axis_cols == []


class TestFewShotCollapseDetection:
    """Test collapse/few-shot collapse detection"""

    def test_detects_degradation(self):
        import pandas as pd

        data = [{
            "model_name": "bad-model",
            "task_id": "test_task",
            "category": "test",
            "score_0shot": 0.9,
            "score_8shot": 0.3,
        }]
        df = pd.DataFrame(data)
        alerts = detect_few_shot_collapse(df)
        assert len(alerts) == 1
        assert alerts[0]["model"] == "bad-model"
        assert alerts[0]["drop_pct"] > 50

    def test_no_degradation(self):
        import pandas as pd

        data = [{
            "model_name": "good-model",
            "task_id": "test_task",
            "category": "test",
            "score_0shot": 0.5,
            "score_8shot": 0.9,
        }]
        df = pd.DataFrame(data)
        alerts = detect_few_shot_collapse(df)
        assert len(alerts) == 0
