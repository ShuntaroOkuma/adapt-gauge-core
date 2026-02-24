"""
Evaluation Execution

Handles single evaluations through full evaluation runs, including result aggregation.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from adapt_gauge_core.domain.constants import DEFAULT_MODELS, SHOT_SCHEDULE
from adapt_gauge_core.domain.entities import EvaluationResult, AcquisitionMetrics
from adapt_gauge_core.domain.value_objects import CostMetrics
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient
from adapt_gauge_core.task_loader import Task, TestCase
from adapt_gauge_core.prompt_builder import build_prompt
from adapt_gauge_core.scoring.scorer import score
from adapt_gauge_core.efficiency_calc import (
    calculate_all_metrics,
    calculate_reliability_metrics,
    aggregate_trial_scores,
    score_variance,
)
from adapt_gauge_core.harness_config import HarnessConfig, load_config


def run_single_evaluation(
    task: Task,
    test_case: TestCase,
    model_client: ModelClient,
    shot_count: int,
    run_id: str,
    grader_client: ModelClient | None = None,
    trial_id: int = 1,
) -> EvaluationResult:
    """
    Execute a single evaluation.

    Args:
        task: Task definition
        test_case: Test case
        model_client: Model client
        shot_count: Number of shots
        run_id: Run ID
        grader_client: Client for LLM grader (used with llm_judge)
        trial_id: Trial ID (default: 1)

    Returns:
        EvaluationResult: Evaluation result
    """
    # Build prompt
    prompt = build_prompt(task, test_case.input, shot_count)

    # Call model
    response = model_client.generate(prompt)

    # Score
    eval_score = score(
        test_case.expected_output,
        response.output,
        test_case.scoring_method,
        grader_client=grader_client,
        acceptable_variations=test_case.acceptable_variations,
        input_text=test_case.input,
    )

    # Convert expected_output to JSON string if it's a dict (for CSV output)
    expected_output_str = (
        json.dumps(test_case.expected_output, ensure_ascii=False)
        if isinstance(test_case.expected_output, dict)
        else test_case.expected_output
    )

    return EvaluationResult(
        run_id=run_id,
        task_id=task.task_id,
        category=task.category,
        model_name=response.model_name,
        shot_count=shot_count,
        input=test_case.input,
        expected_output=expected_output_str,
        actual_output=response.output,
        score=eval_score.score if hasattr(eval_score, "score") else float(eval_score),
        scoring_method=test_case.scoring_method,
        latency_ms=response.latency_ms,
        timestamp=datetime.now().isoformat(),
        trial_id=trial_id,
    )


def _run_model_evaluations(
    model_name: str,
    tasks: list[Task],
    shots: list[int],
    run_id: str,
    progress: dict,
    lock: threading.Lock,
    grader_client: ModelClient | None = None,
) -> list[EvaluationResult]:
    """
    Execute all evaluations for a single model (internal function for parallel execution).

    Args:
        model_name: Model name
        tasks: List of tasks
        shots: List of shot counts
        run_id: Run ID
        progress: Shared dictionary for progress tracking
        lock: Lock for thread-safe progress updates
        grader_client: Client for LLM grader (used with llm_judge)

    Returns:
        list[EvaluationResult]: List of evaluation results
    """
    from adapt_gauge_core.model_client import create_client

    results = []
    client = create_client(model_name)

    for task in tasks:
        for shot in shots:
            for test_case in task.test_cases:
                with lock:
                    progress["current"] += 1
                    current = progress["current"]
                    total = progress["total"]
                    print(f"[{current}/{total}] {task.task_id} | {model_name} | {shot}-shot")

                result = run_single_evaluation(
                    task=task,
                    test_case=test_case,
                    model_client=client,
                    shot_count=shot,
                    run_id=run_id,
                    grader_client=grader_client,
                )
                results.append(result)

                with lock:
                    print(f"  Score: {result.score:.2f} | Latency: {result.latency_ms}ms")

    return results


def run_all_evaluations(
    tasks: list[Task],
    models: list[str] = DEFAULT_MODELS,
    shots: list[int] = SHOT_SCHEDULE,
) -> list[EvaluationResult]:
    """
    Execute evaluations for all combinations (parallel execution per model).

    Args:
        tasks: List of tasks
        models: List of model names
        shots: List of shot counts

    Returns:
        list[EvaluationResult]: All evaluation results
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = sum(len(task.test_cases) for task in tasks) * len(models) * len(shots)

    # Shared state for progress tracking
    progress = {"current": 0, "total": total}
    lock = threading.Lock()

    all_results = []

    # Parallel execution per model
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                _run_model_evaluations,
                model_name,
                tasks,
                shots,
                run_id,
                progress,
                lock
            ): model_name
            for model_name in models
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Error running model {model_name}: {e}")

    return all_results


def aggregate_results(
    results: list[EvaluationResult],
    tasks: list[Task] | None = None,
    config: HarnessConfig | None = None,
) -> pd.DataFrame:
    """
    Aggregate by task x model and calculate efficiency metrics.

    When there are multiple trials:
      1. Average test case scores within each shot x trial
      2. Aggregate across trials (mean/median) to get the final score
      3. Calculate inter-trial variance and pass@k/pass^k

    Args:
        results: List of evaluation results
        tasks: List of task definitions (reserved for future use)
        config: HarnessConfig (loads from env if not provided)

    Returns:
        pd.DataFrame: Aggregated results
    """
    if config is None:
        config = load_config()

    df = pd.DataFrame([asdict(r) for r in results])

    # Add trial_id column for backward compatibility if missing
    if "trial_id" not in df.columns:
        df["trial_id"] = 1

    num_trials = df["trial_id"].nunique()

    # Group by task x model
    grouped = df.groupby(["run_id", "task_id", "category", "model_name"])

    summary_rows = []
    for (run_id, task_id, category, model_name), group in grouped:
        # Calculate scores per shot count, keeping per-trial means
        scores = {}
        shot_trial_means: dict[int, list[float]] = {}
        shot_variances = []

        for shot in SHOT_SCHEDULE:
            shot_data = group[group["shot_count"] == shot]
            if len(shot_data) == 0:
                scores[shot] = 0.0
                continue

            if num_trials > 1:
                trial_means = []
                for trial_id in sorted(shot_data["trial_id"].unique()):
                    trial_shot_data = shot_data[shot_data["trial_id"] == trial_id]
                    trial_means.append(trial_shot_data["score"].mean())

                shot_trial_means[shot] = trial_means
                scores[shot] = aggregate_trial_scores(
                    trial_means, config.trials.aggregation
                )
                shot_variances.append(score_variance(trial_means))
            else:
                scores[shot] = shot_data["score"].mean()

        # Calculate efficiency metrics
        metrics = calculate_all_metrics(scores, config.trials.success_threshold)

        # Aggregate cost metrics (total across all shot counts)
        total_input_tokens = int(group["input_tokens"].sum())
        total_output_tokens = int(group["output_tokens"].sum())
        total_latency_ms = int(group["latency_ms"].sum())

        row = {
            "run_id": run_id,
            "task_id": task_id,
            "category": category,
            "model_name": model_name,
            "score_0shot": scores.get(0, 0),
            "score_1shot": scores.get(1, 0),
            "score_2shot": scores.get(2, 0),
            "score_4shot": scores.get(4, 0),
            "score_8shot": scores.get(8, 0),
            "improvement_rate": metrics["improvement_rate"],
            "threshold_shots": metrics["threshold_shots"],
            "learning_curve_auc": metrics["learning_curve_auc"],
            "num_trials": num_trials,
            # Cost metrics
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_latency_ms": total_latency_ms,
        }

        # Calculate reliability metrics (when there are multiple trials)
        if num_trials > 1 and config.reliability.calculate_pass_at_k:
            shot_reliability = {}
            for shot, trial_means in shot_trial_means.items():
                shot_reliability[shot] = calculate_reliability_metrics(
                    trial_means,
                    config.trials.success_threshold,
                    config.reliability.k_values,
                )

            avg_variance = (
                sum(shot_variances) / len(shot_variances) if shot_variances else 0.0
            )
            row["score_variance"] = avg_variance

            for k in config.reliability.k_values:
                pass_at_k_vals = [
                    sr["pass_at_k"].get(k, 0.0) for sr in shot_reliability.values()
                ]
                pass_all_k_vals = [
                    sr["pass_all_k"].get(k, 0.0) for sr in shot_reliability.values()
                ]
                row[f"pass_at_{k}"] = (
                    sum(pass_at_k_vals) / len(pass_at_k_vals) if pass_at_k_vals else 0.0
                )
                row[f"pass_all_{k}"] = (
                    sum(pass_all_k_vals) / len(pass_all_k_vals) if pass_all_k_vals else 0.0
                )
        else:
            row["score_variance"] = 0.0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
