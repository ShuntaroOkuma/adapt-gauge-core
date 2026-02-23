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
from adapt_gauge_core.efficiency_calc import calculate_all_metrics


def run_single_evaluation(
    task: Task,
    test_case: TestCase,
    model_client: ModelClient,
    shot_count: int,
    run_id: str,
    grader_client: ModelClient | None = None,
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
        score=eval_score,
        scoring_method=test_case.scoring_method,
        latency_ms=response.latency_ms,
        timestamp=datetime.now().isoformat()
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
) -> pd.DataFrame:
    """
    Aggregate by task x model and calculate efficiency metrics.

    Args:
        results: List of evaluation results
        tasks: List of task definitions (reserved for future use)

    Returns:
        pd.DataFrame: Aggregated results
    """
    df = pd.DataFrame([asdict(r) for r in results])

    # Group by task x model
    grouped = df.groupby(["run_id", "task_id", "category", "model_name"])

    summary_rows = []
    for (run_id, task_id, category, model_name), group in grouped:
        # Calculate mean score per shot count (supports multiple test cases)
        shot_scores_mean = group.groupby("shot_count")["score"].mean()
        scores = {shot: shot_scores_mean.get(shot, 0.0) for shot in SHOT_SCHEDULE}

        # Calculate efficiency metrics
        metrics = calculate_all_metrics(scores)

        # Aggregate cost metrics (total across all shot counts)
        total_input_tokens = int(group["input_tokens"].sum())
        total_output_tokens = int(group["output_tokens"].sum())
        total_latency_ms = int(group["latency_ms"].sum())

        summary_rows.append({
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
            # Cost metrics
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_latency_ms": total_latency_ms,
        })

    return pd.DataFrame(summary_rows)
