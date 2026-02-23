"""
Adaptation Efficiency Metrics Calculation

Computes improvement rate per example, threshold-reaching shot count, learning curve AUC,
economy score (accuracy/cost), and reliability metrics.
"""

from math import comb
from typing import Optional

import pandas as pd

from adapt_gauge_core.domain.value_objects import CostMetrics


def calculate_total_cost(metrics: CostMetrics) -> float:
    """
    Calculate total cost

    Total cost = token cost + time cost

    Args:
        metrics: CostMetrics instance

    Returns:
        Total cost (USD)
    """
    token_cost = (
        (metrics.input_tokens / 1_000_000) * metrics.input_price_per_m +
        (metrics.output_tokens / 1_000_000) * metrics.output_price_per_m
    )
    time_cost = (metrics.latency_ms / 1000) * metrics.time_price_per_sec
    return token_cost + time_cost


def calculate_economy_score(accuracy: float, total_cost: float) -> float:
    """
    Calculate economy score (Efficiency axis in the 6-axis evaluation)

    Economy = accuracy / total cost

    Args:
        accuracy: Accuracy (0.0 to 1.0)
        total_cost: Total cost (USD)

    Returns:
        Economy score (accuracy per dollar). Returns 0 if cost is 0.
    """
    if total_cost <= 0:
        return 0.0
    return accuracy / total_cost


def improvement_rate(scores: dict[int, float]) -> float:
    """
    Calculate improvement rate per example

    Args:
        scores: {shot count: score} dictionary (e.g., {0: 0.3, 1: 0.5, 2: 0.7, 4: 0.85, 8: 0.95})

    Returns:
        Improvement rate (difference between 0-shot and 8-shot / 8)
    """
    return (scores[8] - scores[0]) / 8


def threshold_shots(scores: dict[int, float], threshold: float = 0.8) -> int:
    """
    Minimum number of shots required to reach the threshold

    Args:
        scores: {shot count: score} dictionary
        threshold: Threshold value (default 0.8)

    Returns:
        Minimum shot count that reached the threshold, or -1 if not reached
    """
    shot_schedule = [0, 1, 2, 4, 8]

    for shot in shot_schedule:
        if shot in scores and scores[shot] >= threshold:
            return shot

    return -1


def learning_curve_auc(scores: dict[int, float]) -> float:
    """
    Learning curve AUC using the trapezoidal rule

    Calculates the area using the trapezoidal rule with shot count as the x-axis
    and score as the y-axis.
    Normalized over the x-axis range [0, 1, 2, 4, 8] (total width = 8).

    Args:
        scores: {shot count: score} dictionary

    Returns:
        AUC value (0.0 to 1.0, normalized)
    """
    shot_schedule = [0, 1, 2, 4, 8]

    # Verify that scores for all shot counts are available
    for shot in shot_schedule:
        if shot not in scores:
            raise ValueError(f"Score for shot count {shot} is missing")

    # Calculate area using the trapezoidal rule
    # Intervals: [0,1], [1,2], [2,4], [4,8]
    area = 0.0

    # [0, 1] interval (width = 1)
    area += (scores[0] + scores[1]) / 2 * 1

    # [1, 2] interval (width = 1)
    area += (scores[1] + scores[2]) / 2 * 1

    # [2, 4] interval (width = 2)
    area += (scores[2] + scores[4]) / 2 * 2

    # [4, 8] interval (width = 4)
    area += (scores[4] + scores[8]) / 2 * 4

    # Normalize by the total x-axis width (8)
    normalized_auc = area / 8

    return normalized_auc


def calculate_all_metrics(scores: dict[int, float], threshold: float = 0.8) -> dict:
    """
    Calculate all efficiency metrics

    Args:
        scores: {shot count: score} dictionary
        threshold: Threshold for threshold-reaching determination (default 0.8)

    Returns:
        {
            "improvement_rate": float,
            "threshold_shots": int,
            "learning_curve_auc": float
        }
    """
    return {
        "improvement_rate": improvement_rate(scores),
        "threshold_shots": threshold_shots(scores, threshold),
        "learning_curve_auc": learning_curve_auc(scores)
    }


# --- Reliability Metrics ---


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    pass@k: Probability of at least one success in k trials

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of trials
        c: Number of successes
        k: Number of samples drawn

    Returns:
        Probability (0.0 to 1.0)
    """
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if c >= n:
        return 1.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_all_k(n: int, c: int, k: int) -> float:
    """
    pass^k: Probability of all k trials succeeding

    pass^k = C(c, k) / C(n, k)

    Args:
        n: Total number of trials
        c: Number of successes
        k: Number of samples drawn

    Returns:
        Probability (0.0 to 1.0)
    """
    if n <= 0 or k <= 0 or k > n:
        return 0.0
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def score_variance(scores: list[float]) -> float:
    """
    Calculate score variance (population variance)

    Args:
        scores: List of trial scores

    Returns:
        Variance
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def aggregate_trial_scores(scores: list[float], method: str = "mean") -> float:
    """
    Aggregate trial scores

    Args:
        scores: List of trial scores
        method: Aggregation method ("mean" or "median")

    Returns:
        Aggregated score
    """
    if not scores:
        return 0.0

    if method == "median":
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        if n % 2 == 1:
            return sorted_scores[n // 2]
        return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    return sum(scores) / len(scores)


def calculate_reliability_metrics(
    trial_scores: list[float],
    threshold: float = 0.8,
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Calculate reliability metrics

    Args:
        trial_scores: List of scores for each trial
        threshold: Threshold for success determination
        k_values: List of k values for pass@k/pass^k calculation

    Returns:
        {
            "num_trials": int,
            "score_mean": float,
            "score_variance": float,
            "pass_at_k": {k: float},
            "pass_all_k": {k: float},
        }
    """
    if k_values is None:
        k_values = [1, 3]

    n = len(trial_scores)
    c = sum(1 for s in trial_scores if s >= threshold)

    pass_at_k_results = {}
    pass_all_k_results = {}
    for k in k_values:
        pass_at_k_results[k] = pass_at_k(n, c, k)
        pass_all_k_results[k] = pass_all_k(n, c, k)

    return {
        "num_trials": n,
        "score_mean": aggregate_trial_scores(trial_scores, "mean"),
        "score_variance": score_variance(trial_scores),
        "pass_at_k": pass_at_k_results,
        "pass_all_k": pass_all_k_results,
    }


def compute_trial_std(
    raw_df: pd.DataFrame,
    task_id: str,
    shot_schedule: list[int],
) -> Optional[pd.DataFrame]:
    """
    Calculate the standard deviation of scores across trials from raw_df

    For each (model, shot), compute the mean score per trial across test cases,
    then return the standard deviation across trials.

    Args:
        raw_df: Raw results DataFrame (requires trial_id, task_id, model_name, shot_count, score columns)
        task_id: Target task ID
        shot_schedule: List of shot counts to compute

    Returns:
        DataFrame with model_name, shot_count, std columns. None if computation is not possible.
    """
    if raw_df is None or "trial_id" not in raw_df.columns:
        return None
    if raw_df["trial_id"].nunique() <= 1:
        return None

    task_raw = raw_df[raw_df["task_id"] == task_id]
    if task_raw.empty:
        return None

    std_rows = []
    for model in task_raw["model_name"].unique():
        model_data = task_raw[task_raw["model_name"] == model]
        for shot in shot_schedule:
            shot_data = model_data[model_data["shot_count"] == shot]
            if shot_data.empty:
                continue
            trial_means = shot_data.groupby("trial_id")["score"].mean()
            std_rows.append({
                "model_name": model,
                "shot_count": shot,
                "std": float(trial_means.std()),
            })

    return pd.DataFrame(std_rows) if std_rows else None
