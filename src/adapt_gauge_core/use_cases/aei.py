"""
AEI (Adaptation Efficiency Index) Calculation

Integrates 6-axis evaluation scores into a single metric and detects negative learning.
"""

import pandas as pd
from typing import Callable

from adapt_gauge_core.domain.constants import SHOT_SCHEDULE


def compute_aei(df: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate the Adaptation Efficiency Index (AEI).

    Equal-weighted average of the 6-axis scores. Returns None if no axis data exists.

    Args:
        df: Aggregated results DataFrame (containing axis_* columns)

    Returns:
        DataFrame containing AEI scores. None if no axis data is available.
    """
    axis_keys = [
        "axis_Acquisition", "axis_Resilience_Noise", "axis_Resilience_Detect",
        "axis_Efficiency", "axis_Agency", "axis_Fidelity",
    ]
    axis_labels = ["Acquisition", "Resilience-Noise", "Resilience-Detect",
                   "Efficiency", "Agency", "Fidelity"]

    available = [(k, l) for k, l in zip(axis_keys, axis_labels)
                 if k in df.columns and df[k].notna().any()]
    if not available:
        return None

    available_keys = [k for k, _ in available]
    rename_map = {k: l for k, l in available}

    model_means = df.groupby("model_name")[available_keys].mean()
    model_means["AEI"] = model_means[available_keys].mean(axis=1)
    aei_df = model_means.rename(columns=rename_map).reset_index()
    return aei_df.sort_values("AEI", ascending=False)


def _get_shot_scores(row: pd.Series) -> list[tuple[int, float]]:
    """Extract available (shot_count, score) pairs from a summary row."""
    pairs = []
    for shot in SHOT_SCHEDULE:
        col = f"score_{shot}shot"
        if col in row.index and pd.notna(row[col]):
            pairs.append((shot, float(row[col])))
    return pairs


def _get_final_score_col(df: pd.DataFrame) -> str | None:
    """Determine the final shot column available in the DataFrame."""
    for shot in reversed(SHOT_SCHEDULE):
        col = f"score_{shot}shot"
        if col in df.columns:
            return col
    return None


def _make_label_fn(
    label_fn: Callable[[str, str], str] | None,
) -> Callable[[str, str], str]:
    """Return the label function, defaulting to identity on task_id."""
    if label_fn is None:
        return lambda tid, cat: tid
    return label_fn


_MIN_BASELINE = 0.05

# Peak regression thresholds (shared by detect_peak_regression & classify)
_LEARNING_THRESHOLD = 1.1    # Peak must be 10% above 0-shot
_REGRESSION_THRESHOLD = 0.8  # Final must drop below 80% of peak


def _is_peak_regression(
    shot_scores: list[tuple[int, float]],
    score_0: float,
) -> tuple[bool, int, float]:
    """Check whether a shot-score series shows peak regression.

    Returns:
        (is_peak_regression, peak_shot, score_peak)
    """
    if len(shot_scores) < 2 or score_0 <= _MIN_BASELINE:
        return False, 0, 0.0

    peak_shot, score_peak = max(shot_scores[1:], key=lambda x: x[1])
    final_shot, score_final = shot_scores[-1]

    if peak_shot == final_shot:
        return False, peak_shot, score_peak

    if (
        score_peak > score_0 * _LEARNING_THRESHOLD
        and score_final < score_peak * _REGRESSION_THRESHOLD
    ):
        return True, peak_shot, score_peak

    return False, peak_shot, score_peak


def detect_negative_learning(
    df: pd.DataFrame,
    label_fn: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Detect negative learning (performance degradation when examples are added).

    Returns cases where the final shot score has degraded by 10% or more
    compared to the 0-shot score.

    Severity levels:
        - "degradation": 10-50% drop
        - "collapse": 50%+ drop

    Args:
        df: Aggregated results DataFrame
        label_fn: Callback of the form (task_id, category) -> display_label.
                  If None, task_id is returned as-is.

    Returns:
        List of detection results. Each element is a dict containing model,
        task_id, task_label, score_0shot, score_final, drop_pct, severity, and type.
    """
    fn = _make_label_fn(label_fn)
    final_col = _get_final_score_col(df)
    if final_col is None:
        return []
    threshold = 0.9  # Detect when score drops below 90% of the 0-shot score

    alerts: list[dict] = []
    for _, row in df.iterrows():
        score_0 = row.get("score_0shot", 0)
        score_final = row.get(final_col, 0)
        if score_0 > _MIN_BASELINE and score_final < score_0 * threshold:
            drop_pct = (score_0 - score_final) / score_0 * 100
            severity = "collapse" if drop_pct >= 50 else "degradation"
            task_label = fn(
                row.get("task_id", ""), row.get("category", "")
            )
            alerts.append({
                "type": "negative_learning",
                "severity": severity,
                "model": row["model_name"],
                "task_id": row.get("task_id", ""),
                "task_label": task_label,
                "score_0shot": score_0,
                "score_final": score_final,
                "drop_pct": drop_pct,
            })
    alerts.sort(key=lambda x: x["drop_pct"], reverse=True)
    return alerts


def detect_peak_regression(
    df: pd.DataFrame,
    label_fn: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Detect peak regression (model learned then forgot).

    Returns cases where the model's score peaked at an intermediate shot count
    but then dropped by 20% or more at the final shot.

    Conditions:
        - Peak score must be > 10% above the 0-shot score (evidence of learning)
        - Final score must be < 80% of the peak score (significant regression)

    Args:
        df: Aggregated results DataFrame
        label_fn: Callback of the form (task_id, category) -> display_label.
                  If None, task_id is returned as-is.

    Returns:
        List of detection results. Each element is a dict containing model,
        task_id, task_label, score_peak, peak_shot, score_final, drop_pct, and type.
    """
    fn = _make_label_fn(label_fn)

    alerts: list[dict] = []
    for _, row in df.iterrows():
        shot_scores = _get_shot_scores(row)
        score_0 = shot_scores[0][1] if shot_scores else 0.0

        is_regression, peak_shot, score_peak = _is_peak_regression(
            shot_scores, score_0,
        )
        if not is_regression:
            continue

        score_final = shot_scores[-1][1]
        drop_pct = (score_peak - score_final) / score_peak * 100
        task_label = fn(
            row.get("task_id", ""), row.get("category", "")
        )
        alerts.append({
            "type": "peak_regression",
            "model": row["model_name"],
            "task_id": row.get("task_id", ""),
            "task_label": task_label,
            "score_peak": score_peak,
            "peak_shot": peak_shot,
            "score_final": score_final,
            "drop_pct": drop_pct,
        })
    alerts.sort(key=lambda x: x["drop_pct"], reverse=True)
    return alerts


def detect_mid_curve_dip(
    df: pd.DataFrame,
    label_fn: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Detect mid-curve dip (sharp score drop between adjacent shots).

    Returns cases where the score drops by 30% or more between consecutive
    shot counts, indicating instability in the learning process.

    Only the largest dip per model-task pair is reported.

    Args:
        df: Aggregated results DataFrame
        label_fn: Callback of the form (task_id, category) -> display_label.
                  If None, task_id is returned as-is.

    Returns:
        List of detection results. Each element is a dict containing model,
        task_id, task_label, from_shot, to_shot, score_from, score_to,
        drop_pct, and type.
    """
    fn = _make_label_fn(label_fn)
    dip_threshold = 0.7  # Score must drop below 70% of previous shot (30% drop)

    alerts: list[dict] = []
    for _, row in df.iterrows():
        shot_scores = _get_shot_scores(row)
        if len(shot_scores) < 2:
            continue

        # Find the largest adjacent drop for this row
        largest_dip: dict | None = None
        for i in range(1, len(shot_scores)):
            prev_shot, prev_score = shot_scores[i - 1]
            curr_shot, curr_score = shot_scores[i]

            if prev_score <= _MIN_BASELINE:
                continue

            if curr_score < prev_score * dip_threshold:
                drop_pct = (prev_score - curr_score) / prev_score * 100
                if largest_dip is None or drop_pct > largest_dip["drop_pct"]:
                    task_label = fn(
                        row.get("task_id", ""), row.get("category", "")
                    )
                    largest_dip = {
                        "type": "mid_curve_dip",
                        "model": row["model_name"],
                        "task_id": row.get("task_id", ""),
                        "task_label": task_label,
                        "from_shot": prev_shot,
                        "to_shot": curr_shot,
                        "score_from": prev_score,
                        "score_to": curr_score,
                        "drop_pct": drop_pct,
                    }

        if largest_dip is not None:
            alerts.append(largest_dip)

    alerts.sort(key=lambda x: x["drop_pct"], reverse=True)
    return alerts


# ---------------------------------------------------------------------------
# Collapse pattern classification
# ---------------------------------------------------------------------------

_DECLINE_THRESHOLD = 0.10  # 10% overall drop to be considered a decline
_IMMEDIATE_DROP_RATIO = 0.6  # First drop accounts for >=60% of total drop


def classify_collapse_pattern(df: pd.DataFrame) -> list[dict]:
    """Classify collapse pattern for each model-task combination.

    Pattern types:
        - "stable": No significant degradation (monotonic increase, flat, or <10% drop)
        - "immediate_collapse": Sharp drop right after 0-shot that persists
        - "gradual_decline": Steady decrease across shot counts
        - "peak_regression": Score improves then regresses

    Args:
        df: Summary DataFrame with score_Nshot columns.

    Returns:
        List of dicts with model, task_id, pattern, and scores.
    """
    results: list[dict] = []
    for _, row in df.iterrows():
        shot_scores = _get_shot_scores(row)
        if len(shot_scores) < 2:
            continue

        model = row["model_name"]
        task_id = row.get("task_id", "")
        scores_dict = {s: v for s, v in shot_scores}
        score_0 = shot_scores[0][1]
        score_final = shot_scores[-1][1]

        # Classify pattern
        is_regression, _, _ = _is_peak_regression(shot_scores, score_0)

        if score_0 <= _MIN_BASELINE:
            pattern = "stable"
        elif is_regression:
            pattern = "peak_regression"
        elif (score_0 - score_final) / score_0 < _DECLINE_THRESHOLD:
            pattern = "stable"
        else:
            first_drop = score_0 - shot_scores[1][1]
            total_drop = score_0 - score_final
            if total_drop > 0 and first_drop / total_drop >= _IMMEDIATE_DROP_RATIO:
                pattern = "immediate_collapse"
            else:
                pattern = "gradual_decline"

        results.append({
            "model": model,
            "task_id": task_id,
            "scores": scores_dict,
            "pattern": pattern,
        })

    return results


# ---------------------------------------------------------------------------
# Collapse resilience score
# ---------------------------------------------------------------------------

_PATTERN_PENALTY = {
    "stable": 0.0,
    "gradual_decline": 0.5,
    "peak_regression": 0.6,
    "immediate_collapse": 1.0,
}


def calculate_resilience_score(
    df: pd.DataFrame,
    classifications: list[dict] | None = None,
) -> dict[str, float]:
    """Calculate collapse resilience score per model.

    Scores range from 0.0 (always collapses) to 1.0 (fully stable).

    The score combines:
        - Pattern type penalty (stable=0, immediate_collapse=1)
        - Magnitude of the actual drop when collapse occurs

    Args:
        df: Summary DataFrame with score_Nshot columns.
        classifications: Pre-computed classifications to avoid redundant
            computation. If None, classify_collapse_pattern(df) is called.

    Returns:
        Dict mapping model_name to resilience score (0-1).
    """
    if classifications is None:
        classifications = classify_collapse_pattern(df)
    if not classifications:
        return {}

    model_scores: dict[str, list[float]] = {}
    for entry in classifications:
        model = entry["model"]
        pattern = entry["pattern"]
        scores = entry["scores"]

        shot_list = sorted(scores.keys())
        if len(shot_list) < 2:
            continue

        score_0 = scores[shot_list[0]]
        score_final = scores[shot_list[-1]]

        # Base penalty from pattern type
        base_penalty = _PATTERN_PENALTY.get(pattern, 0.0)

        # Scale by actual drop magnitude (0-1)
        if pattern == "peak_regression":
            # For peak regression, measure drop from peak (not from 0-shot)
            score_peak = max(scores[s] for s in shot_list[1:])
            drop_ratio = max(0.0, (score_peak - score_final) / score_peak)
            penalty = base_penalty * min(1.0, drop_ratio)
        elif score_0 > _MIN_BASELINE and pattern != "stable":
            drop_ratio = max(0.0, (score_0 - score_final) / score_0)
            penalty = base_penalty * min(1.0, drop_ratio)
        else:
            penalty = 0.0

        resilience = 1.0 - penalty
        model_scores.setdefault(model, []).append(resilience)

    return {
        model: sum(vals) / len(vals)
        for model, vals in model_scores.items()
    }
