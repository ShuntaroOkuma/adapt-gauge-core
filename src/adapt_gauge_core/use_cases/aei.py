"""
AEI (Adaptation Efficiency Index) Calculation

Integrates 6-axis evaluation scores into a single metric and detects negative learning.
"""

import pandas as pd
from typing import Callable


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


def detect_negative_learning(
    df: pd.DataFrame,
    label_fn: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Detect negative learning (performance degradation when examples are added).

    Returns cases where the final shot (8-shot) score has degraded by 20% or more
    compared to the 0-shot score.

    Args:
        df: Aggregated results DataFrame
        label_fn: Callback of the form (task_id, category) -> display_label.
                  If None, task_id is returned as-is.

    Returns:
        List of detection results. Each element is a dict containing model,
        task_id, task_label, score_0shot, score_final, and drop_pct.
    """
    if label_fn is None:
        label_fn = lambda tid, cat: tid

    has_8shot = "score_8shot" in df.columns
    max_shot_col = "score_8shot" if has_8shot else "score_4shot"
    threshold = 0.8  # Detect when score drops below 80% of the 0-shot score

    alerts: list[dict] = []
    for _, row in df.iterrows():
        score_0 = row.get("score_0shot", 0)
        score_max = row.get(max_shot_col, 0)
        if score_0 > 0.05 and score_max < score_0 * threshold:
            drop_pct = (score_0 - score_max) / score_0 * 100
            task_label = label_fn(
                row.get("task_id", ""), row.get("category", "")
            )
            alerts.append({
                "model": row["model_name"],
                "task_id": row.get("task_id", ""),
                "task_label": task_label,
                "score_0shot": score_0,
                "score_final": score_max,
                "drop_pct": drop_pct,
            })
    # Sort by largest degradation first
    alerts.sort(key=lambda x: x["drop_pct"], reverse=True)
    return alerts
