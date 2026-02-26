"""
adapt-gauge-core Result Viewer

Minimal Streamlit dashboard for viewing evaluation results.
Displays learning curves and collapse (negative learning) detection.

Usage:
    pip install -e ".[viewer]"
    streamlit run src/adapt_gauge_core/viewer.py
    streamlit run src/adapt_gauge_core/viewer.py -- --results-dir results

Requires the package to be installed (e.g. via ``pip install -e .``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adapt_gauge_core.domain.constants import SHOT_SCHEDULE
from adapt_gauge_core.use_cases.aei import (
    detect_negative_learning,
    detect_peak_regression,
    detect_mid_curve_dip,
    classify_collapse_pattern,
    calculate_resilience_score,
)

# -- Colors --
MODEL_COLORS = [
    "#1a73e8", "#e8710a", "#34a853", "#ea4335", "#9334e6",
    "#f538a0", "#00897b", "#6d4c41", "#546e7a", "#d500f9",
]

SHOT_LABELS = {s: f"{s}-shot" for s in SHOT_SCHEDULE}
NEGATIVE_LEARNING_THRESHOLD = 0.02


def _short_model_name(name: str) -> str:
    """Shorten model name for display."""
    parts = name.split("/")
    return parts[-1] if len(parts) > 1 else name


def _find_result_pairs(results_dir: Path) -> list[dict]:
    """Find matching raw_results / summary CSV pairs in results_dir."""
    pairs = []
    for raw_path in sorted(results_dir.glob("raw_results_*.csv"), reverse=True):
        run_id = raw_path.stem.replace("raw_results_", "")
        summary_path = results_dir / f"summary_{run_id}.csv"
        pairs.append({
            "run_id": run_id,
            "raw_path": raw_path,
            "summary_path": summary_path if summary_path.exists() else None,
        })
    return pairs


def _load_data(pair: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load summary and raw DataFrames from a result pair."""
    raw_df = pd.read_csv(pair["raw_path"])
    summary_df = pd.read_csv(pair["summary_path"]) if pair["summary_path"] else None
    return raw_df, summary_df


def _render_learning_curve(summary_df: pd.DataFrame) -> None:
    """Render learning curve charts per task."""
    st.header("Learning Curves")

    tasks = summary_df["task_id"].unique()
    selected_task = st.radio(
        "Task",
        options=tasks,
        horizontal=True,
        label_visibility="collapsed",
    )

    filtered = summary_df[summary_df["task_id"] == selected_task]

    # Build trace list: disambiguate by selection method when multiple exist
    has_multi_sel = (
        "example_selection" in filtered.columns
        and filtered["example_selection"].nunique() > 1
    )

    traces = []
    for _, row in filtered.iterrows():
        sel = row.get("example_selection", "fixed") if has_multi_sel else None
        traces.append((row["model_name"], sel, row))

    fig = go.Figure()

    for i, (model, sel, row) in enumerate(traces):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        short_name = _short_model_name(model)
        if sel:
            short_name = f"{short_name} [{sel}]"

        scores = []
        shots = []
        for shot in SHOT_SCHEDULE:
            col = f"score_{shot}shot"
            if col in row.index:
                shots.append(shot)
                scores.append(row[col])

        fig.add_trace(go.Scatter(
            x=shots,
            y=scores,
            mode="lines+markers",
            name=short_name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8),
        ))

        # Highlight negative learning intervals
        for j in range(1, len(scores)):
            if scores[j] < scores[j - 1] - NEGATIVE_LEARNING_THRESHOLD:
                fig.add_trace(go.Scatter(
                    x=[shots[j - 1], shots[j]],
                    y=[scores[j - 1], scores[j]],
                    mode="lines",
                    line=dict(color="#ea4335", width=3, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # 80% target line
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="#5f6368",
        line_width=1,
        annotation_text="80% target",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#5f6368"),
    )

    fig.update_layout(
        title=f"Learning Curve: {selected_task}",
        xaxis_title="Shot count",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        xaxis_tickvals=SHOT_SCHEDULE,
        xaxis_ticktext=[SHOT_LABELS[s] for s in SHOT_SCHEDULE],
        legend_title="Model",
        template="plotly_white",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


def _alert_model_label(alert: dict, multi_sel: bool) -> str:
    """Build a display label for alert model name, including selection method if needed."""
    short = _short_model_name(alert["model"])
    if multi_sel:
        sel = alert.get("example_selection", "fixed")
        return f"{short} [{sel}]"
    return short


def _render_collapse_detection(summary_df: pd.DataFrame) -> None:
    """Render collapse / negative learning warnings for all 3 detection types."""
    st.header("Collapse Detection")

    multi_sel = (
        "example_selection" in summary_df.columns
        and summary_df["example_selection"].nunique() > 1
    )

    neg_alerts = detect_negative_learning(summary_df)
    peak_alerts = detect_peak_regression(summary_df)
    dip_alerts = detect_mid_curve_dip(summary_df)

    has_any = neg_alerts or peak_alerts or dip_alerts

    if not has_any:
        st.success("No negative learning detected.")
        return

    if neg_alerts:
        st.subheader("Negative Learning")
        st.caption("Final score is worse than 0-shot baseline.")
        for alert in neg_alerts:
            label = _alert_model_label(alert, multi_sel)
            severity = alert["severity"]
            fn = st.error if severity == "collapse" else st.warning
            fn(
                f"**{label}** on `{alert['task_id']}`: "
                f"0-shot {alert['score_0shot']:.1%} -> final {alert['score_final']:.1%} "
                f"({severity}, drop {alert['drop_pct']:.1f}%)"
            )

    if peak_alerts:
        st.subheader("Peak Regression")
        st.caption("Model learned at an intermediate shot count, then regressed.")
        for alert in peak_alerts:
            label = _alert_model_label(alert, multi_sel)
            st.warning(
                f"**{label}** on `{alert['task_id']}`: "
                f"peak {alert['score_peak']:.1%} at {alert['peak_shot']}-shot "
                f"-> final {alert['score_final']:.1%} "
                f"(drop {alert['drop_pct']:.1f}%)"
            )

    if dip_alerts:
        st.subheader("Mid-curve Dip")
        st.caption("Sharp score drop between adjacent shot counts (instability).")
        for alert in dip_alerts:
            label = _alert_model_label(alert, multi_sel)
            st.warning(
                f"**{label}** on `{alert['task_id']}`: "
                f"{alert['from_shot']}-shot {alert['score_from']:.1%} "
                f"-> {alert['to_shot']}-shot {alert['score_to']:.1%} "
                f"(drop {alert['drop_pct']:.1f}%)"
            )

    # Collapse pattern classification
    classifications = classify_collapse_pattern(summary_df)
    if classifications:
        st.subheader("Collapse Pattern Classification")
        st.caption(
            "Each model-task pair classified as: stable, immediate_collapse, "
            "gradual_decline, or peak_regression."
        )
        pattern_df = pd.DataFrame(classifications)
        pattern_df["model"] = pattern_df["model"].apply(_short_model_name)
        display_cols = ["model", "task_id", "pattern"]
        rename_map = {"model": "Model", "task_id": "Task", "pattern": "Pattern"}
        if multi_sel and "example_selection" in pattern_df.columns:
            display_cols.insert(2, "example_selection")
            rename_map["example_selection"] = "Selection"
        st.dataframe(
            pattern_df[display_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )

    # Resilience score — compute per selection method when multiple exist
    res_df = None
    if multi_sel and classifications:
        sel_groups: dict[str, list[dict]] = {}
        for c in classifications:
            sel = c.get("example_selection", "fixed")
            sel_groups.setdefault(sel, []).append(c)

        all_res_rows = []
        for sel, sel_cls in sorted(sel_groups.items()):
            resilience = calculate_resilience_score(summary_df, classifications=sel_cls)
            for m, s in sorted(resilience.items(), key=lambda x: -x[1]):
                all_res_rows.append({
                    "Model": _short_model_name(m),
                    "Selection": sel,
                    "Resilience Score": f"{s:.3f}",
                })
        if all_res_rows:
            res_df = pd.DataFrame(all_res_rows)
    else:
        resilience = calculate_resilience_score(summary_df, classifications=classifications)
        if resilience:
            res_df = pd.DataFrame([
                {"Model": _short_model_name(m), "Resilience Score": f"{s:.3f}"}
                for m, s in sorted(resilience.items(), key=lambda x: -x[1])
            ])

    if res_df is not None and not res_df.empty:
        st.subheader("Collapse Resilience Score")
        st.caption("0.0 = always collapses, 1.0 = fully stable.")
        st.dataframe(res_df, use_container_width=True, hide_index=True)


def _render_metrics_table(summary_df: pd.DataFrame) -> None:
    """Render a summary metrics table."""
    st.header("Metrics Summary")

    display_cols = ["model_name", "task_id", "category"]
    if "example_selection" in summary_df.columns and summary_df["example_selection"].nunique() > 1:
        display_cols.append("example_selection")
    score_cols = [f"score_{s}shot" for s in SHOT_SCHEDULE if f"score_{s}shot" in summary_df.columns]
    metric_cols = ["improvement_rate", "threshold_shots", "learning_curve_auc"]
    optional_cols = [c for c in ["num_trials", "score_variance"] if c in summary_df.columns]
    pass_cols = sorted([c for c in summary_df.columns if c.startswith("pass_")])

    all_cols = display_cols + score_cols + metric_cols + optional_cols + pass_cols
    existing = [c for c in all_cols if c in summary_df.columns]

    styled = summary_df[existing].copy()
    styled["model_name"] = styled["model_name"].apply(_short_model_name)
    styled = styled.rename(columns={"model_name": "Model", "task_id": "Task", "category": "Category"})

    st.dataframe(styled, use_container_width=True, hide_index=True)


def main() -> None:
    # Parse --results-dir from Streamlit args (after --)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args, _ = parser.parse_known_args()

    results_dir = Path(args.results_dir)

    st.set_page_config(page_title="adapt-gauge-core", layout="wide")
    st.title("adapt-gauge-core Results")

    # Find result files
    if not results_dir.exists():
        st.error(f"Results directory not found: `{results_dir}`")
        st.info("Run an evaluation first:\n```\npython -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json\n```")
        return

    pairs = _find_result_pairs(results_dir)
    if not pairs:
        st.warning(f"No result files found in `{results_dir}/`")
        st.info("Run an evaluation first:\n```\npython -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json\n```")
        return

    # Run selector
    run_ids = [p["run_id"] for p in pairs]
    selected_run_id = st.sidebar.selectbox("Run", run_ids, index=0)
    selected_pair = next(p for p in pairs if p["run_id"] == selected_run_id)

    raw_df, summary_df = _load_data(selected_pair)

    if summary_df is None:
        st.warning("Summary CSV not found. Showing raw results only.")
        st.dataframe(raw_df, use_container_width=True)
        return

    # Sidebar: example selection filter
    has_selection = "example_selection" in summary_df.columns
    if has_selection:
        summary_df["example_selection"] = summary_df["example_selection"].fillna("fixed")
        if "example_selection" in raw_df.columns:
            raw_df["example_selection"] = raw_df["example_selection"].fillna("fixed")
        sel_methods = sorted(summary_df["example_selection"].unique())
        if len(sel_methods) > 1:
            selected_sel = st.sidebar.selectbox(
                "Example Selection",
                options=["all"] + sel_methods,
                index=0,
            )
            if selected_sel != "all":
                summary_df = summary_df[summary_df["example_selection"] == selected_sel]
                raw_df = raw_df[raw_df["example_selection"] == selected_sel] if "example_selection" in raw_df.columns else raw_df

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Tasks**: {summary_df['task_id'].nunique()}")
    st.sidebar.markdown(f"**Models**: {summary_df['model_name'].nunique()}")
    if "num_trials" in summary_df.columns:
        st.sidebar.markdown(f"**Trials**: {int(summary_df['num_trials'].iloc[0])}")
    if has_selection and len(summary_df["example_selection"].unique()) > 1:
        st.sidebar.markdown(f"**Selection methods**: {', '.join(sorted(summary_df['example_selection'].unique()))}")
    st.sidebar.markdown(f"**Raw results**: {len(raw_df)} rows")

    # Render sections
    _render_learning_curve(summary_df)
    _render_collapse_detection(summary_df)
    _render_metrics_table(summary_df)


if __name__ == "__main__":
    main()
