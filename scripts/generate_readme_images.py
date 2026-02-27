"""Generate static images for README from demo evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

SHOT_SCHEDULE = [0, 1, 2, 4, 8]
SHOT_LABELS = {s: f"{s}-shot" for s in SHOT_SCHEDULE}
MODEL_COLORS = {
    "claude-haiku-4-5-20251001": "#1a73e8",
    "claude-opus-4-5-20251101": "#e8710a",
    "gemini-2.5-flash": "#34a853",
    "gemini-3-flash-preview": "#ea4335",
    "gemini-3-pro-preview": "#9334e6",
}
COLLAPSE_HIGHLIGHT_THRESHOLD = 0.02

OUTPUT_DIR = Path("docs/images")
SUMMARY_CSV = Path("results/demo/summary_demo.csv")


def short_name(name: str) -> str:
    parts = name.split("/")
    return parts[-1] if len(parts) > 1 else name


def generate_learning_curve(df: pd.DataFrame, task_id: str, output_path: Path) -> None:
    """Generate a learning curve chart for a specific task."""
    filtered = df[df["task_id"] == task_id]
    models = filtered["model_name"].unique()

    fig = go.Figure()

    for model in models:
        row = filtered[filtered["model_name"] == model].iloc[0]
        color = MODEL_COLORS.get(model, "#546e7a")
        sname = short_name(model)

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
            name=sname,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=9),
        ))

        # Highlight few-shot collapse intervals
        for j in range(1, len(scores)):
            if scores[j] < scores[j - 1] - COLLAPSE_HIGHLIGHT_THRESHOLD:
                fig.add_trace(go.Scatter(
                    x=[shots[j - 1], shots[j]],
                    y=[scores[j - 1], scores[j]],
                    mode="lines",
                    line=dict(color="#ea4335", width=3, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="#5f6368",
        line_width=1,
        annotation_text="80% target",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#5f6368"),
    )

    fig.update_layout(
        title=dict(
            text=f"Learning Curve: {task_id}",
            font=dict(size=18),
        ),
        xaxis_title="Shot count",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        xaxis_tickvals=SHOT_SCHEDULE,
        xaxis_ticktext=[SHOT_LABELS[s] for s in SHOT_SCHEDULE],
        legend_title="Model",
        template="plotly_white",
        height=480,
        width=900,
        margin=dict(l=60, r=30, t=60, b=60),
        font=dict(size=13),
    )

    fig.write_image(str(output_path), scale=2)
    print(f"  Generated: {output_path}")


def generate_overview(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a multi-task overview showing all learning curves in a grid."""
    tasks = df["task_id"].unique()
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[str(t) for t in tasks],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, task_id in enumerate(tasks):
        row = idx // 2 + 1
        col = idx % 2 + 1
        filtered = df[df["task_id"] == task_id]

        for model in filtered["model_name"].unique():
            mrow = filtered[filtered["model_name"] == model].iloc[0]
            color = MODEL_COLORS.get(model, "#546e7a")
            sname = short_name(model)

            scores = [mrow[f"score_{s}shot"] for s in SHOT_SCHEDULE if f"score_{s}shot" in mrow.index]

            fig.add_trace(go.Scatter(
                x=SHOT_SCHEDULE[:len(scores)],
                y=scores,
                mode="lines+markers",
                name=sname,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6),
                showlegend=(idx == 0),
                legendgroup=model,
            ), row=row, col=col)

        fig.update_yaxes(range=[0, 1.05], row=row, col=col)
        fig.update_xaxes(
            tickvals=SHOT_SCHEDULE,
            ticktext=[SHOT_LABELS[s] for s in SHOT_SCHEDULE],
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(text="Few-Shot Learning Curves Across Tasks", font=dict(size=18)),
        template="plotly_white",
        height=700,
        width=1100,
        margin=dict(l=50, r=30, t=80, b=50),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )

    fig.write_image(str(output_path), scale=2)
    print(f"  Generated: {output_path}")


def main() -> None:
    print("Generating README images from demo data...")
    df = pd.read_csv(SUMMARY_CSV)

    # 1. Overview: all tasks in a grid
    generate_overview(df, OUTPUT_DIR / "learning-curves-overview.png")

    # 2. Single task with clear collapse pattern
    generate_learning_curve(df, "custom_route_001", OUTPUT_DIR / "learning-curve-collapse.png")

    print("Done!")


if __name__ == "__main__":
    main()
