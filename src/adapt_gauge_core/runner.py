"""
adapt-gauge-core CLI Runner

Minimal CLI for running evaluations with the core pipeline.

Usage:
    python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json
    python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json --models gemini-2.5-flash,claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from adapt_gauge_core.domain.constants import DEFAULT_MODELS, SHOT_SCHEDULE
from adapt_gauge_core.harness_config import load_config
from adapt_gauge_core.model_client import create_client
from adapt_gauge_core.task_loader import load_task_pack
from adapt_gauge_core.use_cases.health_check import (
    run_health_check,
    get_llm_judge_tasks,
    run_grader_health_check,
)
from adapt_gauge_core.use_cases.evaluation import (
    run_single_evaluation,
    aggregate_results,
)
from adapt_gauge_core.use_cases.aei import detect_negative_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="adapt-gauge-core: Evaluate LLM adaptation efficiency",
    )
    parser.add_argument(
        "--task-pack",
        required=True,
        help="Path to the task pack JSON file",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names (default: uses DEFAULT_MODELS)",
    )
    parser.add_argument(
        "--shots",
        default=None,
        help="Comma-separated list of shot counts (default: 0,1,2,4,8)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output CSV files (default: results)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Load config
    config = load_config()

    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS

    # Parse shots
    if args.shots:
        shots = [int(s.strip()) for s in args.shots.split(",")]
    else:
        shots = SHOT_SCHEDULE

    # Load task pack
    print(f"\n=== Loading task pack: {args.task_pack} ===\n")
    task_pack = load_task_pack(args.task_pack)
    tasks = task_pack.tasks
    print(f"  Pack: {task_pack.pack_name}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Models: {models}")
    print(f"  Shots: {shots}")
    print()

    # Step 1: Health check
    available_models, _ = run_health_check(models, create_client)

    if not available_models:
        print("ERROR: No models available. Exiting.")
        sys.exit(1)

    # Step 1b: Grader health check (if llm_judge tasks exist)
    llm_judge_task_ids = get_llm_judge_tasks(task_pack)
    grader_client = None
    if llm_judge_task_ids:
        grader_model = os.environ.get("LLM_JUDGE_GRADER_MODEL", config.llm_judge.grader_model)
        print(f"=== Grader Health Check (llm_judge tasks: {llm_judge_task_ids}) ===\n")
        print(f"  Grader model: {grader_model}... ", end="", flush=True)
        grader_ok, grader_err = run_grader_health_check(grader_model, create_client)
        if grader_ok:
            print("OK")
            grader_client = create_client(grader_model)
        else:
            print("FAILED")
            print(f"  {grader_err}")
            print("  WARNING: llm_judge tasks will fall back to f1 scoring.\n")
        print()

    # Step 2: Run evaluations
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = sum(len(t.test_cases) for t in tasks) * len(available_models) * len(shots)
    print(f"=== Running Evaluations ({total} total) ===\n")

    all_results = []
    current = 0

    for model_name in available_models:
        client = create_client(model_name)
        for task in tasks:
            for shot in shots:
                for test_case in task.test_cases:
                    current += 1
                    print(f"[{current}/{total}] {task.task_id} | {model_name} | {shot}-shot")

                    result = run_single_evaluation(
                        task=task,
                        test_case=test_case,
                        model_client=client,
                        shot_count=shot,
                        run_id=run_id,
                        grader_client=grader_client,
                    )
                    all_results.append(result)
                    print(f"  Score: {result.score:.2f} | Latency: {result.latency_ms}ms")

    # Step 3: Aggregate results
    print(f"\n=== Aggregating Results ===\n")
    summary_df = aggregate_results(all_results, tasks)

    # Step 4: Display learning curves
    print("=== Learning Curves ===\n")
    for task in tasks:
        task_summary = summary_df[summary_df["task_id"] == task.task_id]
        if task_summary.empty:
            continue
        print(f"  Task: {task.task_id}")
        print(f"  {'Model':<40} {'0-shot':>7} {'1-shot':>7} {'2-shot':>7} {'4-shot':>7} {'8-shot':>7} {'AUC':>7}")
        print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for _, row in task_summary.iterrows():
            print(
                f"  {row['model_name']:<40} "
                f"{row['score_0shot']:>7.3f} "
                f"{row['score_1shot']:>7.3f} "
                f"{row['score_2shot']:>7.3f} "
                f"{row['score_4shot']:>7.3f} "
                f"{row['score_8shot']:>7.3f} "
                f"{row['learning_curve_auc']:>7.3f}"
            )
        print()

    # Step 5: Collapse detection
    alerts = detect_negative_learning(summary_df)
    if alerts:
        print("=== Collapse Detection (WARNING) ===\n")
        for alert in alerts:
            print(
                f"  WARNING: {alert['model']} | {alert['task_id']} "
                f"| 0-shot={alert['score_0shot']:.3f} -> final={alert['score_final']:.3f} "
                f"(drop {alert['drop_pct']:.1f}%)"
            )
        print()
    else:
        print("=== Collapse Detection: No issues found ===\n")

    # Step 6: Basic metrics summary
    print("=== Metrics Summary ===\n")
    print(f"  {'Model':<40} {'improvement_rate':>17} {'threshold_shots':>16} {'learning_curve_auc':>19}")
    print(f"  {'-'*40} {'-'*17} {'-'*16} {'-'*19}")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['model_name']:<40} "
            f"{row['improvement_rate']:>17.4f} "
            f"{row['threshold_shots']:>16} "
            f"{row['learning_curve_auc']:>19.4f}"
        )
    print()

    # Step 7: Save CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"raw_results_{run_id}.csv"
    summary_path = output_dir / f"summary_{run_id}.csv"

    raw_df = pd.DataFrame([asdict(r) for r in all_results])
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"=== Output ===\n")
    print(f"  Raw results: {raw_path}")
    print(f"  Summary:     {summary_path}")
    print()


if __name__ == "__main__":
    main()
