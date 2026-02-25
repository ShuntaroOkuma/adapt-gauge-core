"""
adapt-gauge-core CLI Runner

Minimal CLI for running evaluations with the core pipeline.

Usage:
    python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json
    python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json --models gemini-2.5-flash,claude-haiku-4-5-20251001

Resume a previous run:
    python -m adapt_gauge_core.runner --task-pack tasks/task_pack_core_demo.json --run-id 20260101_120000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from adapt_gauge_core.domain.constants import DEFAULT_MODELS, SHOT_SCHEDULE
from adapt_gauge_core.domain.entities import EvaluationResult
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
from adapt_gauge_core.use_cases.aei import (
    detect_negative_learning,
    detect_peak_regression,
    detect_mid_curve_dip,
    classify_collapse_pattern,
    calculate_resilience_score,
)

SAVE_INTERVAL = 10


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
        "--num-trials",
        type=int,
        default=None,
        help="Number of trials (default: HARNESS_NUM_TRIALS from .env)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID to resume a previous run (loads existing results and skips completed evaluations)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output CSV files (default: results)",
    )
    return parser.parse_args()


def _build_skip_set(raw_path: Path) -> tuple[list[dict], set[tuple]]:
    """
    Load existing raw results and build a skip set for resume.

    Args:
        raw_path: Path to the raw results CSV file

    Returns:
        Tuple of (existing results as list of dicts, skip set of eval keys)
    """
    if not raw_path.exists():
        return [], set()

    try:
        existing_df = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError:
        return [], set()

    if existing_df.empty:
        return [], set()

    existing_results = existing_df.to_dict("records")
    skip_set: set[tuple] = set()
    for r in existing_results:
        skip_set.add((
            str(r["task_id"]),
            str(r["model_name"]),
            int(r["shot_count"]),
            int(r.get("trial_id", 1)),
            hashlib.sha256(str(r["input"]).encode("utf-8")).hexdigest(),
        ))

    return existing_results, skip_set


def _save_raw_results(all_results: list[dict], raw_path: Path) -> None:
    """Save raw results to CSV."""
    raw_df = pd.DataFrame(all_results)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Load config
    config = load_config()
    make_client = partial(create_client, config=config)

    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS

    # Determine number of trials
    num_trials = args.num_trials if args.num_trials else config.trials.num_trials
    shots = SHOT_SCHEDULE

    # Determine run_id
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"raw_results_{run_id}.csv"
    summary_path = output_dir / f"summary_{run_id}.csv"

    # Load task pack
    print(f"\n=== Loading task pack: {args.task_pack} ===\n")
    task_pack = load_task_pack(args.task_pack)
    tasks = task_pack.tasks
    print(f"  Pack: {task_pack.pack_name}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Models: {models}")
    print(f"  Shots: {shots}")
    print(f"  Trials: {num_trials}")
    print(f"  Run ID: {run_id}")
    print()

    # Resume: load existing results
    all_results: list[dict] = []
    skip_set: set[tuple] = set()
    if args.run_id:
        all_results, skip_set = _build_skip_set(raw_path)
        if skip_set:
            print(f"=== Resuming: {len(all_results)} evaluations already completed ===\n")

    # Step 1: Health check
    available_models, _ = run_health_check(models, make_client)

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
        grader_ok, grader_err = run_grader_health_check(grader_model, make_client)
        if grader_ok:
            print("OK")
            grader_client = make_client(grader_model)
        else:
            print("FAILED")
            print(f"  {grader_err}")
            print("  WARNING: llm_judge tasks will fall back to f1 scoring.\n")
        print()

    # Step 2: Run evaluations
    num_evals_per_trial = sum(len(t.test_cases) for t in tasks) * len(available_models) * len(shots)
    total = num_evals_per_trial * num_trials
    print(f"=== Running Evaluations ({total} total) ===\n")

    current = 0
    last_saved_count = len(all_results)

    for trial_id in range(1, num_trials + 1):
        trial_label = f"T{trial_id}/{num_trials}" if num_trials > 1 else ""
        for model_name in available_models:
            if config.isolation.new_client_per_trial or trial_id == 1:
                client = make_client(model_name)
            for task in tasks:
                for shot in shots:
                    for test_case in task.test_cases:
                        current += 1

                        # Skip if already completed (resume)
                        eval_key = (
                            task.task_id,
                            model_name,
                            shot,
                            trial_id,
                            hashlib.sha256(test_case.input.encode("utf-8")).hexdigest(),
                        )
                        if eval_key in skip_set:
                            continue

                        print(f"[{current}/{total}] {trial_label} {task.task_id} | {model_name} | {shot}-shot")

                        try:
                            result = run_single_evaluation(
                                task=task,
                                test_case=test_case,
                                model_client=client,
                                shot_count=shot,
                                run_id=run_id,
                                grader_client=grader_client,
                                trial_id=trial_id,
                            )
                            all_results.append(asdict(result))
                            print(f"  Score: {result.score:.2f} | Latency: {result.latency_ms}ms")
                        except Exception as e:
                            error_result = {
                                "run_id": run_id,
                                "task_id": task.task_id,
                                "category": task.category,
                                "model_name": model_name,
                                "shot_count": shot,
                                "input": test_case.input,
                                "expected_output": test_case.expected_output,
                                "actual_output": f"ERROR: {e}",
                                "score": 0.0,
                                "scoring_method": test_case.scoring_method,
                                "latency_ms": 0,
                                "timestamp": datetime.now().isoformat(),
                                "trial_id": trial_id,
                                "input_tokens": 0,
                                "output_tokens": 0,
                            }
                            all_results.append(error_result)
                            print(f"  ERROR: {e}")
                            traceback.print_exc()

                        # Intermediate save
                        if len(all_results) - last_saved_count >= SAVE_INTERVAL:
                            _save_raw_results(all_results, raw_path)
                            last_saved_count = len(all_results)

    # Step 3: Aggregate results
    print(f"\n=== Aggregating Results ===\n")
    result_objects = [
        EvaluationResult(**{k: v for k, v in r.items() if k in EvaluationResult.__dataclass_fields__})
        for r in all_results
    ]
    summary_df = aggregate_results(result_objects, tasks, config=config)

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

    # Step 5: Collapse detection (3 types)
    neg_alerts = detect_negative_learning(summary_df)
    peak_alerts = detect_peak_regression(summary_df)
    dip_alerts = detect_mid_curve_dip(summary_df)

    has_any = neg_alerts or peak_alerts or dip_alerts

    if has_any:
        print("=== Collapse Detection (WARNING) ===\n")
        if neg_alerts:
            print("  --- Negative Learning ---")
            for alert in neg_alerts:
                print(
                    f"  WARNING [{alert['severity']}]: {alert['model']} | {alert['task_id']} "
                    f"| 0-shot={alert['score_0shot']:.3f} -> final={alert['score_final']:.3f} "
                    f"(drop {alert['drop_pct']:.1f}%)"
                )
            print()
        if peak_alerts:
            print("  --- Peak Regression ---")
            for alert in peak_alerts:
                print(
                    f"  WARNING: {alert['model']} | {alert['task_id']} "
                    f"| peak={alert['score_peak']:.3f} at {alert['peak_shot']}-shot "
                    f"-> final={alert['score_final']:.3f} "
                    f"(drop {alert['drop_pct']:.1f}%)"
                )
            print()
        if dip_alerts:
            print("  --- Mid-curve Dip ---")
            for alert in dip_alerts:
                print(
                    f"  WARNING: {alert['model']} | {alert['task_id']} "
                    f"| {alert['from_shot']}-shot={alert['score_from']:.3f} "
                    f"-> {alert['to_shot']}-shot={alert['score_to']:.3f} "
                    f"(drop {alert['drop_pct']:.1f}%)"
                )
            print()
    else:
        print("=== Collapse Detection: No issues found ===\n")

    # Step 5b: Collapse pattern classification & resilience score
    classifications = classify_collapse_pattern(summary_df)
    resilience_scores = calculate_resilience_score(summary_df)

    if classifications:
        pattern_map = {
            (c["model"], c["task_id"]): c["pattern"]
            for c in classifications
        }
        summary_df["collapse_pattern"] = summary_df.apply(
            lambda r: pattern_map.get((r["model_name"], r.get("task_id", "")), ""),
            axis=1,
        )

    if resilience_scores:
        summary_df["resilience_score"] = summary_df["model_name"].map(resilience_scores)

        print("=== Collapse Resilience Score ===\n")
        for model, score in sorted(resilience_scores.items(), key=lambda x: -x[1]):
            print(f"  {model:<40} {score:.3f}")
        print()

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
    _save_raw_results(all_results, raw_path)
    summary_df.to_csv(summary_path, index=False)

    print(f"=== Output ===\n")
    print(f"  Raw results: {raw_path}")
    print(f"  Summary:     {summary_path}")
    print()


if __name__ == "__main__":
    main()
