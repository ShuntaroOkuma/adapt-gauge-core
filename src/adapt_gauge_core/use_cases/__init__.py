"""
Use Cases Layer

Aggregates business logic and provides use cases called from the runner.
"""

from adapt_gauge_core.use_cases.aei import (
    compute_aei,
    detect_few_shot_collapse,
    detect_negative_learning,  # backward compatibility alias
    detect_peak_regression,
    detect_mid_curve_dip,
)
from adapt_gauge_core.use_cases.evaluation import (
    run_single_evaluation,
    _run_model_evaluations,
    run_all_evaluations,
    aggregate_results,
)
from adapt_gauge_core.use_cases.health_check import (
    HEALTH_CHECK_PROMPT,
    health_check_model,
    health_check_all_models,
    run_health_check,
    get_llm_judge_tasks,
    run_grader_health_check,
)

__all__ = [
    # aei
    "compute_aei",
    "detect_few_shot_collapse",
    "detect_negative_learning",  # backward compatibility alias
    "detect_peak_regression",
    "detect_mid_curve_dip",
    # evaluation
    "run_single_evaluation",
    "_run_model_evaluations",
    "run_all_evaluations",
    "aggregate_results",
    # health_check
    "HEALTH_CHECK_PROMPT",
    "health_check_model",
    "health_check_all_models",
    "run_health_check",
    "get_llm_judge_tasks",
    "run_grader_health_check",
]
