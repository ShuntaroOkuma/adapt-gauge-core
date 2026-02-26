"""
Domain Entities

Defines the primary data structures used in the evaluation process.
"""

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    run_id: str
    task_id: str
    category: str
    model_name: str
    shot_count: int
    input: str
    expected_output: str
    actual_output: str
    score: float
    scoring_method: str
    latency_ms: int
    timestamp: str
    trial_id: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    example_selection: str = "fixed"


@dataclass
class AcquisitionMetrics:
    """Learning efficiency metrics"""
    improvement_rate: float      # (score[8] - score[0]) / 8
    threshold_shots: int         # Number of shots to reach 80%
    learning_curve_auc: float    # Learning curve AUC


@dataclass
class HealthCheckResult:
    """Health check result"""
    model_name: str
    success: bool
    latency_ms: int | None
    error: str | None
