"""
Domain Layer

Defines constants, entities, and value objects that form the core of the business logic.
Has no dependencies on external libraries.
"""

from adapt_gauge_core.domain.constants import (
    DEFAULT_MODELS,
    MODEL_PRICING,
    SHOT_SCHEDULE,
    _LOCAL_MODEL_PRICING,
)
from adapt_gauge_core.domain.entities import (
    AcquisitionMetrics,
    EvaluationResult,
    HealthCheckResult,
)
from adapt_gauge_core.domain.value_objects import (
    CostMetrics,
    ModelResponse,
    ScoringResult,
)

__all__ = [
    # constants
    "DEFAULT_MODELS",
    "MODEL_PRICING",
    "SHOT_SCHEDULE",
    "_LOCAL_MODEL_PRICING",
    # entities
    "AcquisitionMetrics",
    "EvaluationResult",
    "HealthCheckResult",
    # value objects
    "CostMetrics",
    "ModelResponse",
    "ScoringResult",
]
