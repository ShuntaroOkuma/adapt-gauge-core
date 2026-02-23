"""
Domain Value Objects

Defines immutable data structures representing values such as scoring results,
model responses, and cost metrics.
"""

from dataclasses import dataclass


@dataclass
class ScoringResult:
    """Scoring result (score + reason)"""

    score: float
    reason: str | None = None


@dataclass
class ModelResponse:
    """Model response"""
    output: str
    latency_ms: int
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CostMetrics:
    """Cost calculation metrics"""
    input_tokens: int
    output_tokens: int
    latency_ms: int

    # Pricing (USD per 1M tokens)
    input_price_per_m: float = 0.0
    output_price_per_m: float = 0.0
    time_price_per_sec: float = 0.0  # USD/sec

    def __post_init__(self):
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
