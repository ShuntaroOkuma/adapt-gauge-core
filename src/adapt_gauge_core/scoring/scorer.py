"""
Scoring dispatch function

Routes to the appropriate scorer based on the scoring method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapt_gauge_core.model_client import ModelClient

from adapt_gauge_core.domain.value_objects import ScoringResult
from adapt_gauge_core.scoring.text_scorers import score_exact_match, score_contains, score_f1
from adapt_gauge_core.scoring.llm_judge import LLMJudgeError, LLMJudgeScorer

logger = logging.getLogger(__name__)

_FALLBACK_SCORERS: dict[str, object] = {
    "exact_match": score_exact_match,
    "contains": score_contains,
    "f1": score_f1,
}


def score(
    expected: str | dict,
    actual: str,
    method: str,
    *,
    grader_client: ModelClient | None = None,
    acceptable_variations: list[str] | None = None,
    input_text: str | None = None,
    fallback_method: str = "f1",
) -> ScoringResult:
    """
    Calculate the score using the specified scoring method

    Args:
        expected: Expected output (string or dictionary for llm_judge rubric)
        actual: Actual output
        method: Scoring method (exact_match, contains, f1, llm_judge)
        grader_client: ModelClient for LLM grader (required when using llm_judge)
        acceptable_variations: Acceptable variations (included in the prompt when using llm_judge)
        input_text: Original input text (passed as context when using llm_judge)
        fallback_method: Fallback scoring method on grader failure (default: "f1")

    Returns:
        ScoringResult (score + scoring reason)

    Raises:
        ValueError: When an unknown scoring method is specified, or grader_client is not provided for llm_judge
    """
    # LLM scoring
    if method == "llm_judge":
        if grader_client is None:
            raise ValueError("grader_client is required for llm_judge scoring")
        judge = LLMJudgeScorer(grader_client)
        try:
            return judge.judge(
                expected, actual,
                acceptable_variations=acceptable_variations,
                input_text=input_text,
            )
        except (LLMJudgeError, Exception) as e:
            logger.warning("LLM judge failed, falling back to '%s': %s", fallback_method, e)
            # Fallback: use the configured method for string expected, 0.0 for dict expected
            if isinstance(expected, str):
                fallback_fn = _FALLBACK_SCORERS.get(fallback_method)
                if fallback_fn is None:
                    logger.warning("Unknown fallback_method '%s', using 'f1'.", fallback_method)
                    fallback_fn = score_f1
                return ScoringResult(
                    score=fallback_fn(expected, actual),
                    reason=f"LLM judge failed, used fallback: {fallback_method}",
                )
            return ScoringResult(score=0.0, reason="LLM judge failed, dict expected: fallback to 0.0")

    # Custom scoring (not available in adapt-gauge-core)
    if method.startswith("custom:"):
        raise ValueError(
            "Custom scoring is not available in adapt-gauge-core. "
            "See https://github.com/prassist-ai/adapt-gauge for details."
        )

    # Standard scoring
    scoring_methods = {
        "exact_match": score_exact_match,
        "contains": score_contains,
        "f1": score_f1,
    }

    if method not in scoring_methods:
        raise ValueError(f"Unknown scoring method: {method} (available: {list(scoring_methods.keys())})")

    if not isinstance(expected, str):
        raise ValueError(f"Standard scoring requires a string expected value: {method}")

    return ScoringResult(score=scoring_methods[method](expected, actual))
