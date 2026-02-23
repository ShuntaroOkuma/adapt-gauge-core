"""
Scoring sub-package

Provides text scoring and LLM Judge scoring logic.
"""

from adapt_gauge_core.domain.value_objects import ScoringResult
from adapt_gauge_core.scoring.scorer import score, _FALLBACK_SCORERS
from adapt_gauge_core.scoring.text_scorers import (
    remove_markdown,
    normalize_text,
    score_exact_match,
    score_contains,
    score_f1,
    _tokenize,
    _char_type,
)
from adapt_gauge_core.scoring.llm_judge import LLMJudgeError, LLMJudgeScorer

__all__ = [
    # value objects (re-exported from domain)
    "ScoringResult",
    # dispatcher
    "score",
    "_FALLBACK_SCORERS",
    # text scorers
    "remove_markdown",
    "normalize_text",
    "score_exact_match",
    "score_contains",
    "score_f1",
    "_tokenize",
    "_char_type",
    # llm judge
    "LLMJudgeError",
    "LLMJudgeScorer",
]
