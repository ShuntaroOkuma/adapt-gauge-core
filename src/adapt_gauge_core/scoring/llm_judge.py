"""
LLM Judge scoring logic

Implements LLMJudgeScorer, which uses a separate model (grader) for scoring.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapt_gauge_core.model_client import ModelClient

from adapt_gauge_core.domain.value_objects import ScoringResult

logger = logging.getLogger(__name__)


class LLMJudgeError(Exception):
    """Error raised during LLM scoring"""
    pass


class LLMJudgeScorer:
    """
    Scorer that uses an LLM as a grader

    Passes the expected output and actual output to a separate model (grader) for scoring.
    """

    # Regex patterns for score extraction
    _SCORE_RE = re.compile(r"(?:score|スコア)\s*[:：]?\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)
    _BARE_FLOAT_RE = re.compile(r"^[01](?:\.\d+)?$|^\.\d+$")

    def __init__(self, grader_client: ModelClient) -> None:
        self._client = grader_client

    def judge(
        self,
        expected: str | dict,
        actual: str,
        *,
        acceptable_variations: list[str] | None = None,
        input_text: str | None = None,
    ) -> ScoringResult:
        """
        Have the LLM score and return the result

        Args:
            expected: Expected output (string or rubric dictionary)
            actual: Actual output
            acceptable_variations: Acceptable variations
            input_text: Original input text (for context)

        Returns:
            ScoringResult (score + scoring reason)

        Raises:
            LLMJudgeError: When parsing the response fails
        """
        prompt = self._build_prompt(
            expected, actual,
            acceptable_variations=acceptable_variations,
            input_text=input_text,
        )
        response = self._client.generate(prompt)
        return self._parse_score(response.output)

    @staticmethod
    def _is_keyword_list(expected: str) -> bool:
        """
        Heuristic to determine whether expected is a keyword list type.

        Conditions:
        - Number of whitespace-separated tokens >= 2
        - Average character count per token is 15 or less
        - Does not contain sentence-ending punctuation
        """
        tokens = expected.strip().split()
        if len(tokens) < 2:
            return False
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        has_sentence_end = any(c in expected for c in "\u3002.!?")
        return avg_len <= 15 and not has_sentence_end

    def _build_prompt(
        self,
        expected: str | dict,
        actual: str,
        *,
        acceptable_variations: list[str] | None = None,
        input_text: str | None = None,
    ) -> str:
        """Build a scoring prompt (branches by keyword type / natural text type / rubric type)"""
        if isinstance(expected, dict):
            return self._build_rubric_prompt(
                expected, actual,
                acceptable_variations=acceptable_variations,
                input_text=input_text,
            )
        if self._is_keyword_list(expected):
            return self._build_keyword_prompt(
                expected, actual,
                acceptable_variations=acceptable_variations,
                input_text=input_text,
            )
        return self._build_natural_prompt(
            expected, actual,
            acceptable_variations=acceptable_variations,
            input_text=input_text,
        )

    def _build_keyword_prompt(
        self,
        expected: str,
        actual: str,
        *,
        acceptable_variations: list[str] | None = None,
        input_text: str | None = None,
    ) -> str:
        """Prompt for keyword list type expected"""
        tokens = expected.strip().split()
        keyword_lines = "\n".join(f"  {i+1}. {kw}" for i, kw in enumerate(tokens))
        parts: list[str] = [
            "You are a strict keyword evaluator.",
            "The EXPECTED output contains a space-separated list of required keywords/phrases.",
            "Your task: check whether each keyword appears in the ACTUAL output "
            "(exact match or clear semantic equivalent).",
            "",
            f"Keywords to check:\n{keyword_lines}",
            "",
            "For each keyword, mark it as FOUND or MISSING.",
            "Score = (number of FOUND keywords) / (total keywords)",
            "",
        ]
        if input_text is not None:
            parts.append(f"INPUT (the question / task that was given):\n{input_text}")
            parts.append("")
        parts.append(f"ACTUAL (the output to evaluate):\n{actual}")
        if acceptable_variations:
            parts.append("")
            parts.append("ACCEPTABLE VARIATIONS (these phrasings are also considered correct):")
            for v in acceptable_variations:
                parts.append(f"  - {v}")
        parts.append("")
        parts.append(
            'Return JSON: {"score": <float 0.0-1.0>, "reason": "Found: [...], Missing: [...]"}'
        )
        return "\n".join(parts)

    def _build_natural_prompt(
        self,
        expected: str,
        actual: str,
        *,
        acceptable_variations: list[str] | None = None,
        input_text: str | None = None,
    ) -> str:
        """Prompt for natural text type expected (5-level rubric)"""
        parts: list[str] = [
            "You are a strict evaluator. Compare the ACTUAL output against the EXPECTED output.",
            "Do NOT check for exact string match. Instead, evaluate whether the ACTUAL output is semantically correct "
            "and follows the direction / intent of the EXPECTED output.",
            "",
            "Rubric:",
            "  1.0 = Perfect: ALL key information present and correct",
            "  0.8 = Good: Most key information present, minor omissions",
            "  0.6 = Partial: Core idea captured but significant details missing",
            "  0.4 = Weak: Only basic concept present, major information missing",
            "  0.2 = Minimal: Very little relevant content",
            "  0.0 = Wrong: Completely incorrect or irrelevant",
            "",
            "Be STRICT. Score 1.0 ONLY if the output covers ALL aspects.",
            "Your score MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0",
            "",
        ]
        if input_text is not None:
            parts.append(f"INPUT (the question / task that was given):\n{input_text}")
            parts.append("")
        parts.append(f"EXPECTED (desired intent / direction of the answer):\n{expected}")
        parts.append("")
        parts.append(f"ACTUAL (the output to evaluate):\n{actual}")
        if acceptable_variations:
            parts.append("")
            parts.append("ACCEPTABLE VARIATIONS (these phrasings are also considered correct):")
            for v in acceptable_variations:
                parts.append(f"  - {v}")
        parts.append("")
        parts.append('Return JSON: {"score": <float>, "reason": "explanation"}')
        return "\n".join(parts)

    def _build_rubric_prompt(
        self,
        expected: dict,
        actual: str,
        *,
        acceptable_variations: list[str] | None = None,
        input_text: str | None = None,
    ) -> str:
        """Prompt for rubric dictionary type expected (backward compatible)"""
        parts: list[str] = [
            "You are an expert grader.",
            "Your task is to judge whether the ACTUAL output fulfills the intent described by the EXPECTED output.",
            "Do NOT check for exact string match. Instead, evaluate whether the ACTUAL output is semantically correct "
            "and follows the direction / intent of the EXPECTED output.",
            "",
            "Scoring guide:",
            "  1.0 = The actual output fully satisfies the intent of the expected output.",
            "  0.5 = Partially correct \u2014 captures the core idea but has notable omissions or inaccuracies.",
            "  0.0 = Completely wrong or irrelevant to the expected intent.",
            "",
            "Return a JSON object with exactly two keys: \"score\" (float 0.0-1.0) and \"reason\" (string).",
            "",
        ]
        if input_text is not None:
            parts.append(f"INPUT (the question / task that was given):\n{input_text}")
            parts.append("")
        parts.append(f"RUBRIC (evaluation criteria):\n{json.dumps(expected, ensure_ascii=False, indent=2)}")
        parts.append("")
        parts.append(f"ACTUAL (the output to evaluate):\n{actual}")
        if acceptable_variations:
            parts.append("")
            parts.append("ACCEPTABLE VARIATIONS (these phrasings are also considered correct):")
            for v in acceptable_variations:
                parts.append(f"  - {v}")
        parts.append("")
        parts.append('Respond ONLY with the JSON object, e.g. {"score": 0.8, "reason": "..."}')
        return "\n".join(parts)

    def _parse_score(self, raw: str) -> ScoringResult:
        """
        Extract score and reason from the grader's response

        Parse order:
        1. JSON extraction (score + reason)
        2. Regex fallback (no reason)
        3. Bare float (no reason)
        4. LLMJudgeError
        """
        text = raw.strip()

        # 1. JSON extraction
        try:
            # Explicitly extract the JSON portion from code blocks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            json_text = match.group(1) if match else text
            data = json.loads(json_text.strip())
            if isinstance(data, dict) and "score" in data:
                return ScoringResult(
                    score=self._clamp(float(data["score"])),
                    reason=data.get("reason"),
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2. Regex fallback
        m = self._SCORE_RE.search(text)
        if m:
            return ScoringResult(score=self._clamp(float(m.group(1))))

        # 3. Bare float (when the response is only a number)
        if self._BARE_FLOAT_RE.match(text):
            return ScoringResult(score=self._clamp(float(text)))

        raise LLMJudgeError(f"Failed to parse score from grader response: {text[:200]}")

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp score to the range 0.0-1.0"""
        return max(0.0, min(1.0, value))
