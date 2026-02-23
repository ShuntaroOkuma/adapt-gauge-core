"""
LLMJudgeScorer の _parse_score テスト

JSON, regex, bare float, エラーパスのパーステストを新パスからインポートして実施。
"""

import pytest
from dataclasses import dataclass

from adapt_gauge_core.scoring.llm_judge import LLMJudgeScorer, LLMJudgeError
from adapt_gauge_core.domain.value_objects import ScoringResult


# ---------------------------------------------------------------------------
# MockGraderClient
# ---------------------------------------------------------------------------

@dataclass
class _MockResponse:
    output: str
    latency_ms: int = 100
    model_name: str = "mock-grader"


class MockGraderClient:
    """テスト用のgraderクライアント"""

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> _MockResponse:
        self.last_prompt = prompt
        return _MockResponse(output=self._response_text)


# ===========================================================================
# _parse_score テスト: JSON パス
# ===========================================================================


class TestParseScoreJSON:
    """_parse_score の JSON パーステスト"""

    def _parse(self, text: str) -> ScoringResult:
        client = MockGraderClient(text)
        judge = LLMJudgeScorer(client)
        return judge._parse_score(text)

    def test_valid_json(self):
        result = self._parse('{"score": 0.85, "reason": "Good"}')
        assert result.score == pytest.approx(0.85)
        assert result.reason == "Good"

    def test_json_in_code_block(self):
        result = self._parse('```json\n{"score": 0.7, "reason": "OK"}\n```')
        assert result.score == pytest.approx(0.7)
        assert result.reason == "OK"

    def test_json_with_surrounding_text(self):
        text = 'Here is my evaluation:\n```json\n{"score": 0.6, "reason": "partial"}\n```\nDone.'
        result = self._parse(text)
        assert result.score == pytest.approx(0.6)
        assert result.reason == "partial"

    def test_score_integer(self):
        result = self._parse('{"score": 1, "reason": "Perfect"}')
        assert result.score == pytest.approx(1.0)

    def test_score_zero(self):
        result = self._parse('{"score": 0, "reason": "Wrong"}')
        assert result.score == pytest.approx(0.0)


# ===========================================================================
# _parse_score テスト: 正規表現フォールバック
# ===========================================================================


class TestParseScoreRegex:
    """_parse_score の正規表現フォールバックテスト"""

    def _parse(self, text: str) -> ScoringResult:
        client = MockGraderClient(text)
        judge = LLMJudgeScorer(client)
        return judge._parse_score(text)

    def test_score_colon_pattern(self):
        result = self._parse("Score: 0.6\nReason: partially correct")
        assert result.score == pytest.approx(0.6)
        assert result.reason is None

    def test_japanese_score_pattern(self):
        result = self._parse("スコア: 0.9")
        assert result.score == pytest.approx(0.9)
        assert result.reason is None

    def test_score_without_colon(self):
        result = self._parse("Score 0.5 is the result")
        assert result.score == pytest.approx(0.5)


# ===========================================================================
# _parse_score テスト: ベアfloat
# ===========================================================================


class TestParseScoreBareFloat:
    """_parse_score のベアfloatテスト"""

    def _parse(self, text: str) -> ScoringResult:
        client = MockGraderClient(text)
        judge = LLMJudgeScorer(client)
        return judge._parse_score(text)

    def test_bare_float(self):
        result = self._parse("0.75")
        assert result.score == pytest.approx(0.75)
        assert result.reason is None

    def test_bare_zero(self):
        result = self._parse("0")
        assert result.score == pytest.approx(0.0)

    def test_bare_one(self):
        result = self._parse("1")
        assert result.score == pytest.approx(1.0)

    def test_bare_dot_notation(self):
        result = self._parse(".5")
        assert result.score == pytest.approx(0.5)


# ===========================================================================
# _parse_score テスト: エラーパス
# ===========================================================================


class TestParseScoreError:
    """_parse_score のエラーパステスト"""

    def _parse(self, text: str) -> ScoringResult:
        client = MockGraderClient(text)
        judge = LLMJudgeScorer(client)
        return judge._parse_score(text)

    def test_unparseable_response(self):
        with pytest.raises(LLMJudgeError, match="Failed to parse score"):
            self._parse("I cannot evaluate this response")

    def test_random_text(self):
        with pytest.raises(LLMJudgeError):
            self._parse("This is completely random text with no score")

    def test_empty_string(self):
        with pytest.raises(LLMJudgeError):
            self._parse("")


# ===========================================================================
# クランプテスト
# ===========================================================================


class TestParseScoreClamp:
    """スコアクランプのテスト"""

    def _parse(self, text: str) -> ScoringResult:
        client = MockGraderClient(text)
        judge = LLMJudgeScorer(client)
        return judge._parse_score(text)

    def test_clamp_above_one(self):
        result = self._parse('{"score": 1.5, "reason": "over"}')
        assert result.score == pytest.approx(1.0)

    def test_clamp_below_zero(self):
        result = self._parse('{"score": -0.3, "reason": "negative"}')
        assert result.score == pytest.approx(0.0)
