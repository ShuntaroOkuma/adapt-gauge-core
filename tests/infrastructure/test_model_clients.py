"""
Tests for model clients

Tests RetryMixin._with_retry() retry behavior and
create_client() factory branching.
"""

import pytest
from unittest.mock import patch, MagicMock

from adapt_gauge_core.infrastructure.model_clients.base import RetryMixin
from adapt_gauge_core.infrastructure.model_clients.factory import create_client
from adapt_gauge_core.infrastructure.model_clients.vertex_ai import VertexAIClient
from adapt_gauge_core.infrastructure.model_clients.claude import ClaudeClient
from adapt_gauge_core.infrastructure.model_clients.lmstudio import LMStudioClient, _strip_thinking


class TestRetryMixin:
    """Tests for RetryMixin._with_retry()"""

    def _make_mixin(self, max_retries=3):
        mixin = RetryMixin()
        mixin.max_retries = max_retries
        return mixin

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_success_on_first_attempt(self, mock_sleep):
        """Should return value without retries on first success"""
        mixin = self._make_mixin()
        fn = MagicMock(return_value="ok")

        result = mixin._with_retry(fn)

        assert result == "ok"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_success_after_two_failures(self, mock_sleep):
        """Should succeed on third attempt after two failures"""
        mixin = self._make_mixin(max_retries=3)
        fn = MagicMock(side_effect=[ValueError("1"), ValueError("2"), "ok"])

        result = mixin._with_retry(fn)

        assert result == "ok"
        assert fn.call_count == 3
        # Exponential backoff: sleep(1), sleep(2)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 2 ** 0
        mock_sleep.assert_any_call(2)  # 2 ** 1

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        """Should raise the last exception when all retries are exhausted"""
        mixin = self._make_mixin(max_retries=3)
        fn = MagicMock(
            side_effect=[ValueError("1"), ValueError("2"), ValueError("final")]
        )

        with pytest.raises(ValueError, match="final"):
            mixin._with_retry(fn)

        assert fn.call_count == 3

    def test_max_retries_zero_raises_value_error(self):
        """Should immediately raise ValueError when max_retries=0"""
        mixin = self._make_mixin(max_retries=0)
        fn = MagicMock(return_value="ok")

        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            mixin._with_retry(fn)

        fn.assert_not_called()

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_retryable_exceptions_filter(self, mock_sleep):
        """Should immediately raise exceptions not in retryable_exceptions"""
        mixin = self._make_mixin(max_retries=3)
        fn = MagicMock(side_effect=TypeError("not retryable"))

        with pytest.raises(TypeError, match="not retryable"):
            mixin._with_retry(fn, retryable_exceptions=(ValueError,))

        fn.assert_called_once()
        mock_sleep.assert_not_called()


class TestCreateClient:
    """Tests for create_client() factory"""

    @patch.dict("os.environ", {"GCP_PROJECT_ID": "test-project"})
    def test_gemini_model_returns_vertex_ai_client(self):
        """Should return VertexAIClient for gemini model names"""
        client = create_client("gemini-2.5-flash")
        assert isinstance(client, VertexAIClient)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_claude_model_returns_claude_client(self):
        """Should return ClaudeClient for claude model names"""
        client = create_client("claude-sonnet-4-5-20250514")
        assert isinstance(client, ClaudeClient)

    def test_lmstudio_model_returns_lmstudio_client(self):
        """Should return LMStudioClient for lmstudio/ prefixed model names"""
        client = create_client("lmstudio/qwen2.5-7b")
        assert isinstance(client, LMStudioClient)


class TestStripThinking:
    """Tests for _strip_thinking() helper"""

    def test_xml_think_tags_removed(self):
        """Should strip <think>...</think> blocks"""
        raw = "<think>Let me reason...</think>\nThe answer is 42."
        assert _strip_thinking(raw) == "The answer is 42."

    def test_xml_think_tags_multiline(self):
        """Should strip multiline <think> blocks"""
        raw = "<think>\nStep 1: ...\nStep 2: ...\n</think>\n\nResult: yes"
        assert _strip_thinking(raw) == "Result: yes"

    def test_xml_think_returns_raw_if_empty_after_strip(self):
        """Should return original when stripping <think> leaves nothing"""
        raw = "<think>All content is thinking</think>"
        assert _strip_thinking(raw) == "<think>All content is thinking</think>"

    def test_plain_text_thinking_process(self):
        """Should strip 'Thinking Process: ... Final Output:' blocks"""
        raw = (
            "Thinking Process:\n"
            "1. **Analyze:** ...\n"
            "2. **Evaluate:** ...\n\n"
            "Final Output:\n"
            "The answer is 42."
        )
        assert _strip_thinking(raw) == "The answer is 42."

    def test_plain_text_thinking_process_final_answer(self):
        """Should strip 'Thinking Process: ... Final Answer:' blocks"""
        raw = (
            "Thinking Process:\n"
            "Some reasoning here.\n\n"
            "Final Answer:\n"
            "Category: spam"
        )
        assert _strip_thinking(raw) == "Category: spam"

    def test_no_thinking_markers_returns_as_is(self):
        """Should return output unchanged when no thinking markers present"""
        raw = "Just a normal response."
        assert _strip_thinking(raw) == "Just a normal response."

    def test_truncated_thinking_returns_as_is(self):
        """Should return output as-is when thinking is truncated (no Final Output marker)"""
        raw = (
            "Thinking Process:\n"
            "1. Step one...\n"
            "2. Step two that got cut off mid-sente"
        )
        assert _strip_thinking(raw) == raw.strip()

    def test_clean_output_unchanged(self):
        """Should leave clean classification output unchanged"""
        raw = "Category: spam"
        assert _strip_thinking(raw) == "Category: spam"
