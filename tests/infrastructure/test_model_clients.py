"""
モデルクライアントのテスト

RetryMixin._with_retry() のリトライ動作と
create_client() のファクトリ分岐をテストする。
"""

import pytest
from unittest.mock import patch, MagicMock

from adapt_gauge_core.infrastructure.model_clients.base import RetryMixin
from adapt_gauge_core.infrastructure.model_clients.factory import create_client
from adapt_gauge_core.infrastructure.model_clients.vertex_ai import VertexAIClient
from adapt_gauge_core.infrastructure.model_clients.claude import ClaudeClient
from adapt_gauge_core.infrastructure.model_clients.lmstudio import LMStudioClient


class TestRetryMixin:
    """RetryMixin._with_retry() のテスト"""

    def _make_mixin(self, max_retries=3):
        mixin = RetryMixin()
        mixin.max_retries = max_retries
        return mixin

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_success_on_first_attempt(self, mock_sleep):
        """初回で成功する場合、リトライなしで値を返す"""
        mixin = self._make_mixin()
        fn = MagicMock(return_value="ok")

        result = mixin._with_retry(fn)

        assert result == "ok"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_success_after_two_failures(self, mock_sleep):
        """2回失敗後、3回目で成功する場合"""
        mixin = self._make_mixin(max_retries=3)
        fn = MagicMock(side_effect=[ValueError("1"), ValueError("2"), "ok"])

        result = mixin._with_retry(fn)

        assert result == "ok"
        assert fn.call_count == 3
        # 指数バックオフ: sleep(1), sleep(2)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 2 ** 0
        mock_sleep.assert_any_call(2)  # 2 ** 1

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        """全リトライ失敗時、最後の例外をraiseする"""
        mixin = self._make_mixin(max_retries=3)
        fn = MagicMock(
            side_effect=[ValueError("1"), ValueError("2"), ValueError("final")]
        )

        with pytest.raises(ValueError, match="final"):
            mixin._with_retry(fn)

        assert fn.call_count == 3

    def test_max_retries_zero_raises_value_error(self):
        """max_retries=0 の場合、ValueError を即座にraiseする"""
        mixin = self._make_mixin(max_retries=0)
        fn = MagicMock(return_value="ok")

        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            mixin._with_retry(fn)

        fn.assert_not_called()

    @patch("adapt_gauge_core.infrastructure.model_clients.base.time.sleep")
    def test_retryable_exceptions_filter(self, mock_sleep):
        """retryable_exceptions に含まれない例外は即座にraiseされる"""
        mixin = self._make_mixin(max_retries=3)
        fn = MagicMock(side_effect=TypeError("not retryable"))

        with pytest.raises(TypeError, match="not retryable"):
            mixin._with_retry(fn, retryable_exceptions=(ValueError,))

        fn.assert_called_once()
        mock_sleep.assert_not_called()


class TestCreateClient:
    """create_client() ファクトリのテスト"""

    @patch.dict("os.environ", {"GCP_PROJECT_ID": "test-project"})
    def test_gemini_model_returns_vertex_ai_client(self):
        """geminiモデル名の場合、VertexAIClientを返す"""
        client = create_client("gemini-2.5-flash")
        assert isinstance(client, VertexAIClient)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_claude_model_returns_claude_client(self):
        """claudeモデル名の場合、ClaudeClientを返す"""
        client = create_client("claude-sonnet-4-5-20250514")
        assert isinstance(client, ClaudeClient)

    def test_lmstudio_model_returns_lmstudio_client(self):
        """lmstudio/プレフィックスの場合、LMStudioClientを返す"""
        client = create_client("lmstudio/qwen2.5-7b")
        assert isinstance(client, LMStudioClient)
