"""
LMStudio (OpenAI-compatible API) model client
"""

import os
import re
import time

import openai
from openai import OpenAI

from adapt_gauge_core.domain.value_objects import ModelResponse
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient, RetryMixin

# Pattern: plain-text thinking markers followed by a final output marker
_PLAIN_THINKING_RE = re.compile(
    r"^(?:Thinking Process|思考プロセス)\s*:?\s*\n[\s\S]*?"
    r"(?:Final (?:Output|Answer)|最終(?:出力|回答))\s*:?\s*\n",
    re.IGNORECASE,
)


def _strip_thinking(raw_output: str) -> str:
    """Strip thinking blocks from model output.

    Handles two formats:
    1. XML ``<think>...</think>`` tags (Qwen 3.5 with temperature > 0)
    2. Plain-text ``Thinking Process: ... Final Output:`` blocks (fallback)

    Returns the original output if stripping would produce an empty string.
    """
    # 1. XML <think> tags
    stripped = re.sub(r"<think>[\s\S]*?</think>\s*", "", raw_output).strip()
    if stripped and stripped != raw_output.strip():
        return stripped

    # 2. Plain-text thinking markers
    stripped = _PLAIN_THINKING_RE.sub("", raw_output).strip()
    if stripped:
        return stripped

    return raw_output.strip()


class LMStudioClient(RetryMixin, ModelClient):
    """Client using LMStudio (OpenAI-compatible API)"""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """
        Args:
            model_name: Model name (e.g. lmstudio/qwen2.5-7b)
            base_url: LMStudio API endpoint (falls back to LMSTUDIO_BASE_URL env var if not specified)
            api_key: API key (falls back to LMSTUDIO_API_KEY env var if not specified; usually not required for LMStudio)
            max_retries: Maximum number of retries (default: 3)
            retry_delay_seconds: Base delay between retries in seconds (default: 5.0)
            max_tokens: Maximum number of tokens (default: 1024)
            temperature: Sampling temperature (default: 0.0; >= 0.6 recommended for thinking models)
        """
        self.model_name = model_name
        # Strip the lmstudio/ prefix to get the model name for the API
        self.api_model_name = model_name.removeprefix("lmstudio/")
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Configuration priority: argument > environment variable > default value
        base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = api_key or os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str) -> ModelResponse:
        """
        Send a prompt and retrieve the response

        Args:
            prompt: Input prompt

        Returns:
            ModelResponse: The model's response

        Raises:
            Exception: If the maximum number of retries is exceeded
        """
        def _call():
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            end_time = time.time()

            latency_ms = int((end_time - start_time) * 1000)
            raw_output = response.choices[0].message.content.strip()

            output = _strip_thinking(raw_output)

            # Retrieve token usage
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0

            return ModelResponse(
                output=output,
                latency_ms=latency_ms,
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return self._with_retry(
            _call,
            retryable_exceptions=(
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.APIStatusError,
            ),
        )
