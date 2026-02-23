"""
LMStudio (OpenAI-compatible API) model client
"""

import os
import time

import openai
from openai import OpenAI

from adapt_gauge_core.domain.value_objects import ModelResponse
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient, RetryMixin


class LMStudioClient(RetryMixin, ModelClient):
    """Client using LMStudio (OpenAI-compatible API)"""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        max_tokens: int = 1024
    ):
        """
        Args:
            model_name: Model name (e.g. lmstudio/qwen2.5-7b)
            base_url: LMStudio API endpoint (falls back to LMSTUDIO_BASE_URL env var if not specified)
            api_key: API key (falls back to LMSTUDIO_API_KEY env var if not specified; usually not required for LMStudio)
            max_retries: Maximum number of retries (default: 3)
            max_tokens: Maximum number of tokens (default: 1024)
        """
        self.model_name = model_name
        # Strip the lmstudio/ prefix to get the model name for the API
        self.api_model_name = model_name.removeprefix("lmstudio/")
        self.max_retries = max_retries
        self.max_tokens = max_tokens

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
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            end_time = time.time()

            latency_ms = int((end_time - start_time) * 1000)
            output = response.choices[0].message.content.strip()

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
