"""
OpenAI API model client
"""

import os
import re
import time

import openai
from openai import OpenAI

from adapt_gauge_core.domain.value_objects import ModelResponse
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient, RetryMixin


class OpenAIClient(RetryMixin, ModelClient):
    """Client for the OpenAI API (GPT-4o, GPT-5.4, etc.)"""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
    ):
        """
        Args:
            model_name: Model name (e.g. gpt-4o-mini, gpt-5.4-mini)
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            max_retries: Maximum number of retries (default: 3)
            retry_delay_seconds: Base delay between retries in seconds (default: 5.0)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=self.api_key)

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
            # GPT-5 base models (gpt-5, gpt-5-mini, gpt-5-nano) reject
            # the temperature parameter entirely.
            is_gpt5_base = bool(
                re.match(r"^gpt-5(-mini|-nano)?$", self.model_name)
            )
            # GPT-5 family models require max_completion_tokens
            # instead of max_tokens.
            is_gpt5_family = self.model_name.startswith("gpt-5")
            token_limit_key = "max_completion_tokens" if is_gpt5_family else "max_tokens"
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                token_limit_key: 1024,
            }
            if not is_gpt5_base:
                params["temperature"] = 0.0

            response = self.client.chat.completions.create(**params)
            end_time = time.time()

            latency_ms = int((end_time - start_time) * 1000)
            raw_output = response.choices[0].message.content.strip()

            # Strip <think>...</think> blocks from reasoning models
            output = re.sub(r"<think>[\s\S]*?</think>\s*", "", raw_output).strip()

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
