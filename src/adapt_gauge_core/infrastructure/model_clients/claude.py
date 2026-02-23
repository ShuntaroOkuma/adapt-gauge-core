"""
Anthropic Claude model client
"""

import os
import time

from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError

from adapt_gauge_core.domain.value_objects import ModelResponse
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient, RetryMixin


class ClaudeClient(RetryMixin, ModelClient):
    """Claude client using the Anthropic API"""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_retries: int = 3
    ):
        """
        Args:
            model_name: Model name (e.g. claude-sonnet-4-5-20250514)
            api_key: Anthropic API key (falls back to environment variable if not specified)
            max_retries: Maximum number of retries (default: 3)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        # Initialize the Anthropic client
        self.client = Anthropic(api_key=self.api_key)

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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()

            latency_ms = int((end_time - start_time) * 1000)
            output = response.content[0].text.strip()

            # Retrieve token usage
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0

            return ModelResponse(
                output=output,
                latency_ms=latency_ms,
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return self._with_retry(
            _call,
            retryable_exceptions=(APIConnectionError, RateLimitError, APIStatusError),
        )
