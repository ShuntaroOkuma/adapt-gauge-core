"""
Vertex AI (Google GenAI SDK) model client
"""

import os
import time

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai.types import GenerateContentConfig, HttpOptions

from adapt_gauge_core.domain.value_objects import ModelResponse
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient, RetryMixin


class VertexAIClient(RetryMixin, ModelClient):
    """Model client using Google GenAI SDK (via Vertex AI)"""

    def __init__(
        self,
        model_name: str,
        project_id: str | None = None,
        location: str | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3
    ):
        """
        Args:
            model_name: Model name (e.g. gemini-2.5-pro, gemini-2.5-flash)
            project_id: GCP project ID (falls back to environment variable if not specified)
            location: Region (falls back to environment variable if not specified)
            timeout_seconds: Timeout in seconds (default: 30)
            max_retries: Maximum number of retries (default: 3)
        """
        self.model_name = model_name
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        # Gemini models are available in us-central1 (some models may not be available in other regions)
        self.location = location or "global"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID is not set")

        # Initialize Google GenAI client (via Vertex AI)
        # Timeout is configured via HttpOptions
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            http_options=HttpOptions(timeout=timeout_seconds * 1000),
        )

        # Set temperature=0 for reproducibility
        self.generation_config = GenerateContentConfig(temperature=0.0)

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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config,
            )
            end_time = time.time()

            latency_ms = int((end_time - start_time) * 1000)

            # Retrieve token usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            return ModelResponse(
                output=response.text.strip(),
                latency_ms=latency_ms,
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return self._with_retry(
            _call,
            retryable_exceptions=(
                google_exceptions.DeadlineExceeded,
                google_exceptions.ServiceUnavailable,
                google_exceptions.ResourceExhausted,
            ),
        )
