"""
Model client base class and retry mixin

Defines the abstract base class inherited by all model clients
and the RetryMixin that consolidates shared retry logic.
"""

import time
from abc import ABC, abstractmethod

from adapt_gauge_core.domain.value_objects import ModelResponse


class RetryMixin:
    """Exponential backoff retry. Subclasses set self.max_retries."""

    max_retries: int = 3

    def _with_retry(self, fn, retryable_exceptions=(Exception,)):
        """
        Execute with exponential backoff retry.

        Args:
            fn: The function to retry (a callable with no arguments)
            retryable_exceptions: Tuple of exception types eligible for retry

        Returns:
            The return value of fn()

        Raises:
            ValueError: If max_retries is less than 1
            Exception: The last exception if max retries are exceeded
        """
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        last_exception: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except retryable_exceptions as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        assert last_exception is not None
        raise last_exception


class ModelClient(ABC):
    """Abstract base class for model clients"""

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse:
        """Send a prompt and retrieve the response"""
        pass
