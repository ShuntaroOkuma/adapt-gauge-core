"""
Model client factory

Creates the appropriate client instance based on the model name.
"""

from __future__ import annotations

from adapt_gauge_core.harness_config import HarnessConfig, load_config
from adapt_gauge_core.infrastructure.model_clients.base import ModelClient
from adapt_gauge_core.infrastructure.model_clients.vertex_ai import VertexAIClient
from adapt_gauge_core.infrastructure.model_clients.claude import ClaudeClient
from adapt_gauge_core.infrastructure.model_clients.lmstudio import LMStudioClient


def create_client(model_name: str, config: HarnessConfig | None = None) -> ModelClient:
    """
    Create the appropriate client based on the model name

    Args:
        model_name: Model name
        config: HarnessConfig (loads from env if not provided)

    Returns:
        ModelClient: The appropriate client instance
    """
    if config is None:
        config = load_config()

    timeout = config.isolation.timeout_seconds
    retries = config.isolation.max_retries
    retry_delay = config.isolation.retry_delay_seconds

    if model_name.startswith("lmstudio/"):
        return LMStudioClient(model_name, max_retries=retries, retry_delay_seconds=retry_delay)
    elif model_name.startswith("claude"):
        return ClaudeClient(model_name, max_retries=retries, retry_delay_seconds=retry_delay)
    else:
        return VertexAIClient(model_name, timeout_seconds=timeout, max_retries=retries, retry_delay_seconds=retry_delay)
