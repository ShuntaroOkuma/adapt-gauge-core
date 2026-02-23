"""
Model client factory

Creates the appropriate client instance based on the model name.
"""

from adapt_gauge_core.infrastructure.model_clients.base import ModelClient
from adapt_gauge_core.infrastructure.model_clients.vertex_ai import VertexAIClient
from adapt_gauge_core.infrastructure.model_clients.claude import ClaudeClient
from adapt_gauge_core.infrastructure.model_clients.lmstudio import LMStudioClient


def create_client(model_name: str) -> ModelClient:
    """
    Create the appropriate client based on the model name

    Args:
        model_name: Model name

    Returns:
        ModelClient: The appropriate client instance
    """
    if model_name.startswith("lmstudio/"):
        return LMStudioClient(model_name)
    elif model_name.startswith("claude"):
        return ClaudeClient(model_name)
    else:
        return VertexAIClient(model_name)
