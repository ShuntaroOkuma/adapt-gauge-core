"""
Model client package

Provides a unified interface to each LLM provider.
"""

from adapt_gauge_core.infrastructure.model_clients.base import ModelClient
from adapt_gauge_core.infrastructure.model_clients.factory import create_client
from adapt_gauge_core.domain.value_objects import ModelResponse

__all__ = ["ModelClient", "ModelResponse", "create_client"]
