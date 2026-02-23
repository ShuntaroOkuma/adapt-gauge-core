"""
Model Client (backward compatibility shim)

Implementation has been moved to adapt_gauge_core.infrastructure.model_clients.
This is a re-export module to maintain existing import paths.
"""

from adapt_gauge_core.infrastructure.model_clients.base import ModelClient  # noqa: F401
from adapt_gauge_core.infrastructure.model_clients.vertex_ai import VertexAIClient  # noqa: F401
from adapt_gauge_core.infrastructure.model_clients.claude import ClaudeClient  # noqa: F401
from adapt_gauge_core.infrastructure.model_clients.lmstudio import LMStudioClient  # noqa: F401
from adapt_gauge_core.infrastructure.model_clients.factory import create_client  # noqa: F401
from adapt_gauge_core.domain.value_objects import ModelResponse  # noqa: F401
