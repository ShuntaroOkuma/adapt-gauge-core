"""
Domain Constants

Centrally manages constants shared across the evaluation harness.
"""

# Shot count schedule
SHOT_SCHEDULE = [0, 1, 2, 4, 8]

# Default model list
DEFAULT_MODELS = [
    # "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    # "gemini-2.5-pro",
    "gemini-2.5-flash",
    # "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    # "claude-opus-4-5-20251101",
]

# Model pricing (USD / 1M tokens)
MODEL_PRICING = {
    "gemini-3-pro-preview": {"input": 1.25, "output": 10.0},
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
}

# Default pricing for local models (LMStudio, etc.)
_LOCAL_MODEL_PRICING = {"input": 0.0, "output": 0.0}
