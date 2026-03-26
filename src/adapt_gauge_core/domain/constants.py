"""
Domain Constants

Centrally manages constants shared across the evaluation harness.
"""

# Shot count schedule
SHOT_SCHEDULE = [0, 1, 2, 4, 8]

# Default model list
DEFAULT_MODELS = [
    # Google Gemini
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    # Anthropic Claude
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-6",
    # OpenAI GPT
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

# Model pricing (USD / 1M tokens)
MODEL_PRICING = {
    # Google Gemini
    "gemini-3.1-pro-preview": {"input": 2.50, "output": 15.0},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.0},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # Anthropic Claude
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # Anthropic Claude (legacy)
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
    # OpenAI GPT
    "gpt-5.4": {"input": 3.0, "output": 20.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
    # OpenAI GPT (legacy)
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Google Gemini (legacy)
    "gemini-2.5-flash-lite": {"input": 0.02, "output": 0.10},
}

# Default pricing for local models (LMStudio, etc.)
_LOCAL_MODEL_PRICING = {"input": 0.0, "output": 0.0}
