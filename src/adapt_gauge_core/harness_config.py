"""
Evaluation Harness Configuration

Manages loading from environment variables and default values.
"""

import os
from dataclasses import dataclass, field, asdict


def _env_bool(key: str, default: bool) -> bool:
    """Convert an environment variable to bool"""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Convert an environment variable to int"""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"The value '{val}' of environment variable '{key}' cannot be converted to an integer.")


def _env_float(key: str, default: float) -> float:
    """Convert an environment variable to float"""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"The value '{val}' of environment variable '{key}' cannot be converted to a number.")


def _env_str(key: str, default: str) -> str:
    """Get an environment variable as a string"""
    return os.environ.get(key, default)


def _env_int_list(key: str, default: list[int]) -> list[int]:
    """Convert an environment variable to a comma-separated list of ints"""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"The value '{val}' of environment variable '{key}' cannot be converted to a comma-separated list of integers.")


@dataclass
class TrialConfig:
    """Trial configuration"""
    num_trials: int = 3
    aggregation: str = "mean"  # mean / median
    success_threshold: float = 0.8


@dataclass
class ReliabilityConfig:
    """Reliability metrics configuration"""
    calculate_pass_at_k: bool = True
    k_values: list[int] = field(default_factory=lambda: [1, 3])


@dataclass
class IsolationConfig:
    """Environment isolation configuration"""
    new_client_per_trial: bool = True
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class LLMJudgeConfig:
    """LLM grader configuration"""
    grader_model: str = "gemini-2.5-flash"
    enabled: bool = False
    timeout_seconds: int = 30
    max_retries: int = 2
    fallback_method: str = "f1"


@dataclass
class LMStudioConfig:
    """LMStudio (OpenAI-compatible local LLM) configuration"""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"


@dataclass
class HarnessConfig:
    """Overall evaluation harness configuration"""
    trials: TrialConfig = field(default_factory=TrialConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {"harness_config": asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "HarnessConfig":
        """Create from dictionary (handles presence/absence of harness_config key)"""
        config_data = data.get("harness_config", data)
        trials = TrialConfig(**config_data.get("trials", {}))
        reliability = ReliabilityConfig(**config_data.get("reliability", {}))
        isolation = IsolationConfig(**config_data.get("isolation", {}))
        llm_judge = LLMJudgeConfig(**config_data.get("llm_judge", {}))
        lmstudio = LMStudioConfig(**config_data.get("lmstudio", {}))
        return cls(
            trials=trials,
            reliability=reliability,
            isolation=isolation,
            llm_judge=llm_judge,
            lmstudio=lmstudio,
        )


def load_config() -> HarnessConfig:
    """
    Load configuration from environment variables

    Uses default values when environment variables are not set.

    Returns:
        HarnessConfig
    """
    trials = TrialConfig(
        num_trials=_env_int("HARNESS_NUM_TRIALS", 3),
        aggregation=_env_str("HARNESS_AGGREGATION", "mean"),
        success_threshold=_env_float("HARNESS_SUCCESS_THRESHOLD", 0.8),
    )
    reliability = ReliabilityConfig(
        calculate_pass_at_k=_env_bool("HARNESS_PASS_AT_K", True),
        k_values=_env_int_list("HARNESS_K_VALUES", [1, 3]),
    )
    isolation = IsolationConfig(
        new_client_per_trial=_env_bool("HARNESS_NEW_CLIENT_PER_TRIAL", True),
        timeout_seconds=_env_int("HARNESS_TIMEOUT_SECONDS", 120),
        max_retries=_env_int("HARNESS_MAX_RETRIES", 3),
        retry_delay_seconds=_env_float("HARNESS_RETRY_DELAY_SECONDS", 1.0),
    )
    llm_judge = LLMJudgeConfig(
        grader_model=_env_str("LLM_JUDGE_GRADER_MODEL", "gemini-2.5-flash"),
        enabled=_env_bool("LLM_JUDGE_ENABLED", False),
        timeout_seconds=_env_int("LLM_JUDGE_TIMEOUT_SECONDS", 30),
        max_retries=_env_int("LLM_JUDGE_MAX_RETRIES", 2),
        fallback_method=_env_str("LLM_JUDGE_FALLBACK_METHOD", "f1"),
    )
    lmstudio = LMStudioConfig(
        base_url=_env_str("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key=_env_str("LMSTUDIO_API_KEY", "lm-studio"),
    )
    return HarnessConfig(
        trials=trials,
        reliability=reliability,
        isolation=isolation,
        llm_judge=llm_judge,
        lmstudio=lmstudio,
    )
