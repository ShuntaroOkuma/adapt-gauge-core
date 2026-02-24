"""
Tests for harness_config.py
"""

import pytest

from adapt_gauge_core.harness_config import (
    TrialConfig,
    ReliabilityConfig,
    IsolationConfig,
    LLMJudgeConfig,
    HarnessConfig,
    load_config,
)


class TestTrialConfig:
    """Tests for TrialConfig dataclass"""

    def test_defaults(self):
        config = TrialConfig()
        assert config.num_trials == 3
        assert config.aggregation == "mean"
        assert config.success_threshold == 0.8

    def test_custom_values(self):
        config = TrialConfig(num_trials=5, aggregation="median", success_threshold=0.9)
        assert config.num_trials == 5
        assert config.aggregation == "median"
        assert config.success_threshold == 0.9


class TestReliabilityConfig:
    """Tests for ReliabilityConfig dataclass"""

    def test_defaults(self):
        config = ReliabilityConfig()
        assert config.calculate_pass_at_k is True
        assert config.k_values == [1, 3]

    def test_custom_values(self):
        config = ReliabilityConfig(calculate_pass_at_k=False, k_values=[1, 5, 10])
        assert config.calculate_pass_at_k is False
        assert config.k_values == [1, 5, 10]


class TestIsolationConfig:
    """Tests for IsolationConfig dataclass"""

    def test_defaults(self):
        config = IsolationConfig()
        assert config.new_client_per_trial is True
        assert config.timeout_seconds == 120
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0


class TestLLMJudgeConfig:
    """Tests for LLMJudgeConfig dataclass"""

    def test_defaults(self):
        config = LLMJudgeConfig()
        assert config.grader_model == "gemini-2.5-flash"
        assert config.enabled is False
        assert config.timeout_seconds == 30
        assert config.max_retries == 2
        assert config.fallback_method == "f1"

    def test_custom_values(self):
        config = LLMJudgeConfig(
            grader_model="claude-haiku-4-5-20251001",
            enabled=True,
            timeout_seconds=60,
            max_retries=5,
            fallback_method="exact_match",
        )
        assert config.grader_model == "claude-haiku-4-5-20251001"
        assert config.enabled is True
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
        assert config.fallback_method == "exact_match"


class TestHarnessConfig:
    """Tests for HarnessConfig dataclass"""

    def test_defaults(self):
        config = HarnessConfig()
        assert isinstance(config.trials, TrialConfig)
        assert isinstance(config.reliability, ReliabilityConfig)
        assert isinstance(config.isolation, IsolationConfig)
        assert isinstance(config.llm_judge, LLMJudgeConfig)

    def test_to_dict(self):
        config = HarnessConfig()
        d = config.to_dict()
        assert "harness_config" in d
        assert "trials" in d["harness_config"]
        assert "reliability" in d["harness_config"]
        assert "isolation" in d["harness_config"]
        assert "llm_judge" in d["harness_config"]
        assert d["harness_config"]["trials"]["num_trials"] == 3
        assert d["harness_config"]["llm_judge"]["enabled"] is False

    def test_from_dict_with_key(self):
        data = {
            "harness_config": {
                "trials": {"num_trials": 5},
                "reliability": {"k_values": [1, 5]},
                "isolation": {"timeout_seconds": 60},
            }
        }
        config = HarnessConfig.from_dict(data)
        assert config.trials.num_trials == 5
        assert config.reliability.k_values == [1, 5]
        assert config.isolation.timeout_seconds == 60
        # Default values should be preserved
        assert config.trials.aggregation == "mean"

    def test_from_dict_without_key(self):
        data = {
            "trials": {"num_trials": 7},
        }
        config = HarnessConfig.from_dict(data)
        assert config.trials.num_trials == 7

    def test_from_dict_empty(self):
        config = HarnessConfig.from_dict({})
        assert config.trials.num_trials == 3  # default
        assert config.llm_judge.enabled is False
        assert config.llm_judge.grader_model == "gemini-2.5-flash"

    def test_from_dict_with_llm_judge(self):
        """Config with llm_judge section"""
        data = {
            "harness_config": {
                "trials": {"num_trials": 3},
                "llm_judge": {
                    "grader_model": "claude-haiku-4-5-20251001",
                    "enabled": True,
                    "timeout_seconds": 60,
                },
            }
        }
        config = HarnessConfig.from_dict(data)
        assert config.llm_judge.grader_model == "claude-haiku-4-5-20251001"
        assert config.llm_judge.enabled is True
        assert config.llm_judge.timeout_seconds == 60
        # Default values should be preserved
        assert config.llm_judge.max_retries == 2
        assert config.llm_judge.fallback_method == "f1"

    def test_roundtrip(self):
        original = HarnessConfig(
            trials=TrialConfig(num_trials=5, aggregation="median"),
        )
        data = original.to_dict()
        restored = HarnessConfig.from_dict(data)
        assert restored.trials.num_trials == 5
        assert restored.trials.aggregation == "median"


class TestLoadConfig:
    """Tests for load_config function (environment variable based)"""

    def test_defaults_without_env(self, monkeypatch):
        """Should return default values when no env vars are set"""
        for key in [
            "HARNESS_NUM_TRIALS", "HARNESS_AGGREGATION", "HARNESS_SUCCESS_THRESHOLD",
            "HARNESS_PASS_AT_K", "HARNESS_K_VALUES",
            "HARNESS_NEW_CLIENT_PER_TRIAL", "HARNESS_TIMEOUT_SECONDS",
            "HARNESS_MAX_RETRIES", "HARNESS_RETRY_DELAY_SECONDS",
            "LLM_JUDGE_GRADER_MODEL", "LLM_JUDGE_ENABLED",
            "LLM_JUDGE_TIMEOUT_SECONDS", "LLM_JUDGE_MAX_RETRIES",
            "LLM_JUDGE_FALLBACK_METHOD",
            "LMSTUDIO_BASE_URL", "LMSTUDIO_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = load_config()
        assert isinstance(config, HarnessConfig)
        assert config.trials.num_trials == 3
        assert config.trials.aggregation == "mean"
        assert config.trials.success_threshold == 0.8
        assert config.reliability.calculate_pass_at_k is True
        assert config.reliability.k_values == [1, 3]
        assert config.isolation.new_client_per_trial is True
        assert config.isolation.timeout_seconds == 120
        assert config.isolation.max_retries == 3
        assert config.isolation.retry_delay_seconds == 1.0
        assert config.llm_judge.grader_model == "gemini-2.5-flash"
        assert config.llm_judge.enabled is False
        assert config.llm_judge.timeout_seconds == 30
        assert config.llm_judge.max_retries == 2
        assert config.llm_judge.fallback_method == "f1"
        assert config.lmstudio.base_url == "http://localhost:1234/v1"
        assert config.lmstudio.api_key == "lm-studio"

    def test_custom_env_values(self, monkeypatch):
        """Should load values from environment variables"""
        monkeypatch.setenv("HARNESS_NUM_TRIALS", "5")
        monkeypatch.setenv("HARNESS_AGGREGATION", "median")
        monkeypatch.setenv("HARNESS_SUCCESS_THRESHOLD", "0.9")
        monkeypatch.setenv("HARNESS_PASS_AT_K", "false")
        monkeypatch.setenv("HARNESS_K_VALUES", "1,5,10")
        monkeypatch.setenv("HARNESS_NEW_CLIENT_PER_TRIAL", "false")
        monkeypatch.setenv("HARNESS_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("HARNESS_MAX_RETRIES", "5")
        monkeypatch.setenv("HARNESS_RETRY_DELAY_SECONDS", "2.0")
        monkeypatch.setenv("LLM_JUDGE_GRADER_MODEL", "claude-haiku-4-5-20251001")
        monkeypatch.setenv("LLM_JUDGE_ENABLED", "true")
        monkeypatch.setenv("LLM_JUDGE_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("LLM_JUDGE_MAX_RETRIES", "5")
        monkeypatch.setenv("LLM_JUDGE_FALLBACK_METHOD", "exact_match")
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://custom:5678/v1")
        monkeypatch.setenv("LMSTUDIO_API_KEY", "custom-key")

        config = load_config()
        assert config.trials.num_trials == 5
        assert config.trials.aggregation == "median"
        assert config.trials.success_threshold == 0.9
        assert config.reliability.calculate_pass_at_k is False
        assert config.reliability.k_values == [1, 5, 10]
        assert config.isolation.new_client_per_trial is False
        assert config.isolation.timeout_seconds == 60
        assert config.isolation.max_retries == 5
        assert config.isolation.retry_delay_seconds == 2.0
        assert config.llm_judge.grader_model == "claude-haiku-4-5-20251001"
        assert config.llm_judge.enabled is True
        assert config.llm_judge.timeout_seconds == 60
        assert config.llm_judge.max_retries == 5
        assert config.llm_judge.fallback_method == "exact_match"
        assert config.lmstudio.base_url == "http://custom:5678/v1"
        assert config.lmstudio.api_key == "custom-key"

    def test_partial_env_values(self, monkeypatch):
        """Should use defaults for unset env vars when only some are set"""
        for key in [
            "HARNESS_NUM_TRIALS", "HARNESS_AGGREGATION", "HARNESS_SUCCESS_THRESHOLD",
            "HARNESS_PASS_AT_K", "HARNESS_K_VALUES",
            "HARNESS_NEW_CLIENT_PER_TRIAL", "HARNESS_TIMEOUT_SECONDS",
            "HARNESS_MAX_RETRIES", "HARNESS_RETRY_DELAY_SECONDS",
            "LLM_JUDGE_GRADER_MODEL", "LLM_JUDGE_ENABLED",
            "LLM_JUDGE_TIMEOUT_SECONDS", "LLM_JUDGE_MAX_RETRIES",
            "LLM_JUDGE_FALLBACK_METHOD",
            "LMSTUDIO_BASE_URL", "LMSTUDIO_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("HARNESS_NUM_TRIALS", "10")
        monkeypatch.setenv("LLM_JUDGE_ENABLED", "true")

        config = load_config()
        assert config.trials.num_trials == 10
        assert config.llm_judge.enabled is True
        # Others should remain at defaults
        assert config.trials.aggregation == "mean"
        assert config.llm_judge.grader_model == "gemini-2.5-flash"

    def test_bool_env_variants(self, monkeypatch):
        """Should handle various boolean env var representations"""
        monkeypatch.setenv("LLM_JUDGE_ENABLED", "1")
        config = load_config()
        assert config.llm_judge.enabled is True

        monkeypatch.setenv("LLM_JUDGE_ENABLED", "yes")
        config = load_config()
        assert config.llm_judge.enabled is True

        monkeypatch.setenv("LLM_JUDGE_ENABLED", "TRUE")
        config = load_config()
        assert config.llm_judge.enabled is True

        monkeypatch.setenv("LLM_JUDGE_ENABLED", "false")
        config = load_config()
        assert config.llm_judge.enabled is False

        monkeypatch.setenv("LLM_JUDGE_ENABLED", "0")
        config = load_config()
        assert config.llm_judge.enabled is False

    def test_invalid_int_env_raises_error(self, monkeypatch):
        """Should raise ValueError for invalid int env var"""
        monkeypatch.setenv("HARNESS_NUM_TRIALS", "abc")
        with pytest.raises(ValueError, match="HARNESS_NUM_TRIALS"):
            load_config()

    def test_invalid_float_env_raises_error(self, monkeypatch):
        """Should raise ValueError for invalid float env var"""
        monkeypatch.setenv("HARNESS_SUCCESS_THRESHOLD", "not-a-number")
        with pytest.raises(ValueError, match="HARNESS_SUCCESS_THRESHOLD"):
            load_config()

    def test_invalid_int_list_env_raises_error(self, monkeypatch):
        """Should raise ValueError for invalid int list env var"""
        monkeypatch.setenv("HARNESS_K_VALUES", "1,abc,3")
        with pytest.raises(ValueError, match="HARNESS_K_VALUES"):
            load_config()
