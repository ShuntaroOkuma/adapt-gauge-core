"""Tests for domain constants"""

from adapt_gauge_core.domain.constants import (
    DEFAULT_MODELS,
    MODEL_PRICING,
    SHOT_SCHEDULE,
    _LOCAL_MODEL_PRICING,
)


def test_default_models_non_empty():
    """DEFAULT_MODELS should be a non-empty list"""
    assert isinstance(DEFAULT_MODELS, list)
    assert len(DEFAULT_MODELS) > 0


def test_all_default_models_have_pricing():
    """All DEFAULT_MODELS should have entries in MODEL_PRICING"""
    for model in DEFAULT_MODELS:
        assert model in MODEL_PRICING, f"{model} not in MODEL_PRICING"


def test_shot_schedule():
    """SHOT_SCHEDULE should have the correct values"""
    assert SHOT_SCHEDULE == [0, 1, 2, 4, 8]


def test_local_model_pricing():
    """_LOCAL_MODEL_PRICING should have input/output keys both set to 0"""
    assert "input" in _LOCAL_MODEL_PRICING
    assert "output" in _LOCAL_MODEL_PRICING
    assert _LOCAL_MODEL_PRICING["input"] == 0.0
    assert _LOCAL_MODEL_PRICING["output"] == 0.0


def test_model_pricing_has_input_output():
    """Each MODEL_PRICING entry should have input/output keys with non-negative values"""
    for model, pricing in MODEL_PRICING.items():
        assert "input" in pricing, f"{model} missing input pricing"
        assert "output" in pricing, f"{model} missing output pricing"
        assert pricing["input"] >= 0, f"{model} has negative input pricing"
        assert pricing["output"] >= 0, f"{model} has negative output pricing"
