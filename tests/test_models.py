"""
Unit tests for the models module.

This module contains tests for the ModelFamily enum, model name mappings,
and data structures for optimization requests and results.
"""

import pytest

from src.pagans.models import (
    FAMILY_MODEL_MAPPINGS,
    MODEL_NAME_MAPPINGS,
    ModelFamily,
    OptimizationRequest,
    OptimizationResult,
    detect_model_family,
    get_supported_models,
    is_supported_model,
)


class TestModelFamily:
    """Test cases for the ModelFamily enum."""

    def test_model_family_values(self):
        """Test that ModelFamily enum has correct values."""
        assert ModelFamily.OPENAI.value == "openai"
        assert ModelFamily.ANTHROPIC.value == "anthropic"
        assert ModelFamily.GOOGLE.value == "google"
        assert ModelFamily.META.value == "meta"
        assert ModelFamily.MISTRAL.value == "mistral"

    def test_model_family_names(self):
        """Test that ModelFamily enum has correct names."""
        assert ModelFamily.OPENAI.name == "OPENAI"
        assert ModelFamily.ANTHROPIC.name == "ANTHROPIC"
        assert ModelFamily.GOOGLE.name == "GOOGLE"
        assert ModelFamily.META.name == "META"
        assert ModelFamily.MISTRAL.name == "MISTRAL"


class TestOptimizationResult:
    """Test cases for the OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult instance."""
        result = OptimizationResult(
            original="Original prompt",
            optimized="Optimized prompt",
            target_model="gpt-4o",
            target_family=ModelFamily.OPENAI,
            optimization_notes="Test optimization",
            optimization_time=1.5,
        )

        assert result.original == "Original prompt"
        assert result.optimized == "Optimized prompt"
        assert result.target_model == "gpt-4o"
        assert result.target_family == ModelFamily.OPENAI
        assert result.optimization_notes == "Test optimization"
        assert result.optimization_time == 1.5

    def test_optimization_result_optional_fields(self):
        """Test OptimizationResult with optional fields."""
        result = OptimizationResult(
            original="Original prompt",
            optimized="Optimized prompt",
            target_model="gpt-4o",
            target_family=ModelFamily.OPENAI,
        )

        assert result.optimization_notes is None
        assert result.optimization_time is None


class TestOptimizationRequest:
    """Test cases for the OptimizationRequest dataclass."""

    def test_optimization_request_creation(self):
        """Test creating an OptimizationRequest instance."""
        request = OptimizationRequest(
            prompt="Test prompt",
            target_model="gpt-4o",
            optimization_notes="Test notes",
        )

        assert request.prompt == "Test prompt"
        assert request.target_model == "gpt-4o"
        assert request.optimization_notes == "Test notes"

    def test_optimization_request_optional_fields(self):
        """Test OptimizationRequest with optional fields."""
        request = OptimizationRequest(
            prompt="Test prompt",
            target_model="gpt-4o",
        )

        assert request.optimization_notes is None


class TestModelDetection:
    """Test cases for model family detection."""

    def test_detect_openai_models(self):
        """Test detection of OpenAI models."""
        assert detect_model_family("gpt-5") == ModelFamily.OPENAI
        assert detect_model_family("gpt-4.1") == ModelFamily.OPENAI
        assert detect_model_family("gpt-4o") == ModelFamily.OPENAI
        assert detect_model_family("gpt-4-turbo") == ModelFamily.OPENAI
        assert detect_model_family("gpt-3.5-turbo") == ModelFamily.OPENAI

    def test_detect_anthropic_models(self):
        """Test detection of Anthropic models."""
        assert detect_model_family("claude-4") == ModelFamily.ANTHROPIC
        assert detect_model_family("claude-4.1") == ModelFamily.ANTHROPIC
        assert detect_model_family("claude-3.5-sonnet") == ModelFamily.ANTHROPIC
        assert detect_model_family("claude-3-opus") == ModelFamily.ANTHROPIC
        assert detect_model_family("claude-3-haiku") == ModelFamily.ANTHROPIC

    def test_detect_google_models(self):
        """Test detection of Google models."""
        assert detect_model_family("gemini-2.5-pro") == ModelFamily.GOOGLE
        assert detect_model_family("gemini-2.5-flash") == ModelFamily.GOOGLE
        assert detect_model_family("gemini-1.5-pro") == ModelFamily.GOOGLE
        assert detect_model_family("gemini-1.5-flash") == ModelFamily.GOOGLE

    def test_detect_meta_models(self):
        """Test detection of Meta models."""
        assert detect_model_family("llama-3.1-70b") == ModelFamily.META
        assert detect_model_family("llama-3.1-8b") == ModelFamily.META
        assert detect_model_family("llama-3-70b") == ModelFamily.META
        assert detect_model_family("llama-3-8b") == ModelFamily.META

    def test_detect_mistral_models(self):
        """Test detection of Mistral models."""
        assert detect_model_family("mixtral-8x7b") == ModelFamily.MISTRAL
        assert detect_model_family("mistral-7b-instruct") == ModelFamily.MISTRAL

    def test_detect_invalid_model(self):
        """Test detection of invalid model names."""
        with pytest.raises(ValueError):
            detect_model_family("invalid-model")

        with pytest.raises(ValueError):
            detect_model_family("unknown-model-123")

        with pytest.raises(ValueError):
            detect_model_family("")

        with pytest.raises(ValueError):
            detect_model_family(None)


class TestModelSupport:
    """Test cases for model support checking."""

    def test_is_supported_model_true(self):
        """Test that supported models return True."""
        assert is_supported_model("gpt-4o")
        assert is_supported_model("claude-3.5-sonnet")
        assert is_supported_model("gemini-1.5-pro")
        assert is_supported_model("llama-3.1-70b")
        assert is_supported_model("mixtral-8x7b")

    def test_is_supported_model_false(self):
        """Test that unsupported models return False."""
        assert not is_supported_model("invalid-model")
        assert not is_supported_model("unknown-model-123")
        assert not is_supported_model("")
        assert not is_supported_model(None)

    def test_get_supported_models(self):
        """Test getting supported models organized by family."""
        supported_models = get_supported_models()

        # Check that all families are present
        assert ModelFamily.OPENAI in supported_models
        assert ModelFamily.ANTHROPIC in supported_models
        assert ModelFamily.GOOGLE in supported_models
        assert ModelFamily.META in supported_models
        assert ModelFamily.MISTRAL in supported_models

        # Check that each family has models
        for family, models in supported_models.items():
            assert isinstance(models, list)
            assert len(models) > 0

    def test_model_name_mappings(self):
        """Test that model name mappings are correctly structured."""
        # Check that MODEL_NAME_MAPPINGS is a dictionary
        assert isinstance(MODEL_NAME_MAPPINGS, dict)

        # Check that all supported models are in the mappings
        for family, models in get_supported_models().items():
            for model in models:
                assert model in MODEL_NAME_MAPPINGS
                assert MODEL_NAME_MAPPINGS[model] == family

    def test_family_model_mappings(self):
        """Test that family model mappings are correctly structured."""
        # Check that FAMILY_MODEL_MAPPINGS is a dictionary
        assert isinstance(FAMILY_MODEL_MAPPINGS, dict)

        # Check that all families are present
        assert ModelFamily.OPENAI.value in FAMILY_MODEL_MAPPINGS
        assert ModelFamily.ANTHROPIC.value in FAMILY_MODEL_MAPPINGS
        assert ModelFamily.GOOGLE.value in FAMILY_MODEL_MAPPINGS
        assert ModelFamily.META.value in FAMILY_MODEL_MAPPINGS
        assert ModelFamily.MISTRAL.value in FAMILY_MODEL_MAPPINGS

        # Check that each family has a list of models
        for family, models in FAMILY_MODEL_MAPPINGS.items():
            assert isinstance(models, list)
            assert len(models) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_detection(self):
        """Test detection with empty string."""
        with pytest.raises(ValueError):
            detect_model_family("")

    def test_none_detection(self):
        """Test detection with None."""
        with pytest.raises(ValueError):
            detect_model_family(None)

    def test_case_sensitivity(self):
        """Test that model detection is case insensitive."""
        # These should all work regardless of case
        assert is_supported_model("GPT-4O")
        assert is_supported_model("CLAUDE-3.5-SONNET")
        assert is_supported_model("GEMINI-1.5-PRO")
        assert is_supported_model("LLAMA-3.1-70B")
        assert is_supported_model("MIXTRAL-8X7B")

    def test_whitespace_handling(self):
        """Test that model detection handles whitespace."""
        # These should all work with whitespace
        assert is_supported_model(" gpt-4o ")
        assert is_supported_model("  claude-3.5-sonnet  ")
        assert is_supported_model("  gemini-1.5-pro  ")
        assert is_supported_model("  llama-3.1-70b  ")
        assert is_supported_model("  mixtral-8x7b  ")
