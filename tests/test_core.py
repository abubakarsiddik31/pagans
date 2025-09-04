"""
Unit tests for the core module.

This module contains tests for the PromptOptimizer class and its core functionality.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.pagans.core import PromptOptimizer
from src.pagans.exceptions import (
    ConfigurationError,
    ModelNotFoundError,
    PromptOptimizerError,
)
from src.pagans.models import ModelFamily, OptimizationResult


class TestPromptOptimizerInitialization:
    """Test cases for PromptOptimizer initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        api_key = "test-api-key"
        optimizer = PromptOptimizer(api_key=api_key)

        assert optimizer.api_key == api_key
        assert optimizer.default_model == "openai/gpt-4o-mini"
        assert optimizer.client is not None

    def test_init_with_environment_variables(self):
        """Test initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "env-api-key",
                "OPENROUTER_BASE_URL": "https://custom.api.url",
                "DEFAULT_OPTIMIZER_MODEL": "custom-model",
            },
        ):
            optimizer = PromptOptimizer()

            assert optimizer.api_key == "env-api-key"
            assert optimizer.base_url == "https://custom.api.url"
            assert optimizer.default_model == "custom-model"

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="API key is required"):
                PromptOptimizer()

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        api_key = "test-api-key"
        base_url = "https://custom.api.url"
        default_model = "custom-model"
        timeout = 30.0
        max_retries = 5
        retry_delay = 2.0

        optimizer = PromptOptimizer(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        assert optimizer.api_key == api_key
        assert optimizer.base_url == base_url
        assert optimizer.default_model == default_model


class TestPromptOptimizerOptimization:
    """Test cases for prompt optimization functionality."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer for testing."""
        with patch("src.pagans.core.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            optimizer = PromptOptimizer(api_key="test-api-key")
            optimizer.client = mock_client

            return optimizer

    @pytest.fixture
    def mock_optimization_result(self):
        """Create a mock optimization result."""
        return OptimizationResult(
            original="Original prompt",
            optimized="Optimized prompt",
            target_model="gpt-4o",
            target_family=ModelFamily.OPENAI,
            optimization_notes="Test optimization",
            optimization_time=1.5,
        )

    def test_optimize_success(self, mock_optimizer, mock_optimization_result):
        """Test successful prompt optimization."""
        # Mock the optimization process
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        # Mock time.time() to simulate elapsed time
        time_values = [1000.0, 1001.5]  # Start time and end time
        with patch("time.time", side_effect=time_values):
            result = asyncio.run(
                mock_optimizer.optimize(
                    prompt="Original prompt",
                    target_model="gpt-4o",
                    optimization_notes="Test optimization",
                )
            )

        # Verify the result
        assert result.original == "Original prompt"
        assert result.optimized == "Optimized prompt"
        assert result.target_model == "gpt-4o"
        assert result.target_family == ModelFamily.OPENAI
        assert result.optimization_notes == "Test optimization"
        assert result.optimization_time == 1.5

    def test_optimize_with_default_model(self, mock_optimizer):
        """Test optimization with default model."""
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        with patch("time.time", return_value=1000.0):
            result = asyncio.run(
                mock_optimizer.optimize(
                    prompt="Original prompt",
                )
            )

        assert result.target_model == "openai/gpt-4o-mini"

    def test_optimize_empty_prompt(self, mock_optimizer):
        """Test optimization with empty prompt raises error."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            asyncio.run(mock_optimizer.optimize(prompt=""))

    def test_optimize_unsupported_model(self, mock_optimizer):
        """Test optimization with unsupported model raises error."""
        with pytest.raises(ModelNotFoundError, match="invalid-model"):
            asyncio.run(
                mock_optimizer.optimize(
                    prompt="Original prompt",
                    target_model="invalid-model",
                )
            )

    def test_optimize_api_error(self, mock_optimizer):
        """Test optimization with API error raises error."""
        mock_optimizer.client.optimize_prompt.side_effect = Exception("API Error")

        with pytest.raises(PromptOptimizerError, match="Failed to optimize prompt"):
            asyncio.run(
                mock_optimizer.optimize(
                    prompt="Original prompt",
                    target_model="gpt-4o",
                )
            )

    def test_optimize_with_cache(self, mock_optimizer, mock_optimization_result):
        """Test optimization with caching."""
        # Mock the optimization process
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        # Add result to cache
        cache_key = "Original prompt:gpt-4o:Test optimization"
        mock_optimizer._optimization_cache[cache_key] = mock_optimization_result

        # Test cached result
        result = asyncio.run(
            mock_optimizer.optimize(
                prompt="Original prompt",
                target_model="gpt-4o",
                optimization_notes="Test optimization",
                use_cache=True,
            )
        )

        # Verify the cached result is returned
        assert result is mock_optimization_result
        # Verify API was not called
        mock_optimizer.client.optimize_prompt.assert_not_called()

    def test_optimize_without_cache(self, mock_optimizer):
        """Test optimization without caching."""
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        with patch("time.time", return_value=1000.0):
            result = asyncio.run(
                mock_optimizer.optimize(
                    prompt="Original prompt",
                    target_model="gpt-4o",
                    optimization_notes="Test optimization",
                    use_cache=False,
                )
            )

        # Verify API was called
        mock_optimizer.client.optimize_prompt.assert_called_once()

    def test_optimize_multiple_prompts(self, mock_optimizer):
        """Test optimization of multiple prompts."""
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch("time.time", return_value=1000.0):
            results = asyncio.run(
                mock_optimizer.optimize_multiple(
                    prompts=prompts,
                    target_model="gpt-4o",
                )
            )

        assert len(results) == 3
        for result in results:
            assert result.optimized == "Optimized prompt"
            assert result.target_model == "gpt-4o"

        # Verify API was called for each prompt
        assert mock_optimizer.client.optimize_prompt.call_count == 3

    def test_optimize_multiple_with_concurrency_limit(self, mock_optimizer):
        """Test optimization with concurrency limit."""
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]

        with patch("time.time", return_value=1000.0):
            results = asyncio.run(
                mock_optimizer.optimize_multiple(
                    prompts=prompts,
                    target_model="gpt-4o",
                    max_concurrent=2,
                )
            )

        assert len(results) == 5
        # Verify API was called for each prompt
        assert mock_optimizer.client.optimize_prompt.call_count == 5

    def test_compare_optimizations(self, mock_optimizer):
        """Test comparison of optimizations across different models."""
        mock_optimizer.client.optimize_prompt.return_value = "Optimized prompt"

        models = ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"]

        with patch("time.time", return_value=1000.0):
            results = asyncio.run(
                mock_optimizer.compare_optimizations(
                    prompt="Test prompt",
                    target_models=models,
                )
            )

        assert len(results) == 3
        for model, result in results.items():
            assert result.optimized == "Optimized prompt"
            assert result.target_model == model

    def test_compare_optimizations_with_error(self, mock_optimizer):
        """Test comparison with some models failing."""
        mock_optimizer.client.optimize_prompt.side_effect = [
            "Optimized prompt",
            Exception("API Error"),
            "Another optimized prompt",
        ]

        models = ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"]

        results = asyncio.run(
            mock_optimizer.compare_optimizations(
                prompt="Test prompt",
                target_models=models,
            )
        )

        assert len(results) == 3
        assert results["gpt-4o"].optimized == "Optimized prompt"
        assert isinstance(results["claude-3.5-sonnet"], PromptOptimizerError)
        assert results["gemini-1.5-pro"].optimized == "Another optimized prompt"

    def test_get_optimization_description(self, mock_optimizer):
        """Test getting optimization description."""
        description = mock_optimizer.get_optimization_description("gpt-4o")
        assert isinstance(description, str)
        assert len(description) > 0

    def test_get_optimization_description_invalid_model(self, mock_optimizer):
        """Test getting optimization description for invalid model."""
        with pytest.raises(ModelNotFoundError):
            mock_optimizer.get_optimization_description("invalid-model")

    def test_is_model_supported(self, mock_optimizer):
        """Test checking if model is supported."""
        assert mock_optimizer.is_model_supported("gpt-4o")
        assert not mock_optimizer.is_model_supported("invalid-model")

    def test_validate_model(self, mock_optimizer):
        """Test model validation."""
        mock_optimizer.client.validate_model.return_value = True

        result = asyncio.run(mock_optimizer.validate_model("gpt-4o"))
        assert result is True

        mock_optimizer.client.validate_model.return_value = False

        result = asyncio.run(mock_optimizer.validate_model("gpt-4o"))
        assert result is False

    def test_clear_cache(self, mock_optimizer):
        """Test clearing the cache."""
        # Add something to cache
        mock_optimizer._optimization_cache["test"] = "test-value"

        # Clear cache
        mock_optimizer.clear_cache()

        # Verify cache is empty
        assert len(mock_optimizer._optimization_cache) == 0

    def test_get_cache_size(self, mock_optimizer):
        """Test getting cache size."""
        # Initially empty
        assert mock_optimizer.get_cache_size() == 0

        # Add something to cache
        mock_optimizer._optimization_cache["test"] = "test-value"

        # Verify size
        assert mock_optimizer.get_cache_size() == 1

    def test_context_manager(self, mock_optimizer):
        """Test using PromptOptimizer as context manager."""
        mock_optimizer.client.close = AsyncMock()

        async def test_context():
            async with PromptOptimizer(api_key="test-api-key") as optimizer:
                assert optimizer is not None
                return optimizer

        optimizer = asyncio.run(test_context())
        mock_optimizer.client.close.assert_called_once()

    def test_close(self, mock_optimizer):
        """Test closing the optimizer."""
        mock_optimizer.client.close = AsyncMock()

        asyncio.run(mock_optimizer.close())
        mock_optimizer.client.close.assert_called_once()
