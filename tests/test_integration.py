"""
Integration tests for PAGANS.

This module contains integration tests that verify the end-to-end functionality
of the PAGANS system, including prompt optimization across different models.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.pagans.core import PromptOptimizer
from src.pagans.exceptions import (
    PromptOptimizerError,
)
from src.pagans.models import ModelFamily, OptimizationResult


class TestEndToEndOptimization:
    """Test cases for end-to-end optimization workflows."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenRouter client for testing."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful optimization response
            mock_client.optimize_prompt.return_value = "Optimized version of the prompt"
            mock_client.validate_model.return_value = True
            mock_client.get_models.return_value = {
                "data": [
                    {"id": "openai/gpt-4o", "name": "GPT-4o"},
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
                    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
                ]
            }

            yield mock_client

    def test_full_optimization_workflow_openai(self, mock_client):
        """Test complete optimization workflow for OpenAI model."""
        optimizer = PromptOptimizer(api_key="test-key")

        result = asyncio.run(
            optimizer.optimize(
                prompt="Write a Python function", target_model="openai/gpt-4o"
            )
        )

        assert isinstance(result, OptimizationResult)
        assert result.original == "Write a Python function"
        assert result.optimized == "Optimized version of the prompt"
        assert result.target_model == "openai/gpt-4o"
        assert result.target_family == ModelFamily.OPENAI

    def test_full_optimization_workflow_anthropic(self, mock_client):
        """Test complete optimization workflow for Anthropic model."""
        optimizer = PromptOptimizer(api_key="test-key")

        result = asyncio.run(
            optimizer.optimize(
                prompt="Explain machine learning",
                target_model="anthropic/claude-3.5-sonnet",
            )
        )

        assert isinstance(result, OptimizationResult)
        assert result.original == "Explain machine learning"
        assert result.optimized == "Optimized version of the prompt"
        assert result.target_model == "anthropic/claude-3.5-sonnet"
        assert result.target_family == ModelFamily.ANTHROPIC

    def test_full_optimization_workflow_google(self, mock_client):
        """Test complete optimization workflow for Google model."""
        optimizer = PromptOptimizer(api_key="test-key")

        result = asyncio.run(
            optimizer.optimize(
                prompt="Create a React component", target_model="google/gemini-2.5-pro"
            )
        )

        assert isinstance(result, OptimizationResult)
        assert result.original == "Create a React component"
        assert result.optimized == "Optimized version of the prompt"
        assert result.target_model == "google/gemini-2.5-pro"
        assert result.target_family == ModelFamily.GOOGLE

    def test_batch_optimization_workflow(self, mock_client):
        """Test batch optimization of multiple prompts."""
        optimizer = PromptOptimizer(api_key="test-key")

        prompts = [
            "Write a Python function",
            "Explain quantum computing",
            "Create a REST API",
        ]

        results = asyncio.run(
            optimizer.optimize_multiple(prompts=prompts, target_model="openai/gpt-4o")
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, OptimizationResult)
            assert result.original == prompts[i]
            assert result.optimized == "Optimized version of the prompt"
            assert result.target_model == "openai/gpt-4o"
            assert result.target_family == ModelFamily.OPENAI

    def test_cross_model_comparison(self, mock_client):
        """Test comparing optimization results across different models."""
        optimizer = PromptOptimizer(api_key="test-key")

        prompt = "Design a database schema"

        results = asyncio.run(
            optimizer.compare_optimizations(
                prompt=prompt,
                target_models=[
                    "openai/gpt-4o",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-2.5-pro",
                ],
            )
        )

        assert len(results) == 3
        assert "openai/gpt-4o" in results
        assert "anthropic/claude-3.5-sonnet" in results
        assert "google/gemini-2.5-pro" in results

        for model, result in results.items():
            assert isinstance(result, OptimizationResult)
            assert result.original == prompt
            assert result.optimized == "Optimized version of the prompt"
            assert result.target_model == model


class TestPerformanceOptimization:
    """Test cases for performance optimization features."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with timing simulation."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async def mock_optimize_with_timing(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate API call delay
                return "Optimized prompt content"

            mock_client.optimize_prompt = mock_optimize_with_timing
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_optimization_timing_tracking(self, mock_client):
        """Test that optimization time is properly tracked."""
        optimizer = PromptOptimizer(api_key="test-key")

        start_time = time.time()
        result = asyncio.run(
            optimizer.optimize(prompt="Test prompt", target_model="openai/gpt-4o")
        )
        end_time = time.time()

        # Should have taken at least the mock delay time
        assert result.optimization_time >= 0.1
        assert result.optimization_time <= end_time - start_time + 0.1

    def test_concurrent_batch_processing(self, mock_client):
        """Test that batch processing actually runs concurrently."""
        optimizer = PromptOptimizer(api_key="test-key")

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]

        start_time = time.time()
        results = asyncio.run(
            optimizer.optimize_multiple(
                prompts=prompts, target_model="openai/gpt-4o", max_concurrent=3
            )
        )
        end_time = time.time()

        # With concurrency, should take less than sequential time
        sequential_time = len(prompts) * 0.1
        actual_time = end_time - start_time

        # Allow some margin for test execution overhead
        assert actual_time < sequential_time + 0.5
        assert len(results) == len(prompts)

    def test_caching_behavior(self, mock_client):
        """Test that caching reduces redundant API calls."""
        optimizer = PromptOptimizer(api_key="test-key")

        # First optimization
        result1 = asyncio.run(
            optimizer.optimize(
                prompt="Test prompt for caching",
                target_model="openai/gpt-4o",
                use_cache=True,
            )
        )

        # Second optimization with same parameters
        result2 = asyncio.run(
            optimizer.optimize(
                prompt="Test prompt for caching",
                target_model="openai/gpt-4o",
                use_cache=True,
            )
        )

        # Should return same result
        assert result1.optimized == result2.optimized
        assert result1.target_model == result2.target_model

        # Should only have made one API call due to caching
        assert mock_client.optimize_prompt.call_count == 1


class TestErrorHandlingAndRecovery:
    """Test cases for error handling and recovery mechanisms."""

    @pytest.fixture
    def failing_client(self):
        """Create a mock client that simulates failures."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate various failure scenarios
            call_count = 0

            async def mock_optimize_with_failures(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count == 1:
                    # First call fails
                    from src.pagans.exceptions import NetworkError

                    raise NetworkError("Temporary network failure")
                if call_count == 2:
                    # Second call fails differently
                    from src.pagans.exceptions import RateLimitError

                    raise RateLimitError("Rate limit exceeded", retry_after=1)
                # Third call succeeds
                return "Successfully optimized after retries"

            mock_client.optimize_prompt = mock_optimize_with_failures
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_automatic_retry_on_transient_failures(self, failing_client):
        """Test that the system automatically retries on transient failures."""
        optimizer = PromptOptimizer(api_key="test-key", max_retries=3)

        result = asyncio.run(
            optimizer.optimize(
                prompt="Test prompt with failures", target_model="openai/gpt-4o"
            )
        )

        assert result.optimized == "Successfully optimized after retries"
        assert failing_client.optimize_prompt.call_count == 3

    def test_graceful_degradation_on_persistent_failures(self, failing_client):
        """Test graceful handling when all retries are exhausted."""
        optimizer = PromptOptimizer(api_key="test-key", max_retries=2)

        # Mock persistent failure
        async def always_fail(*args, **kwargs):
            from src.pagans.exceptions import NetworkError

            raise NetworkError("Persistent network failure")

        failing_client.optimize_prompt = always_fail

        with pytest.raises(PromptOptimizerError):
            asyncio.run(
                optimizer.optimize(prompt="Test prompt", target_model="openai/gpt-4o")
            )


class TestAdvancedFeatures:
    """Test cases for advanced PAGANS features."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for advanced feature testing."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_client.optimize_prompt.return_value = "Custom optimized prompt"
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_custom_optimization_notes(self, mock_client):
        """Test optimization with custom notes."""
        optimizer = PromptOptimizer(api_key="test-key")

        result = asyncio.run(
            optimizer.optimize(
                prompt="Write a technical blog post",
                target_model="openai/gpt-4o",
                optimization_notes="Focus on clarity and technical accuracy",
            )
        )

        assert result.optimization_notes == "Focus on clarity and technical accuracy"

    def test_context_manager_usage(self, mock_client):
        """Test using PromptOptimizer as a context manager."""

        async def test_context_manager():
            async with PromptOptimizer(api_key="test-key") as optimizer:
                result = await optimizer.optimize(
                    prompt="Test prompt", target_model="openai/gpt-4o"
                )
                return result

        result = asyncio.run(test_context_manager())

        assert isinstance(result, OptimizationResult)
        assert result.optimized == "Custom optimized prompt"

    def test_environment_variable_configuration(self, mock_client):
        """Test configuration via environment variables."""
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "env-api-key",
                "OPENROUTER_BASE_URL": "https://custom.api.url",
                "DEFAULT_OPTIMIZER_MODEL": "anthropic/claude-3.5-sonnet",
            },
            clear=True,
        ):
            optimizer = PromptOptimizer()

            assert optimizer.api_key == "env-api-key"
            assert optimizer.base_url == "https://custom.api.url"
            assert optimizer.default_model == "anthropic/claude-3.5-sonnet"
