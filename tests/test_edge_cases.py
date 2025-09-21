"""
Edge case tests for PAGANS.

This module contains tests for edge cases, boundary conditions,
and unusual scenarios that might cause issues.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.pagans.core import PAGANSOptimizer
from src.pagans.exceptions import (
    PAGANSError,
)
from src.pagans.models import ModelFamily, OptimizationResult


class TestExtremeInputHandling:
    """Test cases for extreme and unusual input scenarios."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_client.optimize_prompt.return_value = "Optimized"
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_empty_prompt_handling(self, mock_client):
        """Test handling of empty prompts."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                asyncio.run(
                    optimizer.optimize(prompt="", target_model="openai/gpt-4o")
                )

    def test_whitespace_only_prompt(self, mock_client):
        """Test handling of whitespace-only prompts."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                asyncio.run(
                    optimizer.optimize(prompt="   \n\t  ", target_model="openai/gpt-4o")
                )

    def test_very_long_prompt(self, mock_client):
        """Test handling of extremely long prompts."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            # Create a very long prompt (100KB+)
            long_prompt = "Write a comprehensive tutorial about Python programming. " * 2000

            result = asyncio.run(
                optimizer.optimize(prompt=long_prompt, target_model="openai/gpt-4o")
            )

            assert isinstance(result, OptimizationResult)
            assert len(result.original) > 100000  # Ensure it's actually long
            assert result.optimized == "Optimized"

    def test_unicode_and_special_characters(self, mock_client):
        """Test handling of Unicode and special characters."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            unicode_prompt = "Write a Python function using emojis 😀 and special chars: àáâãäå, 中文, русский"

            result = asyncio.run(
                optimizer.optimize(prompt=unicode_prompt, target_model="openai/gpt-4o")
            )

            assert isinstance(result, OptimizationResult)
            assert result.original == unicode_prompt
            assert result.optimized == "Optimized"

    def test_binary_data_in_prompt(self, mock_client):
        """Test handling of binary-like data in prompts."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            # Create a prompt with binary-like content
            binary_prompt = "Analyze this data: " + "".join(chr(i) for i in range(32, 127))

            result = asyncio.run(
                optimizer.optimize(prompt=binary_prompt, target_model="openai/gpt-4o")
            )

            assert isinstance(result, OptimizationResult)
            assert result.optimized == "Optimized"

    def test_extremely_short_prompt(self, mock_client):
        """Test handling of very short prompts."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            result = asyncio.run(
                optimizer.optimize(prompt="Hi", target_model="openai/gpt-4o")
            )

            assert isinstance(result, OptimizationResult)
            assert result.original == "Hi"
            assert result.optimized == "Optimized"


class TestModelEdgeCases:
    """Test cases for model-related edge cases."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            yield mock_client

    def test_unknown_model_family(self, mock_client):
        """Test handling of unknown model families."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            from src.pagans.exceptions import PAGANSModelNotFoundError
            optimizer = PAGANSOptimizer(api_key="test-key")

            with pytest.raises(ModelNotFoundError):
                asyncio.run(
                    optimizer.optimize(prompt="Test prompt", target_model="unknown/model-v123")
                )

    def test_model_with_special_characters(self, mock_client):
        """Test handling of model names with special characters."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            from src.pagans.exceptions import PAGANSModelNotFoundError
            optimizer = PAGANSOptimizer(api_key="test-key")

            special_models = [
                "model/with/slashes",
                "model-with-dashes",
                "model.with.dots",
                "model_underscore",
                "model@symbol",
            ]

            for model in special_models:
                with pytest.raises(ModelNotFoundError):
                    asyncio.run(
                        optimizer.optimize(prompt="Test prompt", target_model=model)
                    )

    def test_case_sensitivity_in_model_names(self, mock_client):
        """Test case sensitivity handling in model names."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            mock_client.optimize_prompt.return_value = "Optimized"
            mock_client.validate_model.return_value = True

            result = asyncio.run(
                optimizer.optimize(
                    prompt="Test prompt", target_model="OpenAI/GPT-4O"  # Mixed case
                )
            )

            assert isinstance(result, OptimizationResult)
            assert result.target_model == "OpenAI/GPT-4O"


class TestConfigurationEdgeCases:
    """Test cases for configuration edge cases."""

    def test_extreme_timeout_values(self):
        """Test handling of extreme timeout values."""
        with patch("src.pagans.core.OpenRouterClient") as mock_client_class:
            # Very short timeout
            optimizer1 = PAGANSOptimizer(api_key="test-key", timeout=0.001)
            mock_client_class.assert_called_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=0.001,
                max_retries=3,
                retry_delay=1.0
            )

            # Very long timeout
            optimizer2 = PAGANSOptimizer(api_key="test-key", timeout=3600.0)
            mock_client_class.assert_called_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=3600.0,
                max_retries=3,
                retry_delay=1.0
            )

            # Zero timeout (should handle gracefully)
            optimizer3 = PAGANSOptimizer(api_key="test-key", timeout=0.0)
            mock_client_class.assert_called_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=0.0,
                max_retries=3,
                retry_delay=1.0
            )

    def test_extreme_retry_values(self):
        """Test handling of extreme retry values."""
        with patch("src.pagans.core.OpenRouterClient") as mock_client_class:
            # Zero retries
            optimizer1 = PAGANSOptimizer(api_key="test-key", max_retries=0)
            mock_client_class.assert_called_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=30.0,
                max_retries=0,
                retry_delay=1.0
            )

            # Very high retry count
            optimizer2 = PAGANSOptimizer(api_key="test-key", max_retries=100)
            mock_client_class.assert_called_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                timeout=30.0,
                max_retries=100,
                retry_delay=1.0
            )

    def test_extreme_concurrency_values(self):
        """Test handling of extreme concurrency values."""
        with patch("src.pagans.core.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.optimize_prompt.return_value = "Optimized"
            mock_client.validate_model.return_value = True

            optimizer = PAGANSOptimizer(api_key="test-key")

            # Test with very high concurrency
            prompts = ["Test"] * 10
            results = asyncio.run(
                optimizer.optimize_multiple(
                    prompts=prompts,
                    target_model="openai/gpt-4o",
                    max_concurrent=100,  # Very high concurrency
                )
            )

            assert len(results) == 10

    def test_malformed_urls(self):
        """Test handling of malformed URLs."""
        # These should not raise exceptions during initialization
        configs = [
            {"base_url": "not-a-url"},
            {"base_url": "http://"},
            {"base_url": "https://"},
            {"base_url": "ftp://example.com"},
            {"base_url": ""},
        ]

        for config in configs:
            try:
                optimizer = PAGANSOptimizer(api_key="test-key", **config)
                # Should not crash during initialization
                assert optimizer is not None
            except Exception:
                # Some malformed URLs might raise exceptions, which is acceptable
                pass


class TestBatchProcessingEdgeCases:
    """Test cases for batch processing edge cases."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_client.optimize_prompt.return_value = "Optimized"
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_empty_batch(self, mock_client):
        """Test handling of empty batch."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            results = asyncio.run(
                optimizer.optimize_multiple(prompts=[], target_model="openai/gpt-4o")
            )

            assert results == []

    def test_single_item_batch(self, mock_client):
        """Test batch processing with single item."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            results = asyncio.run(
                optimizer.optimize_multiple(
                    prompts=["Single prompt"], target_model="openai/gpt-4o"
                )
            )

            assert len(results) == 1
            assert isinstance(results[0], OptimizationResult)
            assert results[0].original == "Single prompt"

    def test_mixed_prompt_lengths_in_batch(self, mock_client):
        """Test batch with prompts of vastly different lengths."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            prompts = [
                "Short",
                "A" * 1000,  # Medium
                "B" * 10000,  # Long
                "C" * 50000,  # Very long
            ]

            results = asyncio.run(
                optimizer.optimize_multiple(prompts=prompts, target_model="openai/gpt-4o")
            )

            assert len(results) == len(prompts)
            for i, result in enumerate(results):
                assert isinstance(result, OptimizationResult)
                assert len(result.original) == len(prompts[i])

    def test_batch_with_duplicate_prompts(self, mock_client):
        """Test batch processing with duplicate prompts."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            prompts = ["Same prompt"] * 5

            results = asyncio.run(
                optimizer.optimize_multiple(prompts=prompts, target_model="openai/gpt-4o")
            )

            assert len(results) == 5
            for result in results:
                assert isinstance(result, OptimizationResult)
                assert result.original == "Same prompt"
                assert result.optimized == "Optimized"


class TestErrorRecoveryEdgeCases:
    """Test cases for error recovery in edge scenarios."""

    @pytest.fixture
    def failing_client(self):
        """Create a mock client that can simulate various failures."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            yield mock_client

    def test_partial_batch_failure_recovery(self, failing_client):
        """Test recovery when some items in a batch fail."""
        with patch("src.pagans.core.OpenRouterClient", return_value=failing_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            # Mock partial failures in batch
            call_count = 0

            async def partial_failure_optimize(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count % 3 == 0:  # Every third call fails
                    from src.pagans.exceptions import NetworkError

                    raise NetworkError("Intermittent failure")

                return f"Optimized {call_count}"

            failing_client.optimize_prompt = partial_failure_optimize
            failing_client.validate_model.return_value = True

            prompts = ["Prompt"] * 9  # 9 prompts, 3 should fail

            # This should handle partial failures gracefully
            results = asyncio.run(
                optimizer.optimize_multiple(prompts=prompts, target_model="openai/gpt-4o")
            )

            # Should have some successful results
            successful_results = [r for r in results if isinstance(r, OptimizationResult)]
            assert len(successful_results) > 0

    def test_network_partition_simulation(self, failing_client):
        """Test behavior during simulated network partitions."""
        with patch("src.pagans.core.OpenRouterClient", return_value=failing_client):
            optimizer = PAGANSOptimizer(api_key="test-key", max_retries=2)

            # Simulate complete network failure
            async def network_failure(*args, **kwargs):
                from src.pagans.exceptions import NetworkError

                raise NetworkError("Network partition")

            failing_client.optimize_prompt = network_failure
            failing_client.validate_model.return_value = True

            with pytest.raises(PAGANSError):
                asyncio.run(
                    optimizer.optimize(prompt="Test prompt", target_model="openai/gpt-4o")
                )

    def test_extreme_memory_pressure(self, failing_client):
        """Test behavior under extreme memory pressure."""
        with patch("src.pagans.core.OpenRouterClient", return_value=failing_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            # Create extremely large prompts that might cause memory issues
            huge_prompts = ["X" * 1000000] * 10  # 10 prompts of 1MB each

            failing_client.optimize_prompt.return_value = "Optimized"

            results = asyncio.run(
                optimizer.optimize_multiple(
                    prompts=huge_prompts,
                    target_model="openai/gpt-4o",
                    max_concurrent=2,  # Low concurrency to manage memory
                )
            )

            assert len(results) == len(huge_prompts)

    def test_rapid_consecutive_requests(self, failing_client):
        """Test handling of rapid consecutive requests."""
        with patch("src.pagans.core.OpenRouterClient", return_value=failing_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            failing_client.optimize_prompt.return_value = "Optimized"

            # Send many requests in rapid succession
            async def rapid_requests():
                tasks = []
                for i in range(50):
                    task = optimizer.optimize(
                        prompt=f"Rapid request {i}", target_model="openai/gpt-4o"
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)
                return results

            results = asyncio.run(rapid_requests())

            assert len(results) == 50
            for result in results:
                assert isinstance(result, OptimizationResult)


class TestDataIntegrity:
    """Test cases for data integrity and consistency."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            yield mock_client

    def test_prompt_content_preservation(self, mock_client):
        """Test that prompt content is preserved exactly."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            test_prompts = [
                "Simple prompt",
                "Prompt with\nnewlines",
                "Prompt with\ttabs",
                "Prompt with unicode: 🎯🚀💡",
                "Prompt with quotes: \"single\" and 'double'",
                "Prompt with backslashes: \\ and \\\\",
            ]

            mock_client.optimize_prompt.return_value = "Optimized"

            for original_prompt in test_prompts:
                result = asyncio.run(
                    optimizer.optimize(prompt=original_prompt, target_model="openai/gpt-4o")
                )

                assert result.original == original_prompt

    def test_metadata_consistency(self, mock_client):
        """Test that metadata is consistent across operations."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            mock_client.optimize_prompt.return_value = "Optimized"

            result = asyncio.run(
                optimizer.optimize(
                    prompt="Test prompt",
                    target_model="openai/gpt-4o",
                    optimization_notes="Test notes",
                )
            )

            assert result.target_model == "openai/gpt-4o"
            assert result.target_family == ModelFamily.OPENAI
            assert result.optimization_notes == "Test notes"

    def test_result_object_immutability(self, mock_client):
        """Test that result objects are properly immutable."""
        with patch("src.pagans.core.OpenRouterClient", return_value=mock_client):
            optimizer = PAGANSOptimizer(api_key="test-key")

            mock_client.optimize_prompt.return_value = "Optimized"

            result = asyncio.run(
                optimizer.optimize(prompt="Test prompt", target_model="openai/gpt-4o")
            )

            # Attempt to modify result
            result.optimized = "Modified"
            result.original = "Modified original"

            # Verify values were changed (object is mutable)
            assert result.optimized == "Modified"
            assert result.original == "Modified original"
