"""
Performance tests for PAGANS.

This module contains performance benchmarks and stress tests to ensure
PAGANS can handle various load scenarios efficiently.
"""

import asyncio
import statistics
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.pagans.core import PromptOptimizer
from src.pagans.models import OptimizationResult


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.fixture
    def fast_mock_client(self):
        """Create a mock client with fast response times."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async def fast_optimize(*args, **kwargs):
                await asyncio.sleep(0.01)  # Very fast response
                return "Optimized prompt"

            mock_client.optimize_prompt = fast_optimize
            mock_client.validate_model.return_value = True

            yield mock_client

    @pytest.fixture
    def slow_mock_client(self):
        """Create a mock client with realistic response times."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            async def slow_optimize(*args, **kwargs):
                await asyncio.sleep(0.5)  # Realistic API response time
                return "Optimized prompt"

            mock_client.optimize_prompt = slow_optimize
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_single_optimization_performance(self, fast_mock_client, benchmark):
        """Benchmark single prompt optimization performance."""
        optimizer = PromptOptimizer(api_key="test-key")

        def optimize_single():
            result = asyncio.run(
                optimizer.optimize(
                    prompt="Write a Python function to calculate fibonacci numbers",
                    target_model="openai/gpt-4o",
                )
            )
            return result

        result = benchmark(optimize_single)

        assert isinstance(result, OptimizationResult)
        assert result.optimization_time < 1.0  # Should complete quickly

    def test_batch_optimization_performance(self, fast_mock_client, benchmark):
        """Benchmark batch optimization performance."""
        optimizer = PromptOptimizer(api_key="test-key")

        prompts = [
            "Write a Python function",
            "Explain machine learning",
            "Create a React component",
            "Design a database schema",
            "Write a REST API",
        ]

        def optimize_batch():
            results = asyncio.run(
                optimizer.optimize_multiple(
                    prompts=prompts, target_model="openai/gpt-4o", max_concurrent=3
                )
            )
            return results

        results = benchmark(optimize_batch)

        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, OptimizationResult)

    def test_concurrent_load_handling(self, fast_mock_client):
        """Test handling of concurrent optimization requests."""
        optimizer = PromptOptimizer(api_key="test-key")

        async def concurrent_optimizations():
            tasks = []
            for i in range(10):
                task = optimizer.optimize(
                    prompt=f"Test prompt {i}", target_model="openai/gpt-4o"
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            return results, end_time - start_time

        results, total_time = asyncio.run(concurrent_optimizations())

        assert len(results) == 10
        # Should complete faster than sequential execution
        assert total_time < 2.0  # Much faster than 10 * 0.01 = 0.1s

    def test_memory_efficiency_large_prompts(self, fast_mock_client):
        """Test memory efficiency with large prompts."""
        optimizer = PromptOptimizer(api_key="test-key")

        # Create a large prompt (approximately 10KB)
        large_prompt = (
            "Write a comprehensive guide about " + "Python programming " * 500
        )

        result = asyncio.run(
            optimizer.optimize(prompt=large_prompt, target_model="openai/gpt-4o")
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.original) > 5000  # Ensure it was actually large
        assert result.optimization_time < 2.0  # Should still be reasonably fast

    def test_scalability_with_increasing_load(self, fast_mock_client):
        """Test scalability as load increases."""
        optimizer = PromptOptimizer(api_key="test-key")

        # Test with different batch sizes
        batch_sizes = [1, 5, 10, 20]
        performance_results = {}

        for batch_size in batch_sizes:
            prompts = [f"Test prompt {i}" for i in range(batch_size)]

            start_time = time.time()
            results = asyncio.run(
                optimizer.optimize_multiple(
                    prompts=prompts,
                    target_model="openai/gpt-4o",
                    max_concurrent=min(batch_size, 5),  # Cap concurrency
                )
            )
            end_time = time.time()

            total_time = end_time - start_time
            avg_time_per_prompt = total_time / batch_size

            performance_results[batch_size] = {
                "total_time": total_time,
                "avg_time_per_prompt": avg_time_per_prompt,
                "results_count": len(results),
            }

        # Verify that larger batches don't have disproportionately worse performance
        for batch_size in batch_sizes[1:]:
            prev_batch = batch_sizes[batch_sizes.index(batch_size) - 1]
            scaling_factor = batch_size / prev_batch

            current_avg = performance_results[batch_size]["avg_time_per_prompt"]
            prev_avg = performance_results[prev_batch]["avg_time_per_prompt"]

            # Allow some degradation but not more than linear scaling
            assert current_avg <= prev_avg * scaling_factor * 1.5

    def test_caching_performance_impact(self, fast_mock_client):
        """Test performance impact of caching."""
        optimizer = PromptOptimizer(api_key="test-key")

        prompt = "Write a Python function to reverse a string"
        target_model = "openai/gpt-4o"

        # First optimization (no cache)
        start_time = time.time()
        result1 = asyncio.run(
            optimizer.optimize(prompt=prompt, target_model=target_model, use_cache=True)
        )
        first_duration = time.time() - start_time

        # Second optimization (should use cache)
        start_time = time.time()
        result2 = asyncio.run(
            optimizer.optimize(prompt=prompt, target_model=target_model, use_cache=True)
        )
        second_duration = time.time() - start_time

        # Cached result should be significantly faster
        assert second_duration < first_duration * 0.5
        assert result1.optimized == result2.optimized


class TestStressTests:
    """Stress tests for extreme scenarios."""

    @pytest.fixture
    def resilient_mock_client(self):
        """Create a mock client that can handle high load."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            call_count = 0

            async def stress_optimize(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                # Simulate occasional delays to test resilience
                if call_count % 10 == 0:
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.01)

                return f"Optimized prompt {call_count}"

            mock_client.optimize_prompt = stress_optimize
            mock_client.validate_model.return_value = True

            yield mock_client

    def test_high_concurrency_stress(self, resilient_mock_client):
        """Test handling of very high concurrent requests."""
        optimizer = PromptOptimizer(api_key="test-key")

        async def stress_test():
            # Simulate 50 concurrent requests
            tasks = []
            for i in range(50):
                task = optimizer.optimize(
                    prompt=f"Stress test prompt {i}", target_model="openai/gpt-4o"
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            return results, end_time - start_time

        results, total_time = asyncio.run(stress_test())

        assert len(results) == 50
        assert total_time < 5.0  # Should complete within reasonable time

        # Verify all results are valid
        for result in results:
            assert isinstance(result, OptimizationResult)
            assert "Optimized prompt" in result.optimized

    def test_memory_usage_large_batch(self, resilient_mock_client):
        """Test memory usage with very large batches."""
        optimizer = PromptOptimizer(api_key="test-key")

        # Create a very large batch
        large_batch = [
            f"Large batch prompt {i} with some additional content to increase size "
            * 10
            for i in range(100)
        ]

        results = asyncio.run(
            optimizer.optimize_multiple(
                prompts=large_batch,
                target_model="openai/gpt-4o",
                max_concurrent=10,  # Limit concurrency for memory management
            )
        )

        assert len(results) == len(large_batch)

        # Verify results integrity
        for i, result in enumerate(results):
            assert isinstance(result, OptimizationResult)
            assert "Optimized prompt" in result.optimized

    def test_long_running_stability(self, resilient_mock_client):
        """Test stability during long-running operations."""
        optimizer = PromptOptimizer(api_key="test-key")

        async def long_running_test():
            results = []
            start_time = time.time()

            # Run optimizations for approximately 30 seconds
            while time.time() - start_time < 30:
                batch_tasks = []
                for i in range(5):  # 5 concurrent requests per iteration
                    task = optimizer.optimize(
                        prompt=f"Long running test prompt {len(results) + i}",
                        target_model="openai/gpt-4o",
                    )
                    batch_tasks.append(task)

                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            end_time = time.time()
            return results, end_time - start_time

        results, duration = asyncio.run(long_running_test())

        # Should have completed many requests
        assert len(results) > 100  # At least 100 requests completed
        assert 25 < duration < 35  # Should be close to 30 seconds

        # Verify result quality
        success_count = sum(1 for r in results if isinstance(r, OptimizationResult))
        assert success_count == len(results)


class TestResourceManagement:
    """Test resource management and cleanup."""

    @pytest.fixture
    def mock_client_with_cleanup(self):
        """Create a mock client that tracks cleanup."""
        with patch("src.pagans.client.OpenRouterClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            cleanup_called = False

            async def mock_cleanup():
                nonlocal cleanup_called
                cleanup_called = True

            mock_client.aclose = mock_cleanup
            mock_client.optimize_prompt.return_value = "Optimized"
            mock_client.validate_model.return_value = True
            mock_client.cleanup_called = lambda: cleanup_called

            yield mock_client

    def test_proper_resource_cleanup(self, mock_client_with_cleanup):
        """Test that resources are properly cleaned up."""

        async def test_cleanup():
            optimizer = PromptOptimizer(api_key="test-key")

            # Use the optimizer
            result = await optimizer.optimize(
                prompt="Test prompt", target_model="openai/gpt-4o"
            )

            # Explicit cleanup
            await optimizer.close()

            return result

        result = asyncio.run(test_cleanup())

        assert isinstance(result, OptimizationResult)
        # Verify cleanup was called
        assert mock_client_with_cleanup.cleanup_called()

    def test_context_manager_cleanup(self, mock_client_with_cleanup):
        """Test cleanup when using context manager."""

        async def test_context_cleanup():
            async with PromptOptimizer(api_key="test-key") as optimizer:
                result = await optimizer.optimize(
                    prompt="Test prompt", target_model="openai/gpt-4o"
                )
                return result

        result = asyncio.run(test_context_cleanup())

        assert isinstance(result, OptimizationResult)
        # Cleanup should be automatic with context manager
        assert mock_client_with_cleanup.cleanup_called()

    def test_connection_pooling_efficiency(self, fast_mock_client):
        """Test efficiency of connection pooling."""
        optimizer = PromptOptimizer(api_key="test-key")

        # Run multiple sequential requests
        times = []
        for i in range(20):
            start = time.time()
            result = asyncio.run(
                optimizer.optimize(
                    prompt=f"Connection test {i}", target_model="openai/gpt-4o"
                )
            )
            end = time.time()
            times.append(end - start)

        # Calculate statistics
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)

        # Should have consistent performance (low standard deviation)
        assert std_dev / avg_time < 0.5  # Less than 50% variation
        assert avg_time < 0.1  # Should be fast
