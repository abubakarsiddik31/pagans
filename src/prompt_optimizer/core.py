"""
Core PromptOptimizer class with family detection logic.

This module contains the main PromptOptimizer class that orchestrates
the prompt optimization process by detecting model families and using
appropriate optimization prompts.
"""

import asyncio
import os
import time
from typing import Optional, Dict

from .models import (
    ModelFamily,
    OptimizationResult,
    detect_model_family,
    is_supported_model,
    get_supported_models,
)
from .client import OpenRouterClient
from .optimizer_prompts import (
    get_optimization_prompt,
    get_optimization_description,
)
from .constants import (
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    ENV_OPENROUTER_API_KEY,
    ENV_OPENROUTER_BASE_URL,
    ENV_DEFAULT_OPTIMIZER_MODEL,
)
from .exceptions import (
    PromptOptimizerError,
    ModelNotFoundError,
    ConfigurationError,
    OpenRouterAPIError,
)


class PromptOptimizer:
    """
    Main PromptOptimizer class for optimizing prompts across different LLM model families.

    This class provides the primary interface for prompt optimization, handling
    model family detection, prompt selection, and API integration.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        """
        Initialize the PromptOptimizer.

        Args:
            api_key: OpenRouter API key (if None, tries to get from environment)
            base_url: OpenRouter base URL (if None, tries to get from environment)
            default_model: Default model for optimization (if None, uses default)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
        """
        # Get configuration from environment or use defaults
        self.api_key = api_key or os.getenv(ENV_OPENROUTER_API_KEY)
        if not self.api_key:
            raise ConfigurationError(
                f"API key is required. Set {ENV_OPENROUTER_API_KEY} environment variable or pass api_key parameter."
            )

        self.base_url = base_url or os.getenv(ENV_OPENROUTER_BASE_URL, DEFAULT_BASE_URL)
        self.default_model = default_model or os.getenv(
            ENV_DEFAULT_OPTIMIZER_MODEL, DEFAULT_OPTIMIZER_MODEL
        )

        # Initialize API client
        self.client = OpenRouterClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Cache for optimization results
        self._optimization_cache: Dict[str, OptimizationResult] = {}

    async def optimize(
        self,
        prompt: str,
        target_model: Optional[str] = None,
        optimization_notes: Optional[str] = None,
        use_cache: bool = True,
    ) -> OptimizationResult:
        """
        Optimize a prompt for a specific target model.

        Args:
            prompt: The original prompt to optimize
            target_model: The target model name (if None, uses default)
            optimization_notes: Additional notes for optimization
            use_cache: Whether to use cached results

        Returns:
            OptimizationResult containing the optimized prompt and metadata

        Raises:
            Various exceptions based on error type
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Use default model if none specified
        target_model = target_model or self.default_model

        # Validate target model
        if not is_supported_model(target_model):
            raise ModelNotFoundError(target_model)

        # Create cache key
        cache_key = f"{prompt}:{target_model}:{optimization_notes or ''}"

        # Check cache
        if use_cache and cache_key in self._optimization_cache:
            return self._optimization_cache[cache_key]

        # Detect model family
        try:
            family = detect_model_family(target_model)
        except ValueError as e:
            raise ModelNotFoundError(str(e))

        # Get optimization prompt
        try:
            system_prompt = get_optimization_prompt(
                family.value,
                prompt,
                target_model,
            )
        except ValueError as e:
            raise PromptOptimizerError(f"Failed to get optimization prompt: {str(e)}")

        # Track optimization time
        start_time = time.time()

        try:
            # Call OpenRouter API
            optimized_prompt = await self.client.optimize_prompt(
                prompt=prompt,
                model=target_model,
                system_prompt=system_prompt,
            )

            # Calculate optimization time
            optimization_time = time.time() - start_time

            # Create result
            result = OptimizationResult(
                original=prompt,
                optimized=optimized_prompt,
                target_model=target_model,
                target_family=family,
                optimization_notes=optimization_notes,
                optimization_time=optimization_time,
            )

            # Cache result
            if use_cache:
                self._optimization_cache[cache_key] = result

            return result

        except OpenRouterAPIError as e:
            raise PromptOptimizerError(f"Failed to optimize prompt: {str(e)}")

    async def optimize_multiple(
        self,
        prompts: list[str],
        target_model: Optional[str] = None,
        optimization_notes: Optional[str] = None,
        use_cache: bool = True,
        max_concurrent: int = 3,
    ) -> list[OptimizationResult]:
        """
        Optimize multiple prompts concurrently.

        Args:
            prompts: List of prompts to optimize
            target_model: The target model name (if None, uses default)
            optimization_notes: Additional notes for optimization
            use_cache: Whether to use cached results
            max_concurrent: Maximum number of concurrent optimizations

        Returns:
            List of OptimizationResult objects
        """
        if not prompts:
            return []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def optimize_single(prompt: str) -> OptimizationResult:
            async with semaphore:
                return await self.optimize(
                    prompt=prompt,
                    target_model=target_model,
                    optimization_notes=optimization_notes,
                    use_cache=use_cache,
                )

        # Run optimizations concurrently
        tasks = [optimize_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def compare_optimizations(
        self,
        prompt: str,
        target_models: list[str],
        optimization_notes: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, OptimizationResult]:
        """
        Compare optimizations across different target models.

        Args:
            prompt: The original prompt to optimize
            target_models: List of target model names
            optimization_notes: Additional notes for optimization
            use_cache: Whether to use cached results

        Returns:
            Dictionary mapping model names to OptimizationResult objects
        """
        if not target_models:
            return {}

        # Validate all target models
        for model in target_models:
            if not is_supported_model(model):
                raise ModelNotFoundError(model)

        # Optimize for each model
        results = {}
        for model in target_models:
            try:
                result = await self.optimize(
                    prompt=prompt,
                    target_model=model,
                    optimization_notes=optimization_notes,
                    use_cache=use_cache,
                )
                results[model] = result
            except Exception as e:
                # Log error but continue with other models
                results[model] = PromptOptimizerError(
                    f"Failed to optimize for {model}: {str(e)}"
                )

        return results

    def get_supported_models(self) -> Dict[ModelFamily, list[str]]:
        """Get all supported models organized by family."""
        return get_supported_models()

    def get_optimization_description(self, model_name: str) -> str:
        """Get the optimization description for a specific model family."""
        try:
            family = detect_model_family(model_name)
            return get_optimization_description(family.value)
        except ValueError as e:
            raise ModelNotFoundError(str(e))

    def is_model_supported(self, model_name: str) -> bool:
        """Check if a model is supported."""
        return is_supported_model(model_name)

    async def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model is available via the OpenRouter API.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model is available, False otherwise
        """
        return await self.client.validate_model(model_name)

    async def close(self) -> None:
        """Close the API client and clean up resources."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def clear_cache(self) -> None:
        """Clear the optimization cache."""
        self._optimization_cache.clear()

    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self._optimization_cache)
