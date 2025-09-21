"""
PAGANS - Prompts Aligned to Guidelines and Normalization System 😅

A comprehensive Python package for optimizing prompts across different LLM model families.
PAGANS provides tools to align your prompts with model-specific guidelines and normalization
standards using the OpenRouter API.
"""

from .clients.openrouter import OpenRouterClient
from .core import PAGANSOptimizer
from .exceptions import (
    PAGANSAuthenticationError,
    PAGANSConfigurationError,
    PAGANSModelNotFoundError,
    PAGANSNetworkError,
    PAGANSOpenRouterAPIError,
    PAGANSError,
    PAGANSQuotaExceededError,
    PAGANSRateLimitError,
    PAGANSTimeoutError,
    PAGANSValidationError,
)

from .models import (
    ModelFamily,
    OptimizationRequest,
    OptimizationResult,
    detect_model_family,
    get_supported_models,
    is_supported_model,
)
from .models.registry import get_model_registry, register_model_family, register_provider
from .optimizer_prompts import (
    get_optimization_description,
    get_optimization_prompt,
    get_supported_families,
)

__version__ = "0.1.0"
__author__ = "PAGANS Team"
__email__ = "contact@pagans.dev"
__description__ = (
    "Prompts Aligned to Guidelines and Normalization System - Optimize prompts across LLM model families using OpenRouter"
)

# Main exports
__all__ = [
    # Main classes
    "PAGANSOptimizer",
    "OpenRouterClient",
    # Models and data structures
    "ModelFamily",
    "OptimizationResult",
    "OptimizationRequest",
    # Registry and factory components
    "get_model_registry",
    "register_model_family",
    "register_provider",
    # Utility functions
    "detect_model_family",
    "get_supported_models",
    "is_supported_model",
    "get_optimization_prompt",
    "get_optimization_description",
    "get_supported_families",
    # Exceptions
    "PAGANSError",
    "PAGANSOpenRouterAPIError",
    "PAGANSModelNotFoundError",
    "PAGANSConfigurationError",
    "PAGANSNetworkError",
    "PAGANSTimeoutError",
    "PAGANSRateLimitError",
    "PAGANSValidationError",
    "PAGANSAuthenticationError",
    "PAGANSQuotaExceededError",
]


# Convenience functions for quick start
def create_optimizer(
    api_key: str = None,
    base_url: str = None,
    default_model: str = None,
) -> PAGANSOptimizer:
    """
    Create a PAGANS PAGANSOptimizer instance with the given configuration.

    Args:
        api_key: OpenRouter API key (if None, tries to get from environment)
        base_url: OpenRouter base URL (if None, tries to get from environment)
        default_model: Default model for optimization (if None, uses default)

    Returns:
        PAGANSOptimizer instance

    Raises:
        PAGANSConfigurationError: If API key is not provided or found
    """
    return PAGANSOptimizer(
        api_key=api_key,
        base_url=base_url,
        optimizer_model=default_model,
    )


# Quick start example
def quick_start_example():
    """
    Example usage of PAGANS PAGANSOptimizer.

    This function demonstrates how to use PAGANS to optimize a prompt for better LLM performance.
    """
    import asyncio

    async def example():
        # Create optimizer
        optimizer = create_optimizer()

        try:
            # Optimize a prompt
            result = await optimizer.optimize(
                prompt="Explain quantum computing in simple terms",
                target_model="gpt-4o",
            )

            print(f"Original: {result.original}")
            print(f"Optimized: {result.optimized}")
            print(f"Target Family: {result.target_family}")
            print(f"Optimization Time: {result.optimization_time:.2f}s")

        finally:
            await optimizer.close()

    # Run the example
    asyncio.run(example())


# Check if this is being run as a script
if __name__ == "__main__":
    quick_start_example()
