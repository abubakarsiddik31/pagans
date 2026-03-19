"""PAGANS package."""

from .clients import (
    AnthropicClient,
    GoogleAIStudioClient,
    OpenAIClient,
    OpenRouterClient,
    ZAIClient,
)
from .core import PAGANSOptimizer
from .exceptions import (
    PAGANSAuthenticationError,
    PAGANSConfigurationError,
    PAGANSError,
    PAGANSModelNotFoundError,
    PAGANSNetworkError,
    PAGANSOpenRouterAPIError,
    PAGANSQuotaExceededError,
    PAGANSRateLimitError,
    PAGANSTimeoutError,
    PAGANSValidationError,
)
from .models import (
    ModelFamily,
    OptimizationRequest,
    OptimizationResult,
    Provider,
    detect_model_family,
    get_supported_models,
    is_supported_model,
)
from .models.registry import (
    get_model_registry,
    register_model_family,
    register_provider,
)
from .optimizer_prompts import (
    get_optimization_description,
    get_optimization_prompt,
    get_supported_families,
)

__version__ = "0.1.0"
__author__ = "Abu Bakar Siddik"
__email__ = "abubakar1808031@gmail.com"
__description__ = (
    "Prompts Aligned to Guidelines and Normalization System for LLM families "
    "across OpenRouter, OpenAI, Google AI Studio, Anthropic, and Z.ai"
)

# Main exports
__all__ = [
    # Main classes
    "PAGANSOptimizer",
    "OpenRouterClient",
    "OpenAIClient",
    "GoogleAIStudioClient",
    "AnthropicClient",
    "ZAIClient",
    # Models and data structures
    "Provider",
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
    provider: Provider = Provider.OPENROUTER,
) -> PAGANSOptimizer:
    """
    Create a PAGANS PAGANSOptimizer instance with the given configuration.

    Args:
        api_key: Provider API key (if None, tries to get from provider env var)
        base_url: Provider base URL (if None, tries to get from provider env var)
        default_model: Default model for optimization (if None, uses default)
        provider: Optimizer provider to use

    Returns:
        PAGANSOptimizer instance

    Raises:
        PAGANSConfigurationError: If API key is not provided or found
    """
    return PAGANSOptimizer(
        api_key=api_key,
        base_url=base_url,
        provider=provider,
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
