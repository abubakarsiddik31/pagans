"""
Prompt Optimizer - A Python package for optimizing prompts across different LLM model families.

This package provides tools to optimize prompts for various LLM architectures using
family-specific optimization techniques and the OpenRouter API.
"""

__version__ = "0.1.0"
__author__ = "Abu Bakar Siddik"
__email__ = "abubakar1808031@gmail.com"

from .core import PromptOptimizer
from .models import ModelFamily, OptimizationResult, OptimizationRequest
from .exceptions import PromptOptimizerError, OpenRouterAPIError, ModelNotFoundError

__all__ = [
    "PromptOptimizer",
    "ModelFamily",
    "OptimizationResult",
    "OptimizationRequest",
    "PromptOptimizerError",
    "OpenRouterAPIError",
    "ModelNotFoundError",
]
