"""
Optimization prompts for different LLM model families.

This module contains family-specific optimization prompts tailored to each
model architecture based on official documentation and best practices.
"""

from .anthropic import ANTHROPIC_OPTIMIZATION_PROMPT
from .base import BaseOptimizationPrompt, OptimizationPromptManager
from .google import GOOGLE_OPTIMIZATION_PROMPT
from .openai import OPENAI_OPTIMIZATION_PROMPT

_prompt_manager = OptimizationPromptManager()

_prompt_manager.register_prompt("openai", OPENAI_OPTIMIZATION_PROMPT)
_prompt_manager.register_prompt("anthropic", ANTHROPIC_OPTIMIZATION_PROMPT)
_prompt_manager.register_prompt("google", GOOGLE_OPTIMIZATION_PROMPT)


def get_optimization_prompt(
    family: str, original_prompt: str, target_model: str
) -> str:
    """
    Get the optimization prompt for a specific model family.

    Args:
        family: The model family name (e.g., "openai", "anthropic")
        original_prompt: The original prompt to optimize
        target_model: The target model name

    Returns:
        The optimization prompt for the specified family

    Raises:
        ValueError: If the family is not supported
    """
    return _prompt_manager.get_prompt(family, original_prompt, target_model)


def get_optimization_description(family: str) -> str:
    """
    Get the description for a specific model family's optimization approach.

    Args:
        family: The model family name (e.g., "openai", "anthropic")

    Returns:
        The description of the optimization approach

    Raises:
        ValueError: If the family is not supported
    """
    return _prompt_manager.get_description(family)


def get_supported_families() -> list:
    """
    Get list of supported model families.

    Returns:
        List of supported family names
    """
    return _prompt_manager.get_supported_families()


__all__ = [
    "ANTHROPIC_OPTIMIZATION_PROMPT",
    "GOOGLE_OPTIMIZATION_PROMPT",
    "OPENAI_OPTIMIZATION_PROMPT",
    "BaseOptimizationPrompt",
    "OptimizationPromptManager",
    "get_optimization_description",
    "get_optimization_prompt",
    "get_supported_families",
]
