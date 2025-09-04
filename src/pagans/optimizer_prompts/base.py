"""
Base optimization prompt template and utilities.

This module provides the foundation for family-specific optimization prompts.
"""

from abc import ABC, abstractmethod


class BaseOptimizationPrompt(ABC):
    """Abstract base class for optimization prompts."""

    @abstractmethod
    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the optimization prompt for a specific model family.

        Args:
            original_prompt: The original prompt to optimize
            target_model: The target model name

        Returns:
            The optimization prompt template with variables filled
        """

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this optimization approach.

        Returns:
            Description of the optimization strategy
        """


class OptimizationPromptManager:
    """Manager for optimization prompts across different model families."""

    def __init__(self):
        self._prompts: dict[str, BaseOptimizationPrompt] = {}

    def register_prompt(self, family: str, prompt: BaseOptimizationPrompt) -> None:
        """Register an optimization prompt for a model family."""
        self._prompts[family] = prompt

    def get_prompt(self, family: str, original_prompt: str, target_model: str) -> str:
        """Get the optimization prompt for a specific family."""
        if family not in self._prompts:
            raise ValueError(f"No optimization prompt registered for family: {family}")

        return self._prompts[family].get_prompt(original_prompt, target_model)

    def get_description(self, family: str) -> str:
        """Get the description for a specific family."""
        if family not in self._prompts:
            raise ValueError(f"No optimization prompt registered for family: {family}")

        return self._prompts[family].get_description()

    def get_supported_families(self) -> list:
        """Get list of supported model families."""
        return list(self._prompts.keys())


BASE_OPTIMIZATION_TEMPLATE = """
You are an expert at optimizing prompts for {model_family} models.

{family_specific_guidelines}

Take this original prompt and optimize it specifically for {model_family} models:

Original prompt: {original_prompt}
Target model: {target_model}

Return ONLY the optimized prompt, no explanations or meta-commentary.
"""
