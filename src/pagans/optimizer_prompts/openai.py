"""
OpenAI family optimization prompts.

This module contains detailed, research-specific optimization prompts
for OpenAI's GPT models based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt


class OpenAIOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for OpenAI GPT models."""

    def __init__(self):
        super().__init__("openai")

    def get_model_family(self) -> str:
        """Get the model family name."""
        return "OpenAI GPT"

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the OpenAI-specific optimization prompt.

        Based on OpenAI's prompt engineering best practices and documentation.
        """
        return super().get_prompt(original_prompt, target_model)

    def get_description(self) -> str:
        """Get description of OpenAI optimization approach."""
        return """
        Optimization strategy for OpenAI GPT text models focusing on:
        - explicit goals, constraints, and output contract
        - structured instructions for multi-step tasks
        - precise format guidance (schema/sections/style)
        - examples when output consistency is required

        Based on OpenAI's latest model and reasoning prompt guidance.
        """


# Global instance for easy access
OPENAI_OPTIMIZATION_PROMPT = OpenAIOptimizationPrompt()
