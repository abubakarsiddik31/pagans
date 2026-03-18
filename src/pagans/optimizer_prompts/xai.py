"""
xAI family optimization prompts.

This module contains prompt optimization guidance for xAI Grok text models.
"""

from .base import BaseOptimizationPrompt


class XAIOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for xAI Grok models."""

    def __init__(self):
        super().__init__("xai")

    def get_model_family(self) -> str:
        """Get the model family name."""
        return "xAI Grok"

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """Get the xAI-specific optimization prompt."""
        return super().get_prompt(original_prompt, target_model)

    def get_description(self) -> str:
        """Get description of xAI optimization approach."""
        return """
        Optimization strategy for xAI Grok text models focusing on:
        - Strong context grounding with relevant project or task constraints
        - Explicit, concrete goals and completion criteria
        - Detailed system instructions for behavior, edge cases, and quality bar
        - Iterative refinement guidance when first-pass output is insufficient
        - Structured context blocks (XML or Markdown sections) for clarity

        Based on xAI developer guidance for Grok model prompting and migration.
        """


# Global instance for easy access
XAI_OPTIMIZATION_PROMPT = XAIOptimizationPrompt()
