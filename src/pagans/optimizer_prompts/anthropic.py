"""
Anthropic family optimization prompts.

This module contains detailed, research-specific optimization prompts
for Anthropic's Claude models based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt


class AnthropicOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for Anthropic Claude models."""

    def __init__(self):
        super().__init__("anthropic")

    def get_model_family(self) -> str:
        """Get the model family name."""
        return "Anthropic Claude"

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the Anthropic-specific optimization prompt.

        Based on Anthropic's prompt engineering best practices and documentation.
        """
        return super().get_prompt(original_prompt, target_model)

    def get_description(self) -> str:
        """Get description of Anthropic optimization approach."""
        return """
        Optimization strategy for Anthropic Claude models focusing on:
        - Conversational, natural language that feels like a helpful assistant
        - Clear context and background information
        - Step-by-step reasoning for complex tasks
        - Safety-conscious framing and appropriate guardrails
        - Well-structured responses with clear organization
        - Examples that demonstrate the desired behavior

        Based on Anthropic's official documentation and community best practices.
        """


# Global instance for easy access
ANTHROPIC_OPTIMIZATION_PROMPT = AnthropicOptimizationPrompt()
