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
        Optimization strategy for OpenAI GPT models focusing on:
        - Clear, direct instructions with specific action verbs
        - Step-by-step breakdowns for complex tasks
        - Well-structured formatting with headers, bullet points, and code blocks
        - Context setting at the beginning of the prompt
        - Examples when helpful for demonstrating expected output
        - Avoiding overly verbose or ambiguous language

        Based on OpenAI's official prompt engineering guide and community best practices.
        """


# Global instance for easy access
OPENAI_OPTIMIZATION_PROMPT = OpenAIOptimizationPrompt()
