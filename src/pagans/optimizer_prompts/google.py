"""
Google family optimization prompts.

This module contains detailed, research-specific optimization prompts
for Google's Gemini models based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt


class GoogleOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for Google Gemini models."""

    def __init__(self):
        super().__init__("google")

    def get_model_family(self) -> str:
        """Get the model family name."""
        return "Google Gemini"

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the Google-specific optimization prompt.

        Based on Google's prompt engineering best practices and documentation.
        """
        return super().get_prompt(original_prompt, target_model)

    def get_description(self) -> str:
        """Get description of Google optimization approach."""
        return """
        Optimization strategy for Google Gemini models focusing on:
        - Clear, context-rich instructions with specific examples
        - Well-structured formatting with proper organization
        - Explicit task definition and expected output format
        - Context setting and background information
        - Examples that demonstrate the desired behavior
        - Appropriate use of formatting and structure

        Based on Google's official prompt engineering guide and best practices.
        """


# Global instance for easy access
GOOGLE_OPTIMIZATION_PROMPT = GoogleOptimizationPrompt()
