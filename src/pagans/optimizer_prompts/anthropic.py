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
        Advanced optimization strategy for Anthropic Claude models (Claude 4 Opus/Sonnet, Claude 3.7) based on official Anthropic documentation:
        
        Core Principles:
        - Clear, direct, and detailed instructions with explicit expectations
        - Context and motivation provided to help Claude understand goals
        - XML tags for structured organization and better parsing
        - Chain-of-thought reasoning for complex problem solving
        - Strategic use of examples and multishot prompting
        
        Claude 4 Specific Enhancements:
        - More explicit instruction style required for optimal performance
        - Thinking capabilities leveraged for reflection and planning
        - Quality modifiers to encourage detailed, comprehensive responses
        - Structured formatting that matches desired output style
        
        Key Techniques:
        - Tell Claude what TO do rather than what NOT to do
        - Use XML structure: <instructions>, <examples>, <context>, <output_format>
        - Enable step-by-step reasoning with "Think through this step by step"
        - Provide context for why certain approaches are preferred
        - Request high-quality, well-reasoned responses explicitly
        
        Based on Anthropic's latest prompt engineering documentation and Claude 4 best practices.
        """


# Global instance for easy access
ANTHROPIC_OPTIMIZATION_PROMPT = AnthropicOptimizationPrompt()
