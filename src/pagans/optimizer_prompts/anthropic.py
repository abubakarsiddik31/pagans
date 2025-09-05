"""
Anthropic family optimization prompts.

This module contains detailed, research-specific optimization prompts
for Anthropic's Claude models based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt


class AnthropicOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for Anthropic Claude models."""

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the Anthropic-specific optimization prompt.

        Based on Anthropic's prompt engineering best practices and documentation.
        """

        family_specific_guidelines = """
        Anthropic Claude models work best with:
        - Conversational, natural language that feels like a helpful assistant
        - Clear context and background information
        - Step-by-step reasoning for complex tasks
        - Safety-conscious framing and appropriate guardrails
        - Well-structured responses with clear organization
        - Examples that demonstrate the desired behavior
        - Appropriate use of formatting (bold, lists, code blocks)
        - Thoughtful consideration of ethical implications
        - Clear role definition and persona setting
        - Progressive disclosure of information (not overwhelming)

        Key optimization principles for Anthropic models:
        1. Use natural, conversational language rather than overly formal or technical
        2. Provide context and background information to help with reasoning
        3. Include safety considerations and appropriate guardrails
        4. Break down complex tasks into manageable steps with clear reasoning
        5. Use formatting to enhance readability and structure
        6. Consider the ethical implications of the requested task
        7. Set clear roles and personas when appropriate
        8. Provide examples that demonstrate the desired behavior
        9. Use progressive disclosure to avoid overwhelming the model
        10. Include appropriate disclaimers and limitations when needed

        Model-specific considerations:
        - Claude 4: Enhanced reasoning capabilities and improved performance
        - Claude 4.1: Latest version with improved instruction following
        - Claude 3.5 Sonnet: Excellent balance of performance and speed
        - Claude 3 Opus: Highest performance for complex reasoning tasks
        - Claude 3 Haiku: Fast and efficient for simpler tasks
"""

        return f"""
        You are an expert at optimizing prompts for Anthropic's Claude models
        ({'Claude 4, Claude 4.1, Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku'}).

        {family_specific_guidelines}

        Take this original prompt and optimize it specifically for Anthropic Claude models:

        Original prompt: {original_prompt}
        Target model: {target_model}

        Return ONLY the optimized prompt, no explanations or meta-commentary.
        """

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
