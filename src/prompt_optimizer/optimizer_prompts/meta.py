"""
Meta family optimization prompts.

This module contains detailed, research-specific optimization prompts
for Meta's Llama models based on official documentation and best practices.
"""

from typing import Dict, Any

from .base import BaseOptimizationPrompt


class MetaOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for Meta Llama models."""

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the Meta-specific optimization prompt.

        Based on Meta's prompt engineering best practices and documentation.
        """

        family_specific_guidelines = """
Meta Llama models work best with:
- Explicit formatting and clear structure
- Detailed context and background information
- Step-by-step instructions for complex tasks
- Well-defined roles and personas
- Clear constraints and guidelines
- Appropriate use of code blocks and formatting
- Direct, unambiguous language
- System-level instructions for consistent behavior
- Progressive disclosure of information
- Safety considerations and appropriate guardrails

Key optimization principles for Meta models:
1. Use explicit formatting with clear structure and organization
2. Provide detailed context and background information
3. Break down complex tasks into step-by-step instructions
4. Define clear roles and personas when appropriate
5. Include explicit constraints and guidelines
6. Use direct, unambiguous language
7. Provide system-level instructions for consistent behavior
8. Use progressive disclosure to avoid overwhelming the model
9. Include appropriate safety considerations and guardrails
10. Use code blocks for technical or programming-related tasks

Model-specific considerations:
- Llama 3.1 70B: Highest performance with excellent reasoning capabilities
- Llama 3.1 8B: Efficient and capable for most tasks
- Llama 3 70B: Strong performance with good reasoning abilities
- Llama 3 8B: Fast and efficient for straightforward tasks
"""

        return f"""
You are an expert at optimizing prompts for Meta's Llama models ({', '.join(['Llama 3.1 70B', 'Llama 3.1 8B', 'Llama 3 70B', 'Llama 3 8B'])}).

{family_specific_guidelines}

Take this original prompt and optimize it specifically for Meta Llama models:

Original prompt: {original_prompt}
Target model: {target_model}

Return ONLY the optimized prompt, no explanations or meta-commentary.
"""

    def get_description(self) -> str:
        """Get description of Meta optimization approach."""
        return """
Optimization strategy for Meta Llama models focusing on:
- Explicit formatting and clear structure
- Detailed context and background information
- Step-by-step instructions for complex tasks
- Well-defined roles and personas
- Clear constraints and guidelines
- Appropriate use of code blocks and formatting

Based on Meta's official documentation and community best practices.
"""


# Global instance for easy access
META_OPTIMIZATION_PROMPT = MetaOptimizationPrompt()
