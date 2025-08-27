"""
Mistral family optimization prompts.

This module contains detailed, research-specific optimization prompts
for Mistral models based on official documentation and best practices.
"""

from typing import Dict, Any

from .base import BaseOptimizationPrompt


class MistralOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for Mistral models."""

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the Mistral-specific optimization prompt.

        Based on Mistral's prompt engineering best practices and documentation.
        """

        family_specific_guidelines = """
Mistral models work best with:
- Concise, focused instructions with clear objectives
- Task-oriented language that gets straight to the point
- Well-structured formatting with appropriate organization
- Context setting when necessary for complex tasks
- Direct and unambiguous language
- Clear examples for demonstration purposes
- Appropriate use of formatting for readability
- System-level instructions for consistent behavior
- Safety considerations and appropriate guardrails
- Efficient use of tokens for optimal performance

Key optimization principles for Mistral models:
1. Use concise, focused instructions that get straight to the point
2. Be task-oriented and clear about the objective
3. Use well-structured formatting with appropriate organization
4. Provide context only when necessary for complex tasks
5. Use direct and unambiguous language
6. Include clear examples when helpful for demonstration
7. Use appropriate formatting for better readability
8. Provide system-level instructions for consistent behavior
9. Include safety considerations and appropriate guardrails
10. Be efficient with token usage for optimal performance

Model-specific considerations:
- Mixtral 8x7B: High performance with strong reasoning capabilities
- Mistral 7B Instruct: Efficient and capable for most instruction-following tasks
- Both models perform well with clear, concise prompts
- Mixtral benefits from more complex, multi-step reasoning tasks
- Mistral 7B excels at straightforward instruction following
"""

        return f"""
You are an expert at optimizing prompts for Mistral models ({', '.join(['Mixtral 8x7B', 'Mistral 7B Instruct'])}).

{family_specific_guidelines}

Take this original prompt and optimize it specifically for Mistral models:

Original prompt: {original_prompt}
Target model: {target_model}

Return ONLY the optimized prompt, no explanations or meta-commentary.
"""

    def get_description(self) -> str:
        """Get description of Mistral optimization approach."""
        return """
Optimization strategy for Mistral models focusing on:
- Concise, focused instructions with clear objectives
- Task-oriented language that gets straight to the point
- Well-structured formatting with appropriate organization
- Context setting when necessary for complex tasks
- Direct and unambiguous language
- Clear examples for demonstration purposes

Based on Mistral's official documentation and community best practices.
"""


# Global instance for easy access
MISTRAL_OPTIMIZATION_PROMPT = MistralOptimizationPrompt()
