"""
OpenAI family optimization prompts.

This module contains detailed, research-specific optimization prompts
for OpenAI's GPT models based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt


class OpenAIOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for OpenAI GPT models."""

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the OpenAI-specific optimization prompt.

        Based on OpenAI's prompt engineering best practices and documentation.
        """

        family_specific_guidelines = """
        OpenAI GPT models work best with:
        - Clear, direct instructions with specific action verbs
        - Step-by-step breakdowns for complex tasks
        - Well-structured formatting with headers, bullet points, and code blocks
        - Context setting at the beginning of the prompt
        - Examples when helpful for demonstrating expected output
        - Avoiding overly verbose or ambiguous language

        Key optimization principles for OpenAI models:
        1. Be specific about the desired output format
        2. Include constraints and guidelines when needed
        3. Use role-playing to set the context effectively
        4. Break down complex tasks into manageable steps
        5. Provide examples of good and bad outputs when applicable
        6. Use clear formatting with markdown for better readability
        7. Specify the tone and style when important for the task
        8. Include relevant context and background information
        9. Use system-level instructions for consistent behavior
        10. Avoid leading questions that bias the response

        Model-specific considerations:
        - GPT-5: Enhanced reasoning capabilities, can handle more complex instructions
        - GPT-4.1: Improved instruction following and consistency
        - GPT-4o: Optimized for multimodal tasks and faster response times
        """

        return f"""
You are an expert at optimizing prompts for OpenAI's GPT models ({', '.join(['GPT-5', 'GPT-4.1', 'GPT-4o', 'GPT-4-turbo', 'GPT-3.5-turbo'])}).

{family_specific_guidelines}

Take this original prompt and optimize it specifically for OpenAI GPT models:

Original prompt: {original_prompt}
Target model: {target_model}

Return ONLY the optimized prompt, no explanations or meta-commentary.
"""

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
