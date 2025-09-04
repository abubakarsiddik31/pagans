"""
Google family optimization prompts.

This module contains detailed, research-specific optimization prompts
for Google's Gemini models based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt


class GoogleOptimizationPrompt(BaseOptimizationPrompt):
    """Optimization prompt for Google Gemini models."""

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the Google-specific optimization prompt.

        Based on Google's prompt engineering best practices and documentation.
        """

        family_specific_guidelines = """
        Google Gemini models work best with:
        - Clear, context-rich instructions with specific examples
        - Well-structured formatting with proper organization
        - Explicit task definition and expected output format
        - Context setting and background information
        - Examples that demonstrate the desired behavior
        - Appropriate use of formatting and structure
        - Clear constraints and guidelines
        - Step-by-step instructions for complex tasks
        - Multimodal considerations when applicable
        - Safety and ethical considerations

        Key optimization principles for Google models:
        1. Provide clear context and background information
        2. Include specific examples that demonstrate the desired behavior
        3. Define the task explicitly and specify the expected output format
        4. Use well-structured formatting with proper organization
        5. Include clear constraints and guidelines when needed
        6. Break down complex tasks into step-by-step instructions
        7. Consider multimodal capabilities when applicable
        8. Include safety and ethical considerations
        9. Use clear, direct language without ambiguity
        10. Provide relevant domain-specific context when helpful

        Model-specific considerations:
        - Gemini 2.5 Pro: Latest model with enhanced reasoning and multimodal capabilities
        - Gemini 2.5 Flash: Fast and efficient for most tasks
"""

        return f"""
You are an expert at optimizing prompts for Google's Gemini models ({', '.join(['Gemini 2.5 Pro', 'Gemini 2.5 Flash', 'Gemini 1.5 Pro', 'Gemini 1.5 Flash'])}).

{family_specific_guidelines}

Take this original prompt and optimize it specifically for Google Gemini models:

Original prompt: {original_prompt}
Target model: {target_model}

Return ONLY the optimized prompt, no explanations or meta-commentary.
"""

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
