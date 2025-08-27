"""
Optimization prompts for different LLM model families.

This module contains family-specific optimization prompts tailored to each
model architecture based on official documentation and best practices.
"""

from .base import BaseOptimizationPrompt
from .openai import OpenAI_OPTIMIZATION_PROMPT
from .anthropic import Anthropic_OPTIMIZATION_PROMPT
from .google import Google_OPTIMIZATION_PROMPT
from .meta import Meta_OPTIMIZATION_PROMPT
from .mistral import Mistral_OPTIMIZATION_PROMPT

__all__ = [
    "BaseOptimizationPrompt",
    "OpenAI_OPTIMIZATION_PROMPT",
    "Anthropic_OPTIMIZATION_PROMPT",
    "Google_OPTIMIZATION_PROMPT",
    "Meta_OPTIMIZATION_PROMPT",
    "Mistral_OPTIMIZATION_PROMPT",
]
