"""
OpenRouter client for PAGANS.

This module provides the OpenRouter client for prompt optimization.
"""

from .clients.openrouter import OpenRouterClient

# Re-export OpenRouterClient for convenience
__all__ = ["OpenRouterClient"]
