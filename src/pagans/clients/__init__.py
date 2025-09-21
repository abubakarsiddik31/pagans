"""
Client implementations for OpenRouter.

This module contains the OpenRouter client implementation
that follows the BaseClient interface.
"""

from .base import BaseClient
from .openrouter import OpenRouterClient

__all__ = [
    "BaseClient",
    "OpenRouterClient",
]