"""
Provider factory and management system.

This module contains the factory pattern implementation for creating
and managing provider-specific clients.
"""

from .factory import (
    ProviderFactory,
    get_provider_factory,
    register_provider_client,
    get_provider_client,
)

# Import and register the OpenRouter client
from ..clients.openrouter import OpenRouterClient
from ..models import Provider

# Register the OpenRouter client with the factory
register_provider_client(Provider.OPENROUTER, OpenRouterClient)

__all__ = [
    "ProviderFactory",
    "get_provider_factory",
    "register_provider_client",
    "get_provider_client",
]