"""
Provider factory and management system.

This module contains the factory pattern implementation for creating
and managing provider-specific clients.
"""

# Import and register the OpenRouter client
from ..clients.openrouter import OpenRouterClient
from ..models import Provider
from .factory import (
    ProviderFactory,
    get_provider_client,
    get_provider_factory,
    register_provider_client,
)

# Register the OpenRouter client with the factory
register_provider_client(Provider.OPENROUTER, OpenRouterClient)

__all__ = [
    "ProviderFactory",
    "get_provider_client",
    "get_provider_factory",
    "register_provider_client",
]
