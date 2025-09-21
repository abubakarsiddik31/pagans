"""
Provider factory for creating and managing provider clients.

This module implements the factory pattern for creating provider-specific
clients and managing their lifecycle.
"""

import asyncio
from typing import Dict, Any, Optional, Type

from ..clients.base import BaseClient
from ..models import Provider
from ..exceptions import PAGANSConfigurationError


class ProviderFactory:
    """
    Factory for creating and managing provider clients.

    This class implements the factory pattern to create provider-specific
    clients and manage their configuration and lifecycle.
    """

    def __init__(self):
        """Initialize the provider factory."""
        self._clients: Dict[Provider, BaseClient] = {}
        self._client_classes: Dict[Provider, Type[BaseClient]] = {}
        self._configs: Dict[Provider, Dict[str, Any]] = {}

    def register_client(
        self,
        provider: Provider,
        client_class: Type[BaseClient],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a client class for a provider.

        Args:
            provider: The provider to register the client for
            client_class: The client class to register
            config: Default configuration for the provider
        """
        self._client_classes[provider] = client_class
        if config:
            self._configs[provider] = config

    def get_client(self, provider: Provider, config: Optional[Dict[str, Any]] = None) -> BaseClient:
        """
        Get or create a client for the specified provider.

        Args:
            provider: The provider to get the client for
            config: Configuration to use for the client

        Returns:
            The client instance for the provider

        Raises:
            ConfigurationError: If no client is registered for the provider
        """
        if provider not in self._client_classes:
            raise PAGANSConfigurationError(f"No client registered for provider: {provider.value}")

        if provider not in self._clients:
            # Merge default config with provided config
            client_config = self._configs.get(provider, {}).copy()
            if config:
                client_config.update(config)

            # Create new client instance
            client_class = self._client_classes[provider]
            self._clients[provider] = client_class(provider, client_config)

        return self._clients[provider]

    def get_or_create_client(
        self,
        provider: Provider,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseClient:
        """
        Get an existing client or create a new one for the provider.

        Args:
            provider: The provider to get/create the client for
            config: Configuration to use for the client

        Returns:
            The client instance for the provider
        """
        return self.get_client(provider, config)

    def close_client(self, provider: Provider) -> None:
        """
        Close and remove a client for the specified provider.

        Args:
            provider: The provider whose client should be closed
        """
        if provider in self._clients:
            asyncio.create_task(self._clients[provider].close())
            del self._clients[provider]

    def close_all_clients(self) -> None:
        """Close all registered clients."""
        for provider in list(self._clients.keys()):
            self.close_client(provider)

    def is_client_registered(self, provider: Provider) -> bool:
        """
        Check if a client is registered for the provider.

        Args:
            provider: The provider to check

        Returns:
            True if a client is registered, False otherwise
        """
        return provider in self._client_classes

    def get_registered_providers(self) -> list[Provider]:
        """
        Get all providers that have registered clients.

        Returns:
            List of registered providers
        """
        return list(self._client_classes.keys())

    def get_active_clients(self) -> Dict[Provider, BaseClient]:
        """
        Get all currently active clients.

        Returns:
            Dictionary mapping providers to their active clients
        """
        return self._clients.copy()


# Global factory instance
_provider_factory = ProviderFactory()


def get_provider_factory() -> ProviderFactory:
    """
    Get the global provider factory instance.

    Returns:
        The global ProviderFactory instance
    """
    return _provider_factory


def register_provider_client(
    provider: Provider,
    client_class: Type[BaseClient],
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a client class for a provider in the global factory.

    Args:
        provider: The provider to register the client for
        client_class: The client class to register
        config: Default configuration for the provider
    """
    get_provider_factory().register_client(provider, client_class, config)


def get_provider_client(
    provider: Provider,
    config: Optional[Dict[str, Any]] = None
) -> BaseClient:
    """
    Get or create a client for the specified provider using the global factory.

    Args:
        provider: The provider to get the client for
        config: Configuration to use for the client

    Returns:
        The client instance for the provider
    """
    return get_provider_factory().get_client(provider, config)