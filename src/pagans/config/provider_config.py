"""
Provider-specific configuration management for PAGANS.

This module handles configuration loading, validation, and management
for different LLM providers.
"""

import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

from ..models import Provider
from ..exceptions import PAGANSConfigurationError


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    additional_headers: Dict[str, str] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    rate_limits: Dict[str, Union[int, float]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.api_key or not self.api_key.strip():
            raise PAGANSConfigurationError("API key is required")

        if self.timeout <= 0:
            raise PAGANSConfigurationError("Timeout must be positive")

        if self.max_retries < 0:
            raise PAGANSConfigurationError("Max retries must be non-negative")

        if self.retry_delay <= 0:
            raise PAGANSConfigurationError("Retry delay must be positive")


class ConfigManager:
    """
    Manager for provider-specific configurations.

    This class handles loading, validating, and managing configurations
    for different LLM providers.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self._configs: Dict[Provider, ProviderConfig] = {}
        self._default_configs: Dict[Provider, Dict[str, Any]] = {
            Provider.OPENROUTER: {
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
            Provider.OPENAI: {
                "base_url": "https://api.openai.com/v1",
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
            Provider.ANTHROPIC: {
                "base_url": "https://api.anthropic.com/v1",
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
            Provider.GOOGLE: {
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
        }

    def load_from_env(self, provider: Provider) -> ProviderConfig:
        """
        Load configuration for a provider from environment variables.

        Args:
            provider: The provider to load configuration for

        Returns:
            ProviderConfig instance with loaded configuration
        """
        config = ProviderConfig()

        # Load provider-specific environment variables
        env_prefix = provider.value.upper()

        # API key
        api_key_env = f"{env_prefix}_API_KEY"
        config.api_key = os.getenv(api_key_env)

        # Base URL
        base_url_env = f"{env_prefix}_BASE_URL"
        if os.getenv(base_url_env):
            config.base_url = os.getenv(base_url_env)

        # Timeout
        timeout_env = f"{env_prefix}_TIMEOUT"
        if os.getenv(timeout_env):
            try:
                config.timeout = float(os.getenv(timeout_env, "30.0"))
            except ValueError:
                pass

        # Max retries
        max_retries_env = f"{env_prefix}_MAX_RETRIES"
        if os.getenv(max_retries_env):
            try:
                config.max_retries = int(os.getenv(max_retries_env, "3"))
            except ValueError:
                pass

        # Retry delay
        retry_delay_env = f"{env_prefix}_RETRY_DELAY"
        if os.getenv(retry_delay_env):
            try:
                config.retry_delay = float(os.getenv(retry_delay_env, "1.0"))
            except ValueError:
                pass

        # Use default base URL if not set
        if not config.base_url and provider in self._default_configs:
            config.base_url = self._default_configs[provider].get("base_url")

        return config

    def get_config(self, provider: Provider, overrides: Optional[Dict[str, Any]] = None) -> ProviderConfig:
        """
        Get configuration for a provider.

        Args:
            provider: The provider to get configuration for
            overrides: Optional configuration overrides

        Returns:
            ProviderConfig instance
        """
        if provider not in self._configs:
            self._configs[provider] = self.load_from_env(provider)

        config = self._configs[provider]

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        config.validate()
        return config

    def set_config(self, provider: Provider, config: ProviderConfig) -> None:
        """
        Set configuration for a provider.

        Args:
            provider: The provider to set configuration for
            config: The configuration to set
        """
        config.validate()
        self._configs[provider] = config

    def get_all_configs(self) -> Dict[Provider, ProviderConfig]:
        """
        Get all configured providers and their configurations.

        Returns:
            Dictionary mapping providers to their configurations
        """
        return self._configs.copy()

    def clear_config(self, provider: Provider) -> None:
        """
        Clear configuration for a provider.

        Args:
            provider: The provider to clear configuration for
        """
        if provider in self._configs:
            del self._configs[provider]

    def clear_all_configs(self) -> None:
        """Clear all provider configurations."""
        self._configs.clear()


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.

    Returns:
        The global ConfigManager instance
    """
    return _config_manager


def get_provider_config(
    provider: Provider,
    overrides: Optional[Dict[str, Any]] = None
) -> ProviderConfig:
    """
    Get configuration for a provider using the global configuration manager.

    Args:
        provider: The provider to get configuration for
        overrides: Optional configuration overrides

    Returns:
        ProviderConfig instance
    """
    return get_config_manager().get_config(provider, overrides)


def set_provider_config(provider: Provider, config: ProviderConfig) -> None:
    """
    Set configuration for a provider using the global configuration manager.

    Args:
        provider: The provider to set configuration for
        config: The configuration to set
    """
    get_config_manager().set_config(provider, config)