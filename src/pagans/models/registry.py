"""
Model family registry for extensible model support.

This module provides a registry system for managing model families,
providers, and their relationships in an extensible way.
"""

from typing import Dict, List, Set, Optional, Type, Any
from dataclasses import dataclass

from . import ModelFamily, Provider


@dataclass
class ModelFamilyInfo:
    """Information about a model family."""

    family: ModelFamily
    display_name: str
    description: str
    supported_providers: List[Provider]
    default_models: Dict[str, str]  # short_name -> provider_model_name
    aliases: Dict[str, str]  # alias -> canonical_name


@dataclass
class ProviderInfo:
    """Information about a provider."""

    provider: Provider
    display_name: str
    description: str
    base_url: str
    api_version: str
    supported_families: List[ModelFamily]
    model_mappings: Dict[str, str]  # short_name -> provider_model_name
    auth_header: str
    auth_prefix: str = "Bearer"


class ModelRegistry:
    """
    Registry for managing model families and providers.

    This class provides an extensible system for registering new model
    families and providers, and managing their relationships.
    """

    def __init__(self):
        """Initialize the model registry."""
        self._families: Dict[ModelFamily, ModelFamilyInfo] = {}
        self._providers: Dict[Provider, ProviderInfo] = {}
        self._model_to_family: Dict[str, ModelFamily] = {}
        self._family_to_models: Dict[ModelFamily, Set[str]] = {}

    def register_family(self, family_info: ModelFamilyInfo) -> None:
        """
        Register a model family.

        Args:
            family_info: Information about the model family
        """
        self._families[family_info.family] = family_info

        # Update model mappings
        for short_name, provider_model in family_info.default_models.items():
            self._model_to_family[short_name] = family_info.family
            self._model_to_family[provider_model] = family_info.family

        # Update family to models mapping
        if family_info.family not in self._family_to_models:
            self._family_to_models[family_info.family] = set()
        self._family_to_models[family_info.family].update(family_info.default_models.keys())
        self._family_to_models[family_info.family].update(family_info.default_models.values())

    def register_provider(self, provider_info: ProviderInfo) -> None:
        """
        Register a provider.

        Args:
            provider_info: Information about the provider
        """
        self._providers[provider_info.provider] = provider_info

    def get_family_info(self, family: ModelFamily) -> Optional[ModelFamilyInfo]:
        """
        Get information about a model family.

        Args:
            family: The model family to get information for

        Returns:
            ModelFamilyInfo if found, None otherwise
        """
        return self._families.get(family)

    def get_provider_info(self, provider: Provider) -> Optional[ProviderInfo]:
        """
        Get information about a provider.

        Args:
            provider: The provider to get information for

        Returns:
            ProviderInfo if found, None otherwise
        """
        return self._providers.get(provider)

    def get_model_family(self, model_name: str) -> Optional[ModelFamily]:
        """
        Get the model family for a model name.

        Args:
            model_name: The model name to look up

        Returns:
            ModelFamily if found, None otherwise
        """
        # Direct lookup
        if model_name in self._model_to_family:
            return self._model_to_family[model_name]

        # Try case-insensitive lookup
        model_name_lower = model_name.lower()
        for model, family in self._model_to_family.items():
            if model.lower() == model_name_lower:
                return family

        return None

    def get_models_for_family(self, family: ModelFamily) -> List[str]:
        """
        Get all models for a model family.

        Args:
            family: The model family to get models for

        Returns:
            List of model names
        """
        if family in self._family_to_models:
            return list(self._family_to_models[family])
        return []

    def get_supported_providers_for_family(self, family: ModelFamily) -> List[Provider]:
        """
        Get all providers that support a model family.

        Args:
            family: The model family to get providers for

        Returns:
            List of providers that support the family
        """
        family_info = self.get_family_info(family)
        if family_info:
            return family_info.supported_providers
        return []

    def get_provider_model_name(
        self,
        short_name: str,
        provider: Provider
    ) -> Optional[str]:
        """
        Get the provider-specific model name for a short model name.

        Args:
            short_name: The short model name
            provider: The provider to get the model name for

        Returns:
            Provider-specific model name if found, None otherwise
        """
        provider_info = self.get_provider_info(provider)
        if provider_info and short_name in provider_info.model_mappings:
            return provider_info.model_mappings[short_name]

        # Check if family has default mapping
        family = self.get_model_family(short_name)
        if family:
            family_info = self.get_family_info(family)
            if family_info and short_name in family_info.default_models:
                return family_info.default_models[short_name]

        return None

    def is_model_supported(self, model_name: str) -> bool:
        """
        Check if a model is supported.

        Args:
            model_name: The model name to check

        Returns:
            True if the model is supported, False otherwise
        """
        return self.get_model_family(model_name) is not None

    def get_all_families(self) -> List[ModelFamily]:
        """
        Get all registered model families.

        Returns:
            List of all registered model families
        """
        return list(self._families.keys())

    def get_all_providers(self) -> List[Provider]:
        """
        Get all registered providers.

        Returns:
            List of all registered providers
        """
        return list(self._providers.keys())

    def get_families_for_provider(self, provider: Provider) -> List[ModelFamily]:
        """
        Get all model families supported by a provider.

        Args:
            provider: The provider to get families for

        Returns:
            List of model families supported by the provider
        """
        provider_info = self.get_provider_info(provider)
        if provider_info:
            return provider_info.supported_families
        return []


# Global registry instance
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Returns:
        The global ModelRegistry instance
    """
    return _model_registry


def register_model_family(family_info: ModelFamilyInfo) -> None:
    """
    Register a model family in the global registry.

    Args:
        family_info: Information about the model family
    """
    get_model_registry().register_family(family_info)


def register_provider(provider_info: ProviderInfo) -> None:
    """
    Register a provider in the global registry.

    Args:
        provider_info: Information about the provider
    """
    get_model_registry().register_provider(provider_info)


def get_model_family(model_name: str) -> Optional[ModelFamily]:
    """
    Get the model family for a model name using the global registry.

    Args:
        model_name: The model name to look up

    Returns:
        ModelFamily if found, None otherwise
    """
    return get_model_registry().get_model_family(model_name)


def is_model_supported(model_name: str) -> bool:
    """
    Check if a model is supported using the global registry.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is supported, False otherwise
    """
    return get_model_registry().is_model_supported(model_name)