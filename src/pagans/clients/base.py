"""
Base client abstract class for PAGANS provider clients.

This module defines the abstract base class that all provider-specific
clients must implement to ensure consistent interface across providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import Provider


class BaseClient(ABC):
    """
    Abstract base class for all provider clients.

    This class defines the interface that all provider-specific clients
    must implement to ensure consistency across different providers.
    """

    def __init__(self, provider: Provider, config: Dict[str, Any]):
        """
        Initialize the base client.

        Args:
            provider: The provider this client handles
            config: Configuration dictionary for the provider
        """
        self.provider = provider
        self.config = config

    @abstractmethod
    async def optimize_prompt(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """
        Optimize a prompt using the provider's API.

        Args:
            prompt: The original prompt to optimize
            model: The model to use for optimization
            system_prompt: The system prompt for optimization
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for the response

        Returns:
            The optimized prompt

        Raises:
            Various exceptions based on provider-specific errors
        """
        pass

    @abstractmethod
    async def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model is available on this provider.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model is available, False otherwise
        """
        pass

    @abstractmethod
    async def get_models(self) -> Dict[str, Any]:
        """
        Get available models from the provider.

        Returns:
            Dictionary of available models

        Raises:
            Various exceptions based on provider-specific errors
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and clean up resources."""
        pass

    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()