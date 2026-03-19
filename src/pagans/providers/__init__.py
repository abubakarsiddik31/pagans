"""Provider factory and management system."""

from ..clients.anthropic import AnthropicClient
from ..clients.google import GoogleAIStudioClient
from ..clients.openai import OpenAIClient
from ..clients.openrouter import OpenRouterClient
from ..clients.zai import ZAIClient
from ..models import Provider
from .factory import (
    ProviderFactory,
    get_provider_client,
    get_provider_factory,
    register_provider_client,
)

register_provider_client(Provider.OPENROUTER, OpenRouterClient)
register_provider_client(Provider.OPENAI, OpenAIClient)
register_provider_client(Provider.GOOGLE, GoogleAIStudioClient)
register_provider_client(Provider.ANTHROPIC, AnthropicClient)
register_provider_client(Provider.ZAI, ZAIClient)

__all__ = [
    "ProviderFactory",
    "get_provider_client",
    "get_provider_factory",
    "register_provider_client",
]
