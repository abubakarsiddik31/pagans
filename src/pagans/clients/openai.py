"""OpenAI optimizer client implementation."""

from typing import Any

from ..models import Provider
from .openai_compatible import OpenAICompatibleClient


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI API client using Chat Completions."""

    def __init__(self, provider: Provider, config: dict[str, Any]):
        super().__init__(
            provider=provider,
            config=config,
            default_base_url="https://api.openai.com/v1",
        )
