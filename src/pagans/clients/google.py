"""Google AI Studio optimizer client implementation."""

from typing import Any

from ..models import Provider
from .openai_compatible import OpenAICompatibleClient


class GoogleAIStudioClient(OpenAICompatibleClient):
    """Google AI Studio client via the OpenAI-compatible endpoint."""

    def __init__(self, provider: Provider, config: dict[str, Any]):
        super().__init__(
            provider=provider,
            config=config,
            default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
