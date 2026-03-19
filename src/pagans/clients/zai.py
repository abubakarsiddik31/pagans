"""Z.ai optimizer client implementation."""

from typing import Any

from ..models import Provider
from .openai_compatible import OpenAICompatibleClient


class ZAIClient(OpenAICompatibleClient):
    """Z.ai client via the OpenAI-compatible chat completions endpoint."""

    def __init__(self, provider: Provider, config: dict[str, Any]):
        super().__init__(
            provider=provider,
            config=config,
            default_base_url="https://api.z.ai/api/paas/v4",
        )
