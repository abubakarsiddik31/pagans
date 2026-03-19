"""Anthropic optimizer client implementation."""

import asyncio
from typing import Any

import httpx

from ..constants import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY, DEFAULT_TIMEOUT
from ..exceptions import (
    PAGANSAuthenticationError,
    PAGANSModelNotFoundError,
    PAGANSNetworkError,
    PAGANSOpenRouterAPIError,
    PAGANSQuotaExceededError,
    PAGANSRateLimitError,
    PAGANSTimeoutError,
)
from ..models import Provider
from .base import BaseClient


class AnthropicClient(BaseClient):
    """Anthropic Messages API client."""

    def __init__(self, provider: Provider, config: dict[str, Any]):
        super().__init__(provider, config)

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = config.get("base_url", "https://api.anthropic.com/v1").rstrip(
            "/"
        )
        self.api_version = config.get("api_version", "2023-06-01")
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self.max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.retry_delay = config.get("retry_delay", DEFAULT_RETRY_DELAY)

        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
            **self.config.get("additional_headers", {}),
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=self._headers(),
                )

                if response.status_code in {200, 201}:
                    return response.json()
                if response.status_code in {401, 403}:
                    raise PAGANSAuthenticationError("Invalid API key or access forbidden")
                if response.status_code == 404:
                    raise PAGANSModelNotFoundError("Model not found")
                if response.status_code == 429:
                    raise PAGANSRateLimitError()
                if response.status_code >= 500:
                    raise PAGANSNetworkError(f"Server error: {response.status_code}")

                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                if "quota" in error_message.lower():
                    raise PAGANSQuotaExceededError("API quota exceeded")
                raise PAGANSOpenRouterAPIError(
                    message=error_message,
                    status_code=response.status_code,
                    response_data=error_data,
                )

            except httpx.TimeoutException:
                if attempt == self.max_retries:
                    raise PAGANSTimeoutError(self.timeout)
                await asyncio.sleep(self.retry_delay)
            except httpx.NetworkError as exc:
                if attempt == self.max_retries:
                    raise PAGANSNetworkError(f"Network error: {exc!s}")
                await asyncio.sleep(self.retry_delay)
            except (
                PAGANSOpenRouterAPIError,
                PAGANSAuthenticationError,
                PAGANSRateLimitError,
                PAGANSQuotaExceededError,
            ):
                raise

        return {}

    async def optimize_prompt(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        request_data = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response_data = await self._make_request(
            method="POST",
            endpoint="/messages",
            data=request_data,
        )

        content = response_data.get("content", [])
        text_chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                text_chunks.append(str(part["text"]))

        merged = "".join(text_chunks).strip()
        if not merged:
            raise PAGANSOpenRouterAPIError("No text content in Anthropic response")
        return merged

    async def get_models(self) -> dict[str, Any]:
        return await self._make_request(method="GET", endpoint="/models")

    async def validate_model(self, model_name: str) -> bool:
        try:
            await self._make_request(method="GET", endpoint=f"/models/{model_name}")
            return True
        except Exception:
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
