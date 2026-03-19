"""Shared OpenAI-compatible client for PAGANS providers."""

import asyncio
import time
from typing import Any

import httpx

from ..constants import (
    CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_HEADERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
)
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


class OpenAICompatibleClient(BaseClient):
    """Client for providers exposing an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        provider: Provider,
        config: dict[str, Any],
        default_base_url: str,
    ):
        super().__init__(provider, config)

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = config.get("base_url", default_base_url).rstrip("/")
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self.max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.retry_delay = config.get("retry_delay", DEFAULT_RETRY_DELAY)

        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                **DEFAULT_HEADERS,
                "Authorization": f"Bearer {api_key}",
                **config.get("additional_headers", {}),
            },
        )
        self.last_request_time = 0.0
        self.rate_limit_delay = 0.0

    async def close(self) -> None:
        await self.client.aclose()

    async def _enforce_rate_limit(self) -> None:
        current_time = time.time()
        if self.rate_limit_delay > 0:
            wait_time = self.rate_limit_delay - (current_time - self.last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self.last_request_time = time.time()

    def _parse_retry_after(self, response: httpx.Response) -> int | None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                return None
        return None

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._enforce_rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = {
            **DEFAULT_HEADERS,
            "Authorization": f"Bearer {self.api_key}",
            **self.config.get("additional_headers", {}),
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code in {401, 403}:
                    raise PAGANSAuthenticationError("Invalid API key or access forbidden")
                if response.status_code == 404:
                    error_data = response.json()
                    raise PAGANSModelNotFoundError(
                        error_data.get("error", {}).get("message", "Model not found")
                    )
                if response.status_code == 429:
                    error_data = response.json()
                    error_message = (
                        error_data.get("error", {}).get("message", "").lower()
                    )
                    if "quota" in error_message:
                        raise PAGANSQuotaExceededError("API quota exceeded")
                    raise PAGANSRateLimitError(
                        retry_after=self._parse_retry_after(response)
                    )
                if response.status_code >= 500:
                    raise PAGANSNetworkError(f"Server error: {response.status_code}")

                error_data = response.json()
                raise PAGANSOpenRouterAPIError(
                    message=error_data.get("error", {}).get("message", "Unknown error"),
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
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response_data = await self._make_request(
            method="POST",
            endpoint=CHAT_COMPLETIONS_ENDPOINT,
            data=request_data,
        )

        if "choices" not in response_data or len(response_data["choices"]) == 0:
            raise PAGANSOpenRouterAPIError("No choices in response")

        choice = response_data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content")
        if content is None:
            raise PAGANSOpenRouterAPIError("Invalid response format")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_chunks: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text" and chunk.get("text"):
                        text_chunks.append(str(chunk["text"]))
                    elif chunk.get("type") == "output_text" and chunk.get("text"):
                        text_chunks.append(str(chunk["text"]))
            merged = "".join(text_chunks).strip()
            if merged:
                return merged

        raise PAGANSOpenRouterAPIError("Unable to extract optimized prompt text")

    async def get_models(self) -> dict[str, Any]:
        return await self._make_request(method="GET", endpoint="/models")

    async def validate_model(self, model_name: str) -> bool:
        try:
            models_data = await self.get_models()
            models = models_data.get("data", [])
            return any(m.get("id") == model_name for m in models if isinstance(m, dict))
        except Exception:
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
