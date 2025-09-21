"""
OpenRouter client implementation for PAGANS.

This module provides a client for interacting with the OpenRouter API
with proper error handling, rate limiting, and retry logic.
"""

import asyncio
import time
from typing import Any, Dict

import httpx

from .base import BaseClient
from ..models import Provider
from ..exceptions import (
    PAGANSAuthenticationError,
    PAGANSModelNotFoundError,
    PAGANSNetworkError,
    PAGANSOpenRouterAPIError,
    PAGANSQuotaExceededError,
    PAGANSRateLimitError,
    PAGANSTimeoutError,
)
from ..constants import (
    API_VERSION,
    CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_HEADERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
)


class OpenRouterClient(BaseClient):
    """
    Client for interacting with the OpenRouter API.

    This client handles authentication, rate limiting, retry logic,
    and proper error handling for API requests.
    """

    def __init__(self, provider: Provider, config: Dict[str, Any]):
        """
        Initialize the OpenRouter client.

        Args:
            provider: The provider (should be Provider.OPENROUTER)
            config: Configuration dictionary
        """
        super().__init__(provider, config)

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self.max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.retry_delay = config.get("retry_delay", DEFAULT_RETRY_DELAY)

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                **DEFAULT_HEADERS,
                "Authorization": f"Bearer {api_key}",
                **config.get("additional_headers", {}),
            },
        )
        self.last_request_time = 0
        self.rate_limit_delay = 0

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        await self.client.aclose()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            params: Query parameters

        Returns:
            Response data

        Raises:
            Various exceptions based on error type
        """
        await self._enforce_rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = {**DEFAULT_HEADERS, "Authorization": f"Bearer {self.api_key}"}

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

                if response.status_code == 401:
                    msg = "Invalid API key"
                    raise PAGANSAuthenticationError(msg)
                if response.status_code == 403:
                    raise PAGANSAuthenticationError("Access forbidden")
                if response.status_code == 404:
                    error_data = response.json()
                    raise PAGANSModelNotFoundError(
                        error_data.get("error", {}).get("message", "Model not found")
                    )
                if response.status_code == 429:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", "").lower()
                    if "quota" in error_message:
                        raise PAGANSQuotaExceededError("API quota exceeded")
                    else:
                        retry_after = self._parse_retry_after(response)
                        raise PAGANSRateLimitError(retry_after=retry_after)
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

            except httpx.NetworkError as e:
                if attempt == self.max_retries:
                    raise PAGANSNetworkError(f"Network error: {e!s}")
                await asyncio.sleep(self.retry_delay)

            except (
                PAGANSOpenRouterAPIError,
                PAGANSAuthenticationError,
                PAGANSRateLimitError,
                PAGANSQuotaExceededError,
            ):
                raise
        return None

    def _parse_retry_after(self, response: httpx.Response) -> int | None:
        """Parse Retry-After header from response."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                try:
                    retry_time = time.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT")
                    retry_timestamp = time.mktime(retry_time)
                    return int(retry_timestamp - time.time())
                except ValueError:
                    pass
        return None

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        current_time = time.time()

        if self.rate_limit_delay > 0:
            wait_time = self.rate_limit_delay - (current_time - self.last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    async def optimize_prompt(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """
        Optimize a prompt using the OpenRouter API.

        Args:
            prompt: The original prompt to optimize
            model: The model to use for optimization
            system_prompt: The system prompt for optimization
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for the response

        Returns:
            The optimized prompt

        Raises:
            Various exceptions based on error type
        """
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
        if "message" not in choice or "content" not in choice["message"]:
            raise PAGANSOpenRouterAPIError("Invalid response format")

        return choice["message"]["content"].strip()

    async def get_models(self) -> Dict[str, Any]:
        """
        Get available models from OpenRouter API.

        Returns:
            Dictionary of available models

        Raises:
            Various exceptions based on error type
        """
        return await self._make_request(
            method="GET",
            endpoint=f"/{API_VERSION}/models",
        )

    async def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model is available.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model is available, False otherwise
        """
        try:
            models = await self.get_models()
            return any(model["id"] == model_name for model in models.get("data", []))
        except Exception:
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()