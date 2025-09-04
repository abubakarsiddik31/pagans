"""
Unit tests for the client module.

This module contains tests for the OpenRouterClient class and its API interactions.
"""

import asyncio
from datetime import UTC
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.pagans.client import OpenRouterClient
from src.pagans.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    OpenRouterAPIError,
    QuotaExceededError,
    RateLimitError,
)


class TestOpenRouterClientInitialization:
    """Test cases for OpenRouterClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        api_key = "test-api-key"
        client = OpenRouterClient(api_key=api_key)

        assert client.api_key == api_key
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert client.client is not None
        assert client.last_request_time == 0
        assert client.rate_limit_delay == 0

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        api_key = "test-api-key"
        base_url = "https://custom.api.url"
        timeout = 30.0
        max_retries = 5
        retry_delay = 2.0

        client = OpenRouterClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        assert client.api_key == api_key
        assert client.base_url == base_url
        assert client.timeout == timeout
        assert client.max_retries == max_retries
        assert client.retry_delay == retry_delay

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with pytest.raises(ValueError, match="API key is required"):
            OpenRouterClient(api_key=None)

        with pytest.raises(ValueError, match="API key is required"):
            OpenRouterClient(api_key="")

    def test_init_with_base_url_trailing_slash(self):
        """Test initialization with base URL that has trailing slash."""
        client = OpenRouterClient(api_key="test", base_url="https://api.example.com/")
        assert client.base_url == "https://api.example.com"


class TestOpenRouterClientRequestHandling:
    """Test cases for request handling and error management."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.httpx.AsyncClient") as mock_httpx_class:
            mock_httpx = AsyncMock()
            mock_httpx_class.return_value = mock_httpx

            client = OpenRouterClient(api_key="test-api-key")
            client.client = mock_httpx

            return client

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Optimized prompt"}}]
        }
        return mock_response

    def test_make_request_success(self, mock_client, mock_response):
        """Test successful request handling."""
        mock_client.client.request.return_value = mock_response

        result = asyncio.run(
            mock_client._make_request(
                method="POST",
                endpoint="/chat/completions",
                data={"test": "data"},
            )
        )

        assert result == {"choices": [{"message": {"content": "Optimized prompt"}}]}
        mock_client.client.request.assert_called_once()

    def test_make_request_authentication_error(self, mock_client, mock_response):
        """Test authentication error handling."""
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_client.client.request.return_value = mock_response

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_forbidden_error(self, mock_client, mock_response):
        """Test forbidden error handling."""
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": {"message": "Access forbidden"}}
        mock_client.client.request.return_value = mock_response

        with pytest.raises(AuthenticationError, match="Access forbidden"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_model_not_found(self, mock_client, mock_response):
        """Test model not found error handling."""
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Model not found"}}
        mock_client.client.request.return_value = mock_response

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_rate_limit_error(self, mock_client, mock_response):
        """Test rate limit error handling."""
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_client.client.request.return_value = mock_response

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_quota_exceeded(self, mock_client, mock_response):
        """Test quota exceeded error handling."""
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Quota exceeded"}}
        mock_client.client.request.return_value = mock_response

        with pytest.raises(QuotaExceededError, match="API quota exceeded"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_server_error(self, mock_client, mock_response):
        """Test server error handling."""
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {"message": "Internal server error"}
        }
        mock_client.client.request.return_value = mock_response

        with pytest.raises(NetworkError, match="Server error: 500"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_client_error(self, mock_client, mock_response):
        """Test client error handling."""
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Bad request"}}
        mock_client.client.request.return_value = mock_response

        with pytest.raises(OpenRouterAPIError, match="Unknown error"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

    def test_make_request_retry_on_network_error(self, mock_client):
        """Test retry logic on network errors."""
        mock_client.client.request.side_effect = [
            httpx.NetworkError("Network error"),
            Mock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Success"}}]},
            ),
        ]

        result = asyncio.run(
            mock_client._make_request(
                method="POST",
                endpoint="/chat/completions",
                data={"test": "data"},
            )
        )

        assert result == {"choices": [{"message": {"content": "Success"}}]}
        assert mock_client.client.request.call_count == 2

    def test_make_request_retry_on_timeout(self, mock_client):
        """Test retry logic on timeout errors."""
        mock_client.client.request.side_effect = [
            httpx.TimeoutException("Timeout"),
            Mock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "Success"}}]},
            ),
        ]

        result = asyncio.run(
            mock_client._make_request(
                method="POST",
                endpoint="/chat/completions",
                data={"test": "data"},
            )
        )

        assert result == {"choices": [{"message": {"content": "Success"}}]}
        assert mock_client.client.request.call_count == 2

    def test_make_request_max_retries_exceeded(self, mock_client):
        """Test handling when max retries are exceeded."""
        mock_client.client.request.side_effect = httpx.NetworkError("Network error")

        with pytest.raises(NetworkError, match="Network error"):
            asyncio.run(
                mock_client._make_request(
                    method="POST",
                    endpoint="/chat/completions",
                    data={"test": "data"},
                )
            )

        # Should have been called max_retries + 1 times
        assert mock_client.client.request.call_count == 4  # default max_retries = 3


class TestOpenRouterClientRateLimiting:
    """Test cases for rate limiting functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.httpx.AsyncClient") as mock_httpx_class:
            mock_httpx = AsyncMock()
            mock_httpx_class.return_value = mock_httpx

            client = OpenRouterClient(api_key="test-api-key")
            client.client = mock_httpx

            return client

    def test_enforce_rate_limit_no_delay(self, mock_client):
        """Test rate limiting with no delay needed."""
        # Mock time.time() to return a fixed value
        with patch("time.time", return_value=1000.0):
            asyncio.run(mock_client._enforce_rate_limit())

        # Should not have waited
        assert mock_client.last_request_time == 1000.0

    def test_enforce_rate_limit_with_delay(self, mock_client):
        """Test rate limiting with delay needed."""
        # Set up rate limiting
        mock_client.rate_limit_delay = 1000.0
        mock_client.last_request_time = 500.0

        # Mock time.time() to return a value that requires waiting
        with patch("time.time", return_value=600.0):
            with patch("asyncio.sleep") as mock_sleep:
                asyncio.run(mock_client._enforce_rate_limit())

        # Should have waited for the remaining time
        mock_sleep.assert_called_once_with(400.0)
        assert mock_client.last_request_time == 600.0

    def test_parse_retry_after_seconds(self, mock_client):
        """Test parsing Retry-After header with seconds."""
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "60"}

        result = mock_client._parse_retry_after(mock_response)
        assert result == 60

    def test_parse_retry_after_http_date(self, mock_client):
        """Test parsing Retry-After header with HTTP date."""
        from datetime import datetime

        # Create a future date
        future_time = datetime.now(UTC).timestamp() + 3600
        http_date = datetime.fromtimestamp(future_time, UTC).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )

        mock_response = Mock()
        mock_response.headers = {"Retry-After": http_date}

        result = mock_client._parse_retry_after(mock_response)
        assert result == 3600

    def test_parse_retry_after_invalid(self, mock_client):
        """Test parsing Retry-After header with invalid value."""
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "invalid"}

        result = mock_client._parse_retry_after(mock_response)
        assert result is None

    def test_parse_retry_after_missing(self, mock_client):
        """Test parsing Retry-After header when missing."""
        mock_response = Mock()
        mock_response.headers = {}

        result = mock_client._parse_retry_after(mock_response)
        assert result is None


class TestOpenRouterClientAPIMethods:
    """Test cases for API methods."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        with patch("src.pagans.client.httpx.AsyncClient") as mock_httpx_class:
            mock_httpx = AsyncMock()
            mock_httpx_class.return_value = mock_httpx

            client = OpenRouterClient(api_key="test-api-key")
            client.client = mock_httpx

            return client

    def test_optimize_prompt_success(self, mock_client):
        """Test successful prompt optimization."""
        mock_client.client.request.return_value = Mock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Optimized prompt"}}]},
        )

        result = asyncio.run(
            mock_client.optimize_prompt(
                prompt="Original prompt",
                model="gpt-4o",
                system_prompt="System prompt",
            )
        )

        assert result == "Optimized prompt"
        mock_client.client.request.assert_called_once()

    def test_optimize_prompt_invalid_response(self, mock_client):
        """Test prompt optimization with invalid response."""
        mock_client.client.request.return_value = Mock(
            status_code=200, json=lambda: {"invalid": "response"}
        )

        with pytest.raises(OpenRouterAPIError, match="No choices in response"):
            asyncio.run(
                mock_client.optimize_prompt(
                    prompt="Original prompt",
                    model="gpt-4o",
                    system_prompt="System prompt",
                )
            )

    def test_optimize_prompt_invalid_message_format(self, mock_client):
        """Test prompt optimization with invalid message format."""
        mock_client.client.request.return_value = Mock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"invalid": "format"}}]},
        )

        with pytest.raises(OpenRouterAPIError, match="Invalid response format"):
            asyncio.run(
                mock_client.optimize_prompt(
                    prompt="Original prompt",
                    model="gpt-4o",
                    system_prompt="System prompt",
                )
            )

    def test_get_models_success(self, mock_client):
        """Test getting available models."""
        mock_client.client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "gpt-4o", "name": "GPT-4o"},
                    {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
                ]
            },
        )

        result = asyncio.run(mock_client.get_models())

        assert result == {
            "data": [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            ]
        }

    def test_validate_model_success(self, mock_client):
        """Test successful model validation."""
        mock_client.client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "data": [
                    {"id": "gpt-4o", "name": "GPT-4o"},
                    {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
                ]
            },
        )

        result = asyncio.run(mock_client.validate_model("gpt-4o"))

        assert result is True

    def test_validate_model_not_found(self, mock_client):
        """Test model validation when model not found."""
        mock_client.client.request.return_value = Mock(
            status_code=200,
            json=lambda: {
                "data": [{"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"}]
            },
        )

        result = asyncio.run(mock_client.validate_model("gpt-4o"))

        assert result is False

    def test_validate_model_api_error(self, mock_client):
        """Test model validation when API error occurs."""
        mock_client.client.request.side_effect = Exception("API Error")

        result = asyncio.run(mock_client.validate_model("gpt-4o"))

        assert result is False

    def test_context_manager(self, mock_client):
        """Test using OpenRouterClient as context manager."""
        mock_client.client.aclose = AsyncMock()

        async def test_context():
            async with OpenRouterClient(api_key="test-api-key") as client:
                assert client is not None
                return client

        client = asyncio.run(test_context())
        mock_client.client.aclose.assert_called_once()

    def test_close(self, mock_client):
        """Test closing the client."""
        mock_client.client.aclose = AsyncMock()

        asyncio.run(mock_client.close())
        mock_client.client.aclose.assert_called_once()
