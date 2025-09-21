"""
Custom exceptions for PAGANS.

This module defines the exception hierarchy used throughout the package.
"""

from typing import Any


class PAGANSError(Exception):
    """Base exception class for all PAGANS errors."""

    def __init__(self, message: str, details: Any | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class PAGANSOpenRouterAPIError(PAGANSError):
    """Exception raised when OpenRouter API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: dict | None = None,
    ):
        super().__init__(message, response_data)
        self.status_code = status_code
        self.response_data = response_data


class PAGANSModelNotFoundError(PAGANSError):
    """Exception raised when a model is not recognized or supported."""

    def __init__(self, model_name: str):
        message = f"Model '{model_name}' is not supported or recognized"
        super().__init__(message, model_name)


class PAGANSConfigurationError(PAGANSError):
    """Exception raised when configuration is invalid or missing."""


class PAGANSNetworkError(PAGANSError):
    """Exception raised when network-related errors occur."""


class PAGANSTimeoutError(PAGANSError):
    """Exception raised when a request times out."""

    def __init__(self, timeout: float):
        message = f"Request timed out after {timeout} seconds"
        super().__init__(message, timeout)


class PAGANSRateLimitError(PAGANSError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, retry_after: int | None = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Please wait {retry_after} seconds before trying again."
        super().__init__(message, retry_after)


class PAGANSValidationError(PAGANSError):
    """Exception raised when input validation fails."""


class PAGANSAuthenticationError(PAGANSError):
    """Exception raised when authentication fails."""


class PAGANSQuotaExceededError(PAGANSError):
    """Exception raised when API quota is exceeded."""
