"""
Custom exceptions for PAGANS.

This module defines the exception hierarchy used throughout the package.
"""

from typing import Any


class PromptOptimizerError(Exception):
    """Base exception class for all Prompt Optimizer errors."""

    def __init__(self, message: str, details: Any | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class OpenRouterAPIError(PromptOptimizerError):
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


class ModelNotFoundError(PromptOptimizerError):
    """Exception raised when a model is not recognized or supported."""

    def __init__(self, model_name: str):
        message = f"Model '{model_name}' is not supported or recognized"
        super().__init__(message, model_name)


class ConfigurationError(PromptOptimizerError):
    """Exception raised when configuration is invalid or missing."""


class NetworkError(PromptOptimizerError):
    """Exception raised when network-related errors occur."""


class TimeoutError(PromptOptimizerError):
    """Exception raised when a request times out."""

    def __init__(self, timeout: float):
        message = f"Request timed out after {timeout} seconds"
        super().__init__(message, timeout)


class RateLimitError(PromptOptimizerError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, retry_after: int | None = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Please wait {retry_after} seconds before trying again."
        super().__init__(message, retry_after)


class ValidationError(PromptOptimizerError):
    """Exception raised when input validation fails."""


class AuthenticationError(PromptOptimizerError):
    """Exception raised when authentication fails."""


class QuotaExceededError(PromptOptimizerError):
    """Exception raised when API quota is exceeded."""
