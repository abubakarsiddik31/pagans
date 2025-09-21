"""
Constants and configuration values for PAGANS.

This module contains all the default values, environment variable names,
and other constants used throughout the package.
"""

# Environment variable names
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
ENV_OPENROUTER_BASE_URL = "OPENROUTER_BASE_URL"
ENV_PAGANS_OPTIMIZER_MODEL = "PAGANS_OPTIMIZER_MODEL"
ENV_OPTIMIZER_MODEL = "OPTIMIZER_MODEL"  # Backward compatibility

# Default configuration values
DEFAULT_PAGANS_OPTIMIZER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPTIMIZER_MODEL = "openai/gpt-4o-mini"  # Backward compatibility
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds

# API configuration
API_VERSION = "v1"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"

# HTTP headers
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Model family names
FAMILY_OPENAI = "openai"
FAMILY_ANTHROPIC = "anthropic"
FAMILY_GOOGLE = "google"

# Error messages
ERROR_MESSAGES = {
    "missing_api_key": "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.",
    "invalid_model": "Invalid model name: {model_name}",
    "network_error": "Network error occurred while optimizing prompt",
    "api_error": "OpenRouter API error: {error}",
    "timeout_error": "Request timed out after {timeout} seconds",
    "rate_limit_error": "Rate limit exceeded. Please wait before trying again.",
    "unknown_error": "Unknown error occurred: {error}",
}

# Performance targets
PAGANS_TARGET_OPTIMIZATION_TIME = 10.0  # seconds
PAGANS_MAX_TOKENS = 8000  # Maximum tokens for optimization prompts
