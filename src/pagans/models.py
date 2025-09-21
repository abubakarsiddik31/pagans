"""
Core data models and enums for PAGANS.

This module contains the ModelFamily enum, model name mappings,
and data structures for optimization requests and results.
"""

from dataclasses import dataclass
from enum import Enum

from .constants import (
    FAMILY_ANTHROPIC,
    FAMILY_GOOGLE,
    FAMILY_OPENAI,
)


class ModelFamily(Enum):
    """Enum representing different LLM model families."""

    OPENAI = FAMILY_OPENAI
    ANTHROPIC = FAMILY_ANTHROPIC
    GOOGLE = FAMILY_GOOGLE


class Provider(Enum):
    """Enum representing different LLM providers."""
    
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class OptimizationResult:
    """Result of a prompt optimization operation."""

    original: str  # Original prompt
    optimized: str  # Optimized prompt
    target_model: str  # Target model name
    target_family: ModelFamily  # Model family
    provider: Provider  # Provider used
    optimization_notes: str | None = None  # Optimization notes
    tokens_used: int | None = None  # Tokens used in optimization
    optimization_time: float | None = None  # Time taken in seconds


@dataclass
class OptimizationRequest:
    """Request for prompt optimization."""

    prompt: str  # Original prompt
    target_model: str  # Target model name
    provider: Provider | None = None  # Provider to use
    optimization_notes: str | None = None  # Optimization notes


# Short model names that users can specify
SHORT_MODEL_NAMES = {
    # OpenAI models
    "gpt-4o": ModelFamily.OPENAI,
    "gpt-4.1": ModelFamily.OPENAI,
    "gpt-5": ModelFamily.OPENAI,
    
    # Anthropic models  
    "claude-opus-4": ModelFamily.ANTHROPIC,
    "claude-opus-4.1": ModelFamily.ANTHROPIC,
    "claude-sonnet-4": ModelFamily.ANTHROPIC,
    "claude-sonnet-3.7": ModelFamily.ANTHROPIC,
    
    # Google models
    "gemini-2.5-pro": ModelFamily.GOOGLE,
    "gemini-2.5-flash": ModelFamily.GOOGLE,
}

# Provider-specific model name mappings
PROVIDER_MODEL_MAPPINGS = {
    Provider.OPENROUTER: {
        # OpenAI models on OpenRouter
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini", 
        "gpt-5": "openai/gpt-5",
        
        # Anthropic models on OpenRouter
        "claude-opus-4": "anthropic/claude-opus-4-20250514",
        "claude-opus-4.1": "anthropic/claude-opus-4-1-20250805",
        "claude-sonnet-4": "anthropic/claude-sonnet-4-20250514",
        "claude-sonnet-3.7": "anthropic/claude-3-7-sonnet-20250219",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        
        # Google models on OpenRouter
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
    },
    
    Provider.ANTHROPIC: {
        # Anthropic direct API model names
        "claude-opus-4": "claude-opus-4-20250514",
        "claude-opus-4.1": "claude-opus-4-1-20250805", 
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-sonnet-3.7": "claude-3-7-sonnet-20250219",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    },
    
    Provider.OPENAI: {
        # OpenAI direct API model names
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-5": "gpt-5",
    },
    
    Provider.GOOGLE: {
        # Google direct API model names
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }
}

MODEL_MAPPINGS: dict[ModelFamily, list[str]] = {
    ModelFamily.OPENAI: [
        "gpt-4o", "gpt-4o-mini", "gpt-5",
        "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5",
    ],
    ModelFamily.ANTHROPIC: [
        "claude-opus-4", "claude-opus-4.1", "claude-sonnet-4", "claude-sonnet-3.7", "claude-3.5-sonnet",
        "claude-opus-4-20250514", "claude-opus-4-1-20250805", "claude-sonnet-4-20250514", 
        "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022",
        "anthropic/claude-3.5-sonnet", "anthropic/claude-opus-4-1-20250805",
        "anthropic/claude-opus-4-20250514", "anthropic/claude-sonnet-4-20250514", 
        "anthropic/claude-3-7-sonnet-20250219",
    ],
    ModelFamily.GOOGLE: [
        "gemini-2.5-pro", "gemini-2.5-flash",
        "google/gemini-2.5-pro", "google/gemini-2.5-flash",
    ],
}

MODEL_NAME_MAPPINGS: dict[str, ModelFamily] = {}
for family, models in MODEL_MAPPINGS.items():
    for model in models:
        MODEL_NAME_MAPPINGS[model] = family

FAMILY_MODEL_MAPPINGS: dict[str, list[str]] = {
    family.value: models for family, models in MODEL_MAPPINGS.items()
}


def detect_model_family(model_name: str) -> ModelFamily:
    """
    Detect the model family from a model name.

    Args:
        model_name: The name of the model (e.g., "gpt-4o", "claude-3.5-sonnet")

    Returns:
        The detected ModelFamily

    Raises:
        ValueError: If the model name is not recognized
    """
    model_name = model_name.lower().strip()

    for family, models in MODEL_MAPPINGS.items():
        for model in models:
            if model.lower() == model_name:
                return family

    for family, models in MODEL_MAPPINGS.items():
        for model in models:
            if model_name in model.lower() or model.lower() in model_name:
                return family

    msg = f"Unknown model family for: {model_name}"
    raise ValueError(msg)


def get_supported_models() -> dict[ModelFamily, list[str]]:
    """
    Get all supported models organized by family.

    Returns:
        Dictionary mapping ModelFamily to list of supported model names
    """
    return MODEL_MAPPINGS.copy()


def is_supported_model(model_name: str) -> bool:
    """
    Check if a model is supported.

    Args:
        model_name: The name of the model to check

    Returns:
        True if the model is supported, False otherwise
    """
    try:
        detect_model_family(model_name)
        return True
    except ValueError:
        return False


def get_model_family_models(family: ModelFamily) -> list[str]:
    """
    Get all models for a specific family.

    Args:
        family: The model family

    Returns:
        List of model names for the family
    """
    return MODEL_MAPPINGS.get(family, [])


def get_family_models(family_name: str) -> list[str]:
    """
    Get all models for a family by name.

    Args:
        family_name: The name of the family (e.g., "openai", "anthropic")

    Returns:
        List of model names for the family
    """
    try:
        family = ModelFamily(family_name)
        return get_model_family_models(family)
    except ValueError:
        return []


def resolve_model_and_provider(model_name: str, provider: Provider | str | None = None) -> tuple[str, ModelFamily, Provider]:
    """
    Resolve a short model name and provider to the actual model name, family, and provider.
    
    Args:
        model_name: Short model name (e.g., "claude-sonnet-4", "gpt-4o")
        provider: Provider enum or string (e.g., "anthropic", "openrouter")
        
    Returns:
        Tuple of (actual_model_name, model_family, provider)
        
    Raises:
        ValueError: If model or provider is not supported
    """
    # Convert string provider to enum
    if isinstance(provider, str):
        try:
            provider = Provider(provider.lower())
        except ValueError:
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Default to OpenRouter if no provider specified
    if provider is None:
        provider = Provider.OPENROUTER
    
    # Check if it's a short model name
    if model_name in SHORT_MODEL_NAMES:
        model_family = SHORT_MODEL_NAMES[model_name]
        
        # Get provider-specific model name
        if provider in PROVIDER_MODEL_MAPPINGS and model_name in PROVIDER_MODEL_MAPPINGS[provider]:
            actual_model_name = PROVIDER_MODEL_MAPPINGS[provider][model_name]
            return actual_model_name, model_family, provider
        else:
            raise ValueError(f"Model '{model_name}' is not available on provider '{provider.value}'")
    
    # Fall back to legacy detection for backward compatibility
    try:
        model_family = detect_model_family(model_name)
        return model_name, model_family, provider
    except ValueError:
        raise ValueError(f"Unknown model: {model_name}")


def get_supported_models_by_provider(provider: Provider | str) -> dict[str, ModelFamily]:
    """
    Get all supported short model names for a specific provider.
    
    Args:
        provider: Provider enum or string
        
    Returns:
        Dictionary mapping short model names to model families
    """
    if isinstance(provider, str):
        try:
            provider = Provider(provider.lower())
        except ValueError:
            return {}
    
    if provider not in PROVIDER_MODEL_MAPPINGS:
        return {}
    
    result = {}
    for short_name in PROVIDER_MODEL_MAPPINGS[provider]:
        if short_name in SHORT_MODEL_NAMES:
            result[short_name] = SHORT_MODEL_NAMES[short_name]
    
    return result


def get_all_supported_providers() -> list[Provider]:
    """Get all supported providers."""
    return list(Provider)


def get_provider_model_name(short_name: str, provider: Provider) -> str:
    """
    Get the provider-specific model name for a short model name.
    
    Args:
        short_name: Short model name (e.g., "claude-sonnet-4")
        provider: Provider enum
        
    Returns:
        Provider-specific model name
        
    Raises:
        ValueError: If model is not available on the provider
    """
    if provider not in PROVIDER_MODEL_MAPPINGS:
        raise ValueError(f"Provider '{provider.value}' is not supported")
    
    if short_name not in PROVIDER_MODEL_MAPPINGS[provider]:
        raise ValueError(f"Model '{short_name}' is not available on provider '{provider.value}'")
    
    return PROVIDER_MODEL_MAPPINGS[provider][short_name]
