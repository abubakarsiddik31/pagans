"""
Core data models and enums for PAGANS.

This package is the canonical source for model family detection and
OpenRouter model-name mappings.
"""

from dataclasses import dataclass
from enum import Enum

from ..constants import (
    FAMILY_ANTHROPIC,
    FAMILY_GOOGLE,
    FAMILY_OPENAI,
    FAMILY_XAI,
)


class Provider(Enum):
    """Enum representing available inference providers for PAGANS."""

    OPENROUTER = "openrouter"


class ModelFamily(Enum):
    """Enum representing supported LLM model families."""

    OPENAI = FAMILY_OPENAI
    ANTHROPIC = FAMILY_ANTHROPIC
    GOOGLE = FAMILY_GOOGLE
    XAI = FAMILY_XAI


@dataclass
class OptimizationResult:
    """Result of a prompt optimization operation."""

    original: str
    optimized: str
    target_model: str
    target_family: ModelFamily
    optimization_notes: str | None = None
    tokens_used: int | None = None
    optimization_time: float | None = None


@dataclass
class OptimizationRequest:
    """Request for prompt optimization."""

    prompt: str
    target_model: str
    optimization_notes: str | None = None


# Short names users can pass to PAGANS.
SHORT_MODEL_NAMES: dict[str, ModelFamily] = {
    # OpenAI (latest + backward-compatible aliases)
    "gpt-5.4": ModelFamily.OPENAI,
    "gpt-5.4-pro": ModelFamily.OPENAI,
    "gpt-5.4-mini": ModelFamily.OPENAI,
    "gpt-5.4-nano": ModelFamily.OPENAI,
    "gpt-5-mini": ModelFamily.OPENAI,
    "gpt-5-nano": ModelFamily.OPENAI,
    "gpt-5": ModelFamily.OPENAI,
    "gpt-4.1": ModelFamily.OPENAI,
    "gpt-4o": ModelFamily.OPENAI,

    # Anthropic (latest + backward-compatible aliases)
    "claude-opus-4.6": ModelFamily.ANTHROPIC,
    "claude-sonnet-4.6": ModelFamily.ANTHROPIC,
    "claude-haiku-4.5": ModelFamily.ANTHROPIC,
    "claude-opus-4.1": ModelFamily.ANTHROPIC,
    "claude-opus-4": ModelFamily.ANTHROPIC,
    "claude-sonnet-4": ModelFamily.ANTHROPIC,
    "claude-3.5-sonnet": ModelFamily.ANTHROPIC,

    # Google Gemini (latest text models + stable line)
    "gemini-3.1-pro-preview": ModelFamily.GOOGLE,
    "gemini-3-flash-preview": ModelFamily.GOOGLE,
    "gemini-3.1-flash-lite-preview": ModelFamily.GOOGLE,
    "gemini-2.5-pro": ModelFamily.GOOGLE,
    "gemini-2.5-flash": ModelFamily.GOOGLE,

    # xAI Grok (text models)
    "grok-4.20-beta": ModelFamily.XAI,
    "grok-4.20-multi-agent-beta": ModelFamily.XAI,
    "grok-4": ModelFamily.XAI,
    "grok-4-fast": ModelFamily.XAI,
    "grok-code-fast-1": ModelFamily.XAI,
}


# OpenRouter routed model ids.
OPENROUTER_MODEL_MAPPINGS: dict[str, str] = {
    # OpenAI
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-pro": "openai/gpt-5.4-pro",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
    "gpt-5-mini": "openai/gpt-5.4-mini",
    "gpt-5-nano": "openai/gpt-5.4-nano",
    "gpt-5": "openai/gpt-5",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4o": "openai/gpt-4o",

    # Anthropic
    "claude-opus-4.6": "anthropic/claude-opus-4.6",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-opus-4.1": "anthropic/claude-opus-4.1",
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",

    # Google
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",

    # xAI
    "grok-4.20-beta": "x-ai/grok-4.20-beta",
    "grok-4.20-multi-agent-beta": "x-ai/grok-4.20-multi-agent-beta",
    "grok-4": "x-ai/grok-4",
    "grok-4-fast": "x-ai/grok-4-fast",
    "grok-code-fast-1": "x-ai/grok-code-fast-1",
}


MODEL_MAPPINGS: dict[ModelFamily, list[str]] = {
    ModelFamily.OPENAI: [
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5",
        "gpt-4.1",
        "gpt-4o",
        "openai/gpt-5.4",
        "openai/gpt-5.4-pro",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
        "openai/gpt-5",
        "openai/gpt-4.1",
        "openai/gpt-4o",
    ],
    ModelFamily.ANTHROPIC: [
        "claude-opus-4.6",
        "claude-sonnet-4.6",
        "claude-haiku-4.5",
        "claude-opus-4.1",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3.5-sonnet",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-opus-4",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
    ],
    ModelFamily.GOOGLE: [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
    ],
    ModelFamily.XAI: [
        "grok-4.20-beta",
        "grok-4.20-multi-agent-beta",
        "grok-4",
        "grok-4-fast",
        "grok-code-fast-1",
        "x-ai/grok-4.20-beta",
        "x-ai/grok-4.20-multi-agent-beta",
        "x-ai/grok-4",
        "x-ai/grok-4-fast",
        "x-ai/grok-code-fast-1",
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
    """Detect the model family from a model name."""
    normalized = model_name.lower().strip()

    for family, models in MODEL_MAPPINGS.items():
        for model in models:
            if model.lower() == normalized:
                return family

    for family, models in MODEL_MAPPINGS.items():
        for model in models:
            lower_model = model.lower()
            if normalized in lower_model or lower_model in normalized:
                return family

    msg = f"Unknown model family for: {normalized}"
    raise ValueError(msg)


def get_supported_models() -> dict[ModelFamily, list[str]]:
    """Get all supported models organized by family."""
    return MODEL_MAPPINGS.copy()


def is_supported_model(model_name: str) -> bool:
    """Check if a model is supported."""
    try:
        detect_model_family(model_name)
        return True
    except ValueError:
        return False


def get_model_family_models(family: ModelFamily) -> list[str]:
    """Get all models for a specific family."""
    return MODEL_MAPPINGS.get(family, [])


def get_family_models(family_name: str) -> list[str]:
    """Get all models for a family by name."""
    try:
        family = ModelFamily(family_name)
        return get_model_family_models(family)
    except ValueError:
        return []


def resolve_model_and_provider(model_name: str) -> tuple[str, ModelFamily]:
    """Resolve a short model name to OpenRouter model id and family."""
    normalized = model_name.strip()

    if normalized in SHORT_MODEL_NAMES:
        model_family = SHORT_MODEL_NAMES[normalized]
        if normalized in OPENROUTER_MODEL_MAPPINGS:
            return OPENROUTER_MODEL_MAPPINGS[normalized], model_family

        msg = f"Model '{normalized}' is not available on OpenRouter"
        raise ValueError(msg)

    try:
        return normalized, detect_model_family(normalized)
    except ValueError as e:
        raise ValueError(f"Unknown model: {normalized}") from e


def get_supported_models_by_provider(provider: Provider | str) -> dict[str, ModelFamily]:
    """Return short model names supported by a provider."""
    if isinstance(provider, str):
        provider = Provider(provider.lower())

    if provider != Provider.OPENROUTER:
        return {}

    return {
        short_name: SHORT_MODEL_NAMES[short_name]
        for short_name in OPENROUTER_MODEL_MAPPINGS
    }


def get_all_supported_providers() -> list[Provider]:
    """Get all providers supported by PAGANS runtime."""
    return [Provider.OPENROUTER]


def get_provider_model_name(short_name: str, provider: Provider) -> str:
    """Resolve a short name to provider-specific model id."""
    if provider != Provider.OPENROUTER:
        msg = f"Provider not supported: {provider.value}"
        raise ValueError(msg)

    if short_name not in OPENROUTER_MODEL_MAPPINGS:
        msg = f"Model '{short_name}' is not available on OpenRouter"
        raise ValueError(msg)

    return OPENROUTER_MODEL_MAPPINGS[short_name]


from .registry import (
    ModelFamilyInfo,
    ProviderInfo,
    ModelRegistry,
    get_model_registry,
    register_model_family,
    register_provider,
    get_model_family,
    is_model_supported,
)


__all__ = [
    "ModelFamily",
    "Provider",
    "OptimizationResult",
    "OptimizationRequest",
    "detect_model_family",
    "get_supported_models",
    "is_supported_model",
    "get_model_family_models",
    "get_family_models",
    "resolve_model_and_provider",
    "get_supported_models_by_provider",
    "get_all_supported_providers",
    "get_provider_model_name",
    "FAMILY_MODEL_MAPPINGS",
    "MODEL_NAME_MAPPINGS",
    "ModelFamilyInfo",
    "ProviderInfo",
    "ModelRegistry",
    "get_model_registry",
    "register_model_family",
    "register_provider",
    "get_model_family",
    "is_model_supported",
]
