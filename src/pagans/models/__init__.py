"""
Core data models and enums for PAGANS.

This module contains the ModelFamily enum, model name mappings,
and data structures for optimization requests and results.
"""

# Import from the models.py file directly to avoid circular imports
# Use a try/except to handle the circular import issue
try:
    from ..models import (
        ModelFamily,
        OptimizationResult,
        OptimizationRequest,
        detect_model_family,
        get_supported_models,
        is_supported_model,
        get_model_family_models,
        get_family_models,
        resolve_model_and_provider,
    )
except ImportError:
    # Fallback for circular import issues - define minimal classes locally
    from enum import Enum
    from dataclasses import dataclass

    class ModelFamily(Enum):
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        GOOGLE = "google"

    class Provider(Enum):
        OPENROUTER = "openrouter"
        ANTHROPIC = "anthropic"
        OPENAI = "openai"
        GOOGLE = "google"

    @dataclass
    class OptimizationResult:
        original: str
        optimized: str
        target_model: str
        target_family: ModelFamily
        provider: Provider
        optimization_notes: str | None = None
        tokens_used: int | None = None
        optimization_time: float | None = None

    @dataclass
    class OptimizationRequest:
        prompt: str
        target_model: str
        provider: Provider | None = None
        optimization_notes: str | None = None

    def detect_model_family(model_name: str) -> ModelFamily:
        model_name = model_name.lower()
        if "gpt" in model_name or "openai" in model_name:
            return ModelFamily.OPENAI
        elif "claude" in model_name or "anthropic" in model_name:
            return ModelFamily.ANTHROPIC
        elif "gemini" in model_name or "google" in model_name:
            return ModelFamily.GOOGLE
        else:
            raise ValueError(f"Unknown model family for: {model_name}")

    def get_supported_models() -> dict[ModelFamily, list[str]]:
        return {
            ModelFamily.OPENAI: ["gpt-4o", "gpt-4o-mini"],
            ModelFamily.ANTHROPIC: ["claude-3.5-sonnet", "claude-opus-4"],
            ModelFamily.GOOGLE: ["gemini-2.5-pro", "gemini-2.5-flash"],
        }

    def is_supported_model(model_name: str) -> bool:
        try:
            detect_model_family(model_name)
            return True
        except ValueError:
            return False

    def get_model_family_models(family: ModelFamily) -> list[str]:
        models = get_supported_models()
        return models.get(family, [])

    def get_family_models(family_name: str) -> list[str]:
        try:
            family = ModelFamily(family_name)
            return get_model_family_models(family)
        except ValueError:
            return []

    def resolve_model_and_provider(model_name: str, provider: Provider | str | None = None) -> tuple[str, ModelFamily, Provider]:
        if isinstance(provider, str):
            provider = Provider(provider.lower())
        if provider is None:
            provider = Provider.OPENROUTER

        model_family = detect_model_family(model_name)
        return model_name, model_family, provider

    def get_supported_models_by_provider(provider: Provider | str) -> dict[str, ModelFamily]:
        if isinstance(provider, str):
            provider = Provider(provider.lower())
        return {}

    def get_all_supported_providers() -> list[Provider]:
        return list(Provider)

    def get_provider_model_name(short_name: str, provider: Provider) -> str:
        return short_name
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
    # Original models
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
    # Registry components
    "ModelFamilyInfo",
    "ProviderInfo",
    "ModelRegistry",
    "get_model_registry",
    "register_model_family",
    "register_provider",
    "get_model_family",
    "is_model_supported",
]