"""
Core data models and enums for Prompt Optimizer.

This module contains the ModelFamily enum, model name mappings,
and data structures for optimization requests and results.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.prompt_optimizer.constants import (
    FAMILY_ANTHROPIC,
    FAMILY_GOOGLE,
    FAMILY_OPENAI,
)


class ModelFamily(Enum):
    """Enum representing different LLM model families."""

    OPENAI = FAMILY_OPENAI
    ANTHROPIC = FAMILY_ANTHROPIC
    GOOGLE = FAMILY_GOOGLE


@dataclass
class OptimizationResult:
    """Result of a prompt optimization operation."""

    original: str  # Original prompt
    optimized: str  # Optimized prompt
    target_model: str  # Target model name
    target_family: ModelFamily  # Model family
    optimization_notes: Optional[str] = None  # Optimization notes
    tokens_used: Optional[int] = None  # Tokens used in optimization
    optimization_time: Optional[float] = None  # Time taken in seconds


@dataclass
class OptimizationRequest:
    """Request for prompt optimization."""

    prompt: str  # Original prompt
    target_model: str  # Target model name
    optimization_notes: Optional[str] = None  # Optimization notes


MODEL_MAPPINGS: Dict[ModelFamily, List[str]] = {
    ModelFamily.OPENAI: [
        "gpt-5",
        "gpt-4.1",
        "gpt-4o",
    ],
    ModelFamily.ANTHROPIC: [
        "claude-4",
        "claude-4.1",
        "claude-3.7-sonnet",
    ],
    ModelFamily.GOOGLE: [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
}

MODEL_NAME_MAPPINGS: Dict[str, ModelFamily] = {}
for family, models in MODEL_MAPPINGS.items():
    for model in models:
        MODEL_NAME_MAPPINGS[model] = family

FAMILY_MODEL_MAPPINGS: Dict[str, List[str]] = {
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

    raise ValueError(f"Unknown model family for: {model_name}")


def get_supported_models() -> Dict[ModelFamily, List[str]]:
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


def get_model_family_models(family: ModelFamily) -> List[str]:
    """
    Get all models for a specific family.

    Args:
        family: The model family

    Returns:
        List of model names for the family
    """
    return MODEL_MAPPINGS.get(family, [])


def get_family_models(family_name: str) -> List[str]:
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
