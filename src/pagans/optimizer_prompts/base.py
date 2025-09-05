"""
Base optimization prompt template and utilities.

This module provides the foundation for family-specific optimization prompts
using Jinja2 templating for centralized prompt management.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


class TemplateManager:
    """Manages Jinja2 templates for prompt optimization."""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def get_template(self, template_name: str) -> Template:
        """Get a Jinja template by name."""
        return self.env.get_template(f"{template_name}.jinja")

    def render_template(self, template_name: str, **kwargs) -> str:
        """Render a template with the given context variables."""
        template = self.get_template(template_name)
        return template.render(**kwargs)


# Global template manager instance
_template_manager = TemplateManager()


class BaseOptimizationPrompt(ABC):
    """Abstract base class for optimization prompts using Jinja templates."""

    def __init__(self, template_name: str):
        self.template_name = template_name
        self.template_manager = _template_manager

    def get_prompt(self, original_prompt: str, target_model: str) -> str:
        """
        Get the optimization prompt for a specific model family.

        Args:
            original_prompt: The original prompt to optimize
            target_model: The target model name

        Returns:
            The optimization prompt rendered from Jinja template
        """
        return self.template_manager.render_template(
            self.template_name,
            original_prompt=original_prompt,
            target_model=target_model,
            model_family=self.get_model_family(),
        )

    @abstractmethod
    def get_model_family(self) -> str:
        """Get the model family name for this optimization prompt."""

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this optimization approach.

        Returns:
            Description of the optimization strategy
        """


class OptimizationPromptManager:
    """Manager for optimization prompts across different model families."""

    def __init__(self):
        self._prompts: dict[str, BaseOptimizationPrompt] = {}

    def register_prompt(self, family: str, prompt: BaseOptimizationPrompt) -> None:
        """Register an optimization prompt for a model family."""
        self._prompts[family] = prompt

    def get_prompt(self, family: str, original_prompt: str, target_model: str) -> str:
        """Get the optimization prompt for a specific family."""
        if family not in self._prompts:
            raise ValueError(f"No optimization prompt registered for family: {family}")

        return self._prompts[family].get_prompt(original_prompt, target_model)

    def get_description(self, family: str) -> str:
        """Get the description for a specific family."""
        if family not in self._prompts:
            raise ValueError(f"No optimization prompt registered for family: {family}")

        return self._prompts[family].get_description()

    def get_supported_families(self) -> list:
        """Get list of supported model families."""
        return list(self._prompts.keys())
