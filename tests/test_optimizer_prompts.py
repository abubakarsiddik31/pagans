"""
Unit tests for the optimizer prompts module.

This module contains tests for the optimization prompts and prompt manager functionality.
"""

from unittest.mock import Mock

import pytest

from src.pagans.optimizer_prompts import (
    BaseOptimizationPrompt,
    OptimizationPromptManager,
    get_optimization_description,
    get_optimization_prompt,
    get_supported_families,
)
from src.pagans.optimizer_prompts.anthropic import (
    ANTHROPIC_OPTIMIZATION_PROMPT,
)
from src.pagans.optimizer_prompts.google import GOOGLE_OPTIMIZATION_PROMPT
from src.pagans.optimizer_prompts.openai import OPENAI_OPTIMIZATION_PROMPT


class TestBaseOptimizationPrompt:
    """Test cases for the BaseOptimizationPrompt class."""

    def test_base_prompt_abstract_methods(self):
        """Test that BaseOptimizationPrompt is abstract."""
        with pytest.raises(TypeError):
            BaseOptimizationPrompt()

    def test_base_prompt_get_prompt_not_implemented(self):
        """Test that abstract class cannot be instantiated without implementing get_prompt."""

        class TestPrompt(BaseOptimizationPrompt):
            def get_description(self):
                return "Test description"

        with pytest.raises(TypeError, match="Can't instantiate abstract class TestPrompt"):
            TestPrompt()

    def test_base_prompt_get_description_not_implemented(self):
        """Test that abstract class cannot be instantiated without implementing get_description."""

        class TestPrompt(BaseOptimizationPrompt):
            def get_prompt(self, original_prompt, target_model):
                return "Test prompt"

        with pytest.raises(TypeError, match="Can't instantiate abstract class TestPrompt"):
            TestPrompt()


class TestOptimizationPromptManager:
    """Test cases for the OptimizationPromptManager class."""

    def test_init_empty(self):
        """Test initialization with no prompts."""
        manager = OptimizationPromptManager()
        assert len(manager._prompts) == 0

    def test_register_prompt(self):
        """Test registering a prompt."""
        manager = OptimizationPromptManager()
        mock_prompt = Mock()

        manager.register_prompt("test_family", mock_prompt)

        assert "test_family" in manager._prompts
        assert manager._prompts["test_family"] is mock_prompt

    def test_register_prompt_overwrite(self):
        """Test registering a prompt overwrites existing one."""
        manager = OptimizationPromptManager()
        mock_prompt1 = Mock()
        mock_prompt2 = Mock()

        manager.register_prompt("test_family", mock_prompt1)
        manager.register_prompt("test_family", mock_prompt2)

        assert manager._prompts["test_family"] is mock_prompt2

    def test_get_prompt_success(self):
        """Test getting a prompt successfully."""
        manager = OptimizationPromptManager()
        mock_prompt = Mock()
        mock_prompt.get_prompt.return_value = "Test prompt"

        manager.register_prompt("test_family", mock_prompt)

        result = manager.get_prompt("test_family", "original", "target")

        assert result == "Test prompt"
        mock_prompt.get_prompt.assert_called_once_with("original", "target")

    def test_get_prompt_not_found(self):
        """Test getting a prompt that doesn't exist."""
        manager = OptimizationPromptManager()

        with pytest.raises(ValueError, match="No optimization prompt registered for family: test_family"):
            manager.get_prompt("test_family", "original", "target")

    def test_get_description_success(self):
        """Test getting a description successfully."""
        manager = OptimizationPromptManager()
        mock_prompt = Mock()
        mock_prompt.get_description.return_value = "Test description"

        manager.register_prompt("test_family", mock_prompt)

        result = manager.get_description("test_family")

        assert result == "Test description"
        mock_prompt.get_description.assert_called_once()

    def test_get_description_not_found(self):
        """Test getting a description for a family that doesn't exist."""
        manager = OptimizationPromptManager()

        with pytest.raises(ValueError, match="No optimization prompt registered for family: test_family"):
            manager.get_description("test_family")

    def test_get_supported_families_empty(self):
        """Test getting supported families when none registered."""
        manager = OptimizationPromptManager()

        result = manager.get_supported_families()

        assert result == []

    def test_get_supported_families_with_prompts(self):
        """Test getting supported families with registered prompts."""
        manager = OptimizationPromptManager()
        mock_prompt1 = Mock()
        mock_prompt2 = Mock()

        manager.register_prompt("family1", mock_prompt1)
        manager.register_prompt("family2", mock_prompt2)

        result = manager.get_supported_families()

        assert result == ["family1", "family2"]

    def test_get_supported_families_no_duplicates(self):
        """Test that supported families have no duplicates."""
        manager = OptimizationPromptManager()
        mock_prompt = Mock()

        manager.register_prompt("family1", mock_prompt)
        manager.register_prompt("family1", mock_prompt)  # Register same family again

        result = manager.get_supported_families()

        assert result == ["family1"]  # No duplicates


class TestFamilySpecificPrompts:
    """Test cases for family-specific optimization prompts."""

    def test_openai_prompt_structure(self):
        """Test that OpenAI prompt has correct structure."""
        prompt = OPENAI_OPTIMIZATION_PROMPT.get_prompt(
            original_prompt="Test prompt", target_model="gpt-4o"
        )

        assert "OpenAI's GPT models" in prompt
        assert "gpt-4o" in prompt
        assert "Test prompt" in prompt
        assert "Return ONLY the optimized prompt" in prompt

    def test_anthropic_prompt_structure(self):
        """Test that Anthropic prompt has correct structure."""
        prompt = ANTHROPIC_OPTIMIZATION_PROMPT.get_prompt(
            original_prompt="Test prompt", target_model="claude-3.5-sonnet"
        )

        assert "Anthropic's Claude models" in prompt
        assert "claude-3.5-sonnet" in prompt
        assert "Test prompt" in prompt
        assert "Return ONLY the optimized prompt" in prompt

    def test_google_prompt_structure(self):
        """Test that Google prompt has correct structure."""
        prompt = GOOGLE_OPTIMIZATION_PROMPT.get_prompt(
            original_prompt="Test prompt", target_model="gemini-1.5-pro"
        )

        assert "Google's Gemini models" in prompt
        assert "gemini-1.5-pro" in prompt
        assert "Test prompt" in prompt
        assert "Return ONLY the optimized prompt" in prompt

    def test_openai_description(self):
        """Test that OpenAI description is correct."""
        description = OPENAI_OPTIMIZATION_PROMPT.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "OpenAI" in description

    def test_anthropic_description(self):
        """Test that Anthropic description is correct."""
        description = ANTHROPIC_OPTIMIZATION_PROMPT.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "Anthropic" in description

    def test_google_description(self):
        """Test that Google description is correct."""
        description = GOOGLE_OPTIMIZATION_PROMPT.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "Google" in description


class TestGlobalPromptFunctions:
    """Test cases for global prompt functions."""

    def test_get_optimization_prompt_success(self):
        """Test getting optimization prompt successfully."""
        result = get_optimization_prompt("openai", "Test prompt", "gpt-4o")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "OpenAI's GPT models" in result
        assert "Test prompt" in result
        assert "gpt-4o" in result

    def test_get_optimization_prompt_invalid_family(self):
        """Test getting optimization prompt with invalid family."""
        with pytest.raises(ValueError, match="No optimization prompt registered for family: invalid_family"):
            get_optimization_prompt("invalid_family", "Test prompt", "gpt-4o")

    def test_get_optimization_description_success(self):
        """Test getting optimization description successfully."""
        result = get_optimization_description("openai")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "OpenAI" in result

    def test_get_optimization_description_invalid_family(self):
        """Test getting optimization description with invalid family."""
        with pytest.raises(ValueError, match="No optimization prompt registered for family: invalid_family"):
            get_optimization_description("invalid_family")

    def test_get_supported_families(self):
        """Test getting supported families."""
        result = get_supported_families()

        assert isinstance(result, list)
        assert len(result) > 0
        assert "openai" in result
        assert "anthropic" in result
        assert "google" in result


class TestPromptIntegration:
    """Test cases for prompt integration and consistency."""

    def test_all_families_have_prompts(self):
        """Test that all supported families have prompts registered."""
        families = get_supported_families()

        for family in families:
            prompt = get_optimization_prompt(family, "Test prompt", "test-model")
            description = get_optimization_description(family)

            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert isinstance(description, str)
            assert len(description) > 0

    def test_prompt_consistency(self):
        """Test that prompts are consistent across families."""
        test_prompt = "Test prompt"
        test_model = "test-model"

        families = get_supported_families()

        for family in families:
            prompt = get_optimization_prompt(family, test_prompt, test_model)

            # All prompts should contain the original prompt and target model
            assert test_prompt in prompt
            assert test_model in prompt

            # All prompts should instruct to return only the optimized prompt
            assert "Return ONLY the optimized prompt" in prompt

    def test_description_consistency(self):
        """Test that descriptions are consistent across families."""
        families = get_supported_families()

        for family in families:
            description = get_optimization_description(family)

            # All descriptions should be strings with content
            assert isinstance(description, str)
            assert len(description) > 0

            # All descriptions should mention the family name
            assert family in description.lower()

    def test_prompt_manager_global_instance(self):
        """Test that the global prompt manager instance works correctly."""
        # Test that the global functions use the same manager instance
        prompt1 = get_optimization_prompt("openai", "Test prompt", "gpt-4o")
        prompt2 = get_optimization_prompt("openai", "Test prompt", "gpt-4o")

        assert prompt1 == prompt2  # Should be the same result

        # Test that the manager has all families registered
        families = get_supported_families()
        assert len(families) >= 3  # Should have at least 3 families (OpenAI, Anthropic, Google)

    def test_prompt_parameter_substitution(self):
        """Test that prompt parameters are properly substituted."""
        original_prompt = "Original test prompt with {variables}"
        target_model = "gpt-4o"

        prompt = get_optimization_prompt("openai", original_prompt, target_model)

        # The original prompt should be included in the system prompt
        assert original_prompt in prompt
        assert target_model in prompt

        # The system prompt should be properly formatted
        assert "You are an expert at optimizing prompts" in prompt
