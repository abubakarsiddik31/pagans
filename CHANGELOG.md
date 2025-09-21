# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-21

### Added
- Initial release of PAGANS (Prompts Aligned to Guidelines and Normalization System)
- Support for OpenAI GPT, Anthropic Claude, and Google Gemini model optimization
- Core prompt optimization engine with provider-specific templates
- Command-line interface for prompt optimization
- Comprehensive test suite with integration tests
- Professional package structure with proper separation of concerns

### Features
- Multi-provider prompt optimization (OpenAI, Anthropic, Google)
- Template-based prompt generation using Jinja2
- HTTP client integration with httpx
- Pydantic models for type safety
- Environment-based configuration
- Rich error handling and custom exceptions

### Documentation
- Comprehensive README with usage examples
- API documentation and examples for each provider
- Development setup and contribution guidelines

### Development
- Modern Python 3.12+ project structure
- Comprehensive testing with pytest
- Code quality tools (black, isort, ruff, mypy)
- Pre-commit hooks for code quality
- CI/CD ready configuration

[0.1.0]: https://github.com/abubakarsiddik31/pagans/releases/tag/v0.1.0