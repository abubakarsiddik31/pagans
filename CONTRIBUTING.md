# Contributing to PAGANS

Thanks for your interest in contributing.

## Before You Start

- Search existing issues and pull requests before opening a new one.
- For substantial changes, open an issue first to align on scope and design.
- Keep pull requests focused and reviewable.

## Development Setup

```bash
git clone https://github.com/abubakarsiddik31/pagans.git
cd pagans
uv sync --dev
uv pip install -e .
uv run pre-commit install
```

Set your API key for integration scenarios:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Development Workflow

1. Create a branch from `develop` (or `main` if only `main` is used in your fork).
2. Make your change.
3. Add or update tests.
4. Run local checks.
5. Open a pull request.

## Local Quality Checks

```bash
uv run ruff check .
uv run mypy src/
uv run pytest -q
```

If you touched packaging or docs, also run:

```bash
uv run python -m build
uv run twine check dist/*
```

## Pull Request Requirements

- Clear description of what changed and why
- Tests for behavior changes
- Docs/examples updates when user-facing behavior changes
- No unrelated refactors in the same PR

## Commit Guidance

Use clear, imperative commit messages, for example:

- `add grok family prompt template`
- `fix model family detection for provider-prefixed ids`
- `update notebook quickstart for openrouter`

## Reporting Bugs

Open a GitHub issue and include:

- Expected behavior
- Actual behavior
- Minimal reproduction
- Environment details (OS, Python, package version)
- Relevant logs or traceback

## Conduct

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
