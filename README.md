# PAGANS - Prompt Optimization Framework

<div align="center">
  <img src="logo.png" alt="PAGANS Logo" width="180" />
</div>

[![PyPI version](https://badge.fury.io/py/pagans.svg)](https://pypi.org/project/pagans/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/abubakarsiddik31/pagans/actions/workflows/build.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/build.yml)
[![Test](https://github.com/abubakarsiddik31/pagans/actions/workflows/test.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/test.yml)
[![Lint](https://github.com/abubakarsiddik31/pagans/actions/workflows/lint.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/lint.yml)

PAGANS (Prompts Aligned to Guidelines and Normalization System) is a prompt
optimization framework for LLM applications. It adapts prompts to the target
model family automatically so the same prompt can perform better across
OpenAI, Anthropic, Google, and xAI model series.

## Features

- Fast prompt optimization for production workflows
- Family-aware optimization for OpenAI, Anthropic, Google, and xAI models
- Automatic model-family detection from short and provider-prefixed names
- Async-first API for single, compare, and batch operations
- Built-in caching to reduce repeated optimization cost
- CLI support for optimize, compare, and batch workflows
- Optimizer provider support for OpenRouter, OpenAI, Google AI Studio, Anthropic, and Z.ai

## Installation

```bash
pip install pagans
```

## Quick Start

```python
import asyncio

from pagans import PAGANSOptimizer, Provider


async def main() -> None:
    async with PAGANSOptimizer(provider=Provider.OPENROUTER) as optimizer:
        result = await optimizer.optimize(
            prompt="Write a robust retry policy for external API calls.",
            target_model="openai/gpt-4o",
        )
        print(result.optimized)


asyncio.run(main())
```

## Provider Configuration

Set one provider key (or pass `api_key` and `base_url` explicitly).

```bash
# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Google AI Studio (Gemini OpenAI-compatible endpoint)
export GOOGLE_API_KEY="your-gemini-api-key"
export GOOGLE_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_BASE_URL="https://api.anthropic.com/v1"

# Z.ai
export ZAI_API_KEY="your-zai-api-key"
export ZAI_BASE_URL="https://api.z.ai/api/paas/v4"

# Optional global override for optimizer model
export PAGANS_OPTIMIZER_MODEL="openai/gpt-4o-mini"
```

### Python Provider Examples

```python
from pagans import PAGANSOptimizer, Provider

# OpenAI as optimizer provider
optimizer = PAGANSOptimizer(provider=Provider.OPENAI)

# Google AI Studio as optimizer provider
optimizer = PAGANSOptimizer(provider=Provider.GOOGLE)

# Anthropic as optimizer provider
optimizer = PAGANSOptimizer(provider=Provider.ANTHROPIC)

# Z.ai as optimizer provider
optimizer = PAGANSOptimizer(provider=Provider.ZAI)
```

## CLI

PAGANS installs a CLI as `pagans`.

```bash
pagans --provider openrouter optimize \
  --prompt "Explain quantum computing for beginners" \
  --target-model gpt-4o
```

```bash
pagans --provider openai compare \
  --prompt "Design an event-driven order system" \
  --models "gpt-4o,claude-sonnet-4,gemini-2.5-pro"
```

```bash
pagans --provider anthropic batch \
  --prompts-file ./prompts.txt \
  --target-model gemini-2.5-pro
```

## Model Family Series

PAGANS optimizes prompts for these model families and series:

- OpenAI GPT series
- Anthropic Claude series
- Google Gemini series
- xAI Grok series

PAGANS detects the target model family and applies the matching optimization strategy automatically.

## Notebooks

Provider notebook examples are available in [`notebooks/`](./notebooks):

- `pagans_quickstart.ipynb`
- `pagans_openrouter_optimizer.ipynb`
- `pagans_openai_optimizer.ipynb`
- `pagans_google_optimizer.ipynb`
- `pagans_anthropic_optimizer.ipynb`
- `pagans_zai_optimizer.ipynb`

## Links

- Source: https://github.com/abubakarsiddik31/pagans
- Issues: https://github.com/abubakarsiddik31/pagans/issues

## License

MIT
