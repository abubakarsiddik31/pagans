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
OpenAI, Anthropic, Google, and xAI model series through OpenRouter.

## ✨ Features

- 🚀 Fast prompt optimization for production workflows
- 🎯 Family-aware optimization for OpenAI, Anthropic, Google, and xAI models
- 🔍 Automatic model-family detection from short and provider-prefixed names
- ⚡ Async-first API for single, compare, and batch operations
- 🧠 Built-in caching to reduce repeated optimization cost
- 🛠️ CLI support for optimize, compare, and batch workflows
- 🔗 Unified OpenRouter integration path

## Installation

```bash
pip install pagans
```

## Quick Start

```python
import asyncio
from pagans import PAGANSOptimizer


async def main() -> None:
    async with PAGANSOptimizer() as optimizer:
        result = await optimizer.optimize(
            prompt="Write a robust retry policy for external API calls.",
            target_model="openai/gpt-4o",
        )
        print(result.optimized)


asyncio.run(main())
```

Required environment variable:

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Optional environment variables:

```bash
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export PAGANS_OPTIMIZER_MODEL="openai/gpt-4o-mini"
```

## CLI

PAGANS installs a CLI as `pagans`.

```bash
pagans optimize --prompt "Explain quantum computing for beginners" --target-model gpt-4o
```

```bash
pagans compare \
  --prompt "Design an event-driven order system" \
  --models "gpt-4o,claude-sonnet-4,gemini-2.5-pro"
```

```bash
pagans batch --prompts-file ./prompts.txt --target-model gemini-2.5-pro
```

## 🎯 Model Family Series

PAGANS optimizes prompts for these model families and series:

- OpenAI GPT series
- Anthropic Claude series
- Google Gemini series
- xAI Grok series

PAGANS detects the target model family and applies the matching optimization strategy automatically.

## Links

- Source: https://github.com/abubakarsiddik31/pagans
- Issues: https://github.com/abubakarsiddik31/pagans/issues

## License

MIT
