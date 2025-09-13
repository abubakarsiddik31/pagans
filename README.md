# PAGANS - Prompts Aligned to Guidelines and Normalization System 😅

<div align="center">
  <img src="logo.png" alt="PAGANS Logo" width="200"/>
</div>

[![PyPI version](https://badge.fury.io/py/pagans.svg)](https://pypi.org/project/pagans/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pagans/badge/?version=latest)](https://pagans.readthedocs.io/en/latest/?badge=latest)
[![Build](https://github.com/abubakarsiddik31/pagans/actions/workflows/build.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/build.yml)
[![Test](https://github.com/abubakarsiddik31/pagans/actions/workflows/test.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/test.yml)
[![Lint](https://github.com/abubakarsiddik31/pagans/actions/workflows/lint.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/lint.yml)


A comprehensive Python package for optimizing prompts across different LLM model families using OpenRouter API. PAGANS helps developers align their prompts with model-specific guidelines and normalization standards, ensuring optimal performance across OpenAI GPT, Anthropic Claude, and Google Gemini models.

## ✨ Features

- 🚀 **Fast Optimization**: Optimize prompts in under 10 seconds
- 🎯 **Multi-Provider Support**: OpenAI GPT, Anthropic Claude, Google Gemini
- 🔄 **Async Support**: Full asynchronous API for high-throughput optimization
- 📊 **Performance Metrics**: Track optimization time and token usage
- 🔧 **Easy Integration**: Simple API that integrates seamlessly into existing workflows
- 🧠 **Smart Caching**: Built-in caching to avoid redundant optimizations
- ⚡ **Batch Processing**: Optimize multiple prompts concurrently

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install pagans
```

### From Source

```bash
git clone https://github.com/abubakarsiddik31/pagans.git
cd pagans
uv sync
```

### Development Installation

```bash
git clone https://github.com/abubakarsiddik31/pagans.git
cd pagans
uv sync --dev
```

## 🔧 Quick Start

### Basic Usage

```python
import asyncio
from pagans import PromptOptimizer

async def main():
    # Initialize optimizer with your OpenRouter API key
    optimizer = PromptOptimizer(api_key="your-openrouter-api-key")

    # Your original prompt
    prompt = "Write a Python function to calculate fibonacci numbers"

    # Optimize for a specific model and provider
    result = await optimizer.optimize(
        prompt=prompt,
        target_model="claude-sonnet-4",  # Short model name
        provider="openrouter"            # Provider selection
    )

    print(f"Original: {result.original}")
    print(f"Optimized: {result.optimized}")
    print(f"Model: {result.target_model}")
    print(f"Provider: {result.provider.value}")
    print(f"Family: {result.target_family.value}")
    print(f"Time: {result.optimization_time:.2f}s")

    await optimizer.close()

asyncio.run(main())
```

### How It Works

PAGANS separates the **target model** (what you're optimizing for) from the **optimizer model** (what does the work):

```python
# The optimizer model (from .env) does the actual optimization work
# The target model is what you're optimizing FOR
result = await optimizer.optimize(
    prompt="Explain quantum computing",
    target_model="claude-sonnet-4",  # What you're optimizing FOR
    provider="openrouter"            # How to access the target model
)

# Same optimization strategy for Claude whether via Anthropic or OpenRouter
result2 = await optimizer.optimize(
    prompt="Explain quantum computing", 
    target_model="claude-sonnet-4",  # Same target
    provider="anthropic"             # Different provider, same optimization
)
```

### Environment Configuration

```bash
export OPENROUTER_API_KEY="your-api-key-here"
export OPTIMIZER_MODEL="openai/gpt-4o-mini"  # Model that does the optimization work
```

### Using Environment Variables

```bash
export OPENROUTER_API_KEY="your-api-key-here"
export OPTIMIZER_MODEL="anthropic/claude-3.5-sonnet"  # Optional: Model that does optimization work
```

```python
import asyncio
from pagans import PromptOptimizer

async def main():
    # Configuration loaded from environment variables
    optimizer = PromptOptimizer()

    result = await optimizer.optimize(
        prompt="Explain quantum computing",
        target_model="claude-3.5-sonnet",  # What you're optimizing FOR
        provider="openrouter"              # How to access target model
    )

    print(f"Optimized for: {result.target_model}")
    print(f"Optimization: {result.optimized}")
    await optimizer.close()

asyncio.run(main())
```

### Context Manager (Recommended)

```python
import asyncio
from pagans import PromptOptimizer

async def main():
    async with PromptOptimizer() as optimizer:
        result = await optimizer.optimize(
            prompt="Create a REST API in Node.js",
            target_model="google/gemini-2.5-pro"
        )
        print(result.optimized)
```

## 🎯 Supported Models

### Short Model Names (Recommended)
Use these convenient short names with any supported provider:

| Family | Short Names | Providers |
|--------|-------------|-----------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-5` | `openrouter`, `openai` |
| **Anthropic** | `claude-opus-4`, `claude-opus-4.1`, `claude-sonnet-4`, `claude-sonnet-3.7`, `claude-3.5-sonnet` | `openrouter`, `anthropic` |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash` | `openrouter`, `google` |

### Provider-Specific Model Names
| Provider | Example Usage |
|----------|---------------|
| **OpenRouter** | `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, `google/gemini-2.5-pro` |
| **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-sonnet-4-20250514` |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash` |

> **✨ Key Feature**: Separate target model (what you optimize FOR) from optimizer model (what does the work). Same optimization strategy regardless of provider!

## 📚 Examples

See the `examples/` directory for complete working examples:

### OpenAI GPT Example
```bash
# Set your OpenRouter API key (get one at https://openrouter.ai/)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Run the example
uv run examples/openai_example.py
```

### Anthropic Claude Example
```bash
# Set your OpenRouter API key (get one at https://openrouter.ai/)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Run the example
uv run examples/anthropic_example.py
```

### Google Gemini Example
```bash
# Set your OpenRouter API key (get one at https://openrouter.ai/)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Run the example
uv run examples/google_example.py
```

### Simple Example
```bash
# Simple example showing the updated approach
uv run examples/simple_example.py
```

## 🔧 Advanced Usage

### Batch Processing

```python
import asyncio
from pagans import PromptOptimizer

async def batch_optimize():
    prompts = [
        "Write a Python function",
        "Explain machine learning",
        "Create a React component"
    ]

    async with PromptOptimizer() as optimizer:
        results = await optimizer.optimize_multiple(
            prompts=prompts,
            target_model="openai/gpt-4o",
            max_concurrent=3
        )

        for result in results:
            print(f"Optimized: {result.optimized}")
```

### Compare Across Models

```python
import asyncio
from pagans import PromptOptimizer

async def compare_models():
    prompt = "Write a function to reverse a string"

    async with PromptOptimizer() as optimizer:
        results = await optimizer.compare_optimizations(
            prompt=prompt,
            target_models=[
                ("gpt-4o", "openrouter"),
                ("claude-3.5-sonnet", "openrouter"), 
                ("gemini-2.5-pro", "openrouter")
            ]
        )

        for model, result in results.items():
            print(f"{model}: {result.optimized[:50]}...")
```

### Custom Configuration

```python
import asyncio
from pagans import PromptOptimizer

async def custom_config():
    optimizer = PromptOptimizer(
        api_key="your-api-key",
        base_url="https://custom.openrouter.url",
        timeout=30.0,
        max_retries=5,
        retry_delay=2.0
    )

    result = await optimizer.optimize(
        prompt="Your prompt here",
        target_model="openai/gpt-4o"
    )

    await optimizer.close()
```

## 🏗️ API Reference

### PromptOptimizer

#### `__init__(api_key=None, base_url=None, default_model=None, timeout=30.0, max_retries=3, retry_delay=1.0)`

Initialize the PromptOptimizer.

**Parameters:**
- `api_key` (str, optional): OpenRouter API key
- `base_url` (str, optional): OpenRouter base URL
- `default_model` (str, optional): Default model for optimization
- `timeout` (float): Request timeout in seconds
- `max_retries` (int): Maximum number of retry attempts
- `retry_delay` (float): Delay between retry attempts

#### `async optimize(prompt, target_model=None, optimization_notes=None, use_cache=True)`

Optimize a prompt for a specific target model.

**Parameters:**
- `prompt` (str): The original prompt to optimize
- `target_model` (str, optional): Target model name
- `optimization_notes` (str, optional): Additional optimization notes
- `use_cache` (bool): Whether to use caching

**Returns:** `OptimizationResult`

#### `async optimize_multiple(prompts, target_model=None, optimization_notes=None, use_cache=True, max_concurrent=3)`

Optimize multiple prompts concurrently.

**Parameters:**
- `prompts` (list[str]): List of prompts to optimize
- `target_model` (str, optional): Target model name
- `optimization_notes` (str, optional): Additional optimization notes
- `use_cache` (bool): Whether to use caching
- `max_concurrent` (int): Maximum concurrent optimizations

**Returns:** `list[OptimizationResult]`

### OptimizationResult

**Attributes:**
- `original` (str): Original prompt
- `optimized` (str): Optimized prompt
- `target_model` (str): Target model name
- `target_family` (ModelFamily): Model family enum
- `optimization_notes` (str, optional): Optimization notes
- `tokens_used` (int, optional): Tokens used in optimization
- `optimization_time` (float, optional): Time taken in seconds

## ⚙️ Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `OPTIMIZER_MODEL`: The model that does the optimization work (optional)
- `OPENROUTER_BASE_URL`: Custom OpenRouter base URL (optional)

### .env File Support

Create a `.env` file in your project root:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPTIMIZER_MODEL=openai/gpt-4o-mini
```

**Important Environment Variables:**
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `OPTIMIZER_MODEL`: The model that does the optimization work (optional, defaults to `openai/gpt-4o-mini`)
- `OPENROUTER_BASE_URL`: Custom OpenRouter base URL (optional)

## 🧪 Testing

This project uses `uv` for dependency management and running commands.

Run the test suite:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=pagans --cov-report=html
```

## 🚀 What's Coming Next

We're actively working on exciting new features! Here's what's on our roadmap:

### 🔥 **Coming Soon**
1. **🔑 Multi-Provider API Support** - Direct integration with OpenAI, Anthropic, Google, and Groq APIs (no more OpenRouter dependency!)
2. **🤖 Extended Model Families** - Support for Meta Llama, Mistral, Cohere, and Perplexity models
3. **⚡ Advanced Caching** - Redis-based distributed caching with semantic similarity matching

### 🎯 **Planned Features**
- **🔄 A/B Testing Framework** - Compare optimization strategies and automatically select the best approach
- **📊 Performance Analytics** - Track optimization success rates, cost savings, and performance metrics  
- **🛠️ CLI Tool** - Command-line interface for batch processing and automation
- **🌐 Web Dashboard** - Simple web UI for testing and managing optimizations
- **🔌 Framework Integrations** - Native support for LangChain, LlamaIndex, and Haystack
- **📱 IDE Extensions** - VS Code extension for in-editor prompt optimization

Want to contribute to any of these features? Check out our [agents.md](agents.md) for developer guidelines!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For detailed development guidelines, see [agents.md](agents.md).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


