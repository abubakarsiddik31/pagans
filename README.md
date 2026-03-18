# PAGANS - Prompt Optimization Framework 😅

<div align="center">
  <img src="logo.png" alt="PAGANS Logo" width="200"/>

[![PyPI version](https://badge.fury.io/py/pagans.svg)](https://pypi.org/project/pagans/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pagans/badge/?version=latest)](https://pagans.readthedocs.io/en/latest/?badge=latest)
[![Build](https://github.com/abubakarsiddik31/pagans/actions/workflows/build.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/build.yml)
[![Test](https://github.com/abubakarsiddik31/pagans/actions/workflows/test.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/test.yml)
[![Lint](https://github.com/abubakarsiddik31/pagans/actions/workflows/lint.yml/badge.svg)](https://github.com/abubakarsiddik31/pagans/actions/workflows/lint.yml)


**PAGANS** (Prompts Aligned to Guidelines and Normalization System) is a powerful prompt optimization framework that works exclusively with OpenRouter to optimize prompts across different LLM model families. By leveraging model-specific optimization strategies, PAGANS ensures your prompts are perfectly aligned with each model family's unique characteristics and best practices.
</div>

## ✨ Features

- 🚀 **Fast Optimization**: Optimize prompts in under 10 seconds
- 🎯 **Model Family Optimization**: Specialized optimization strategies for OpenAI, Anthropic, xAI Grok, and Google model families
- 🔗 **OpenRouter Integration**: Seamless integration with OpenRouter for access to all major model families
- 🔄 **Async Support**: Full asynchronous API for high-throughput optimization
- 📊 **Performance Metrics**: Track optimization time and token usage
- 🔧 **Easy Integration**: Simple API that integrates seamlessly into existing workflows
- 🧠 **Smart Caching**: Built-in caching to avoid redundant optimizations
- ⚡ **Batch Processing**: Optimize multiple prompts concurrently
- 📈 **Model Comparison**: Compare optimization results across different model families
- 🎨 **Family-Aware Prompts**: Automatically adapts prompts to each model family's unique characteristics

## 🎯 Model Family Optimization

**PAGANS** excels at understanding the unique characteristics of different LLM model families and optimizing prompts accordingly:

### **OpenAI Models** (GPT Series)
- Emphasizes clear structure and explicit instructions
- Benefits from chain-of-thought prompting techniques
- Optimized for conversational and creative tasks

### **Anthropic Models** (Claude Series)
- Focuses on safety and helpfulness guidelines
- Works best with detailed context and examples
- Excels at analysis and reasoning tasks

### **Google Models** (Gemini Series)
- Leverages multimodal capabilities when available
- Optimized for factual accuracy and research
- Performs well with structured data and technical content

### **xAI Models** (Grok Series)
- Responds best to explicit goals, constraints, and concrete context
- Strong performance for reasoning-heavy and agentic coding tasks
- Benefits from structured prompt sections and iterative refinement

**The optimization process automatically detects the target model family and applies the most effective strategies for that specific family, ensuring optimal performance across all supported models.**

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
from pagans import PAGANSOptimizer

async def main():
    # Initialize PAGANS optimizer with your OpenRouter API key
    optimizer = PAGANSOptimizer(api_key="your-openrouter-api-key")

    # Your original prompt
    prompt = "Write a Python function to calculate fibonacci numbers"

    # Optimize for a specific model family
    result = await optimizer.optimize(
        prompt=prompt,
        target_model="claude-sonnet-4"  # Short model name
    )

    print(f"Original: {result.original}")
    print(f"Optimized: {result.optimized}")
    print(f"Target Model: {result.target_model}")
    print(f"Model Family: {result.target_family.value}")
    print(f"Optimization Time: {result.optimization_time:.2f}s")

    await optimizer.close()

asyncio.run(main())
```

### How It Works

PAGANS separates the **target model** (what you're optimizing for) from the **optimizer model** (what does the work):

```python
# The optimizer model (from environment) does the actual optimization work
# The target model is what you're optimizing FOR
result = await optimizer.optimize(
    prompt="Explain quantum computing",
    target_model="claude-sonnet-4"  # What you're optimizing FOR
)

# PAGANS automatically detects the model family and applies the right optimization strategy
result2 = await optimizer.optimize(
    prompt="Explain quantum computing",
    target_model="gpt-4o"  # Different model family, different optimization strategy
)
```

### Environment Configuration

```bash
export OPENROUTER_API_KEY="your-api-key-here"
export PAGANS_OPTIMIZER_MODEL="openai/gpt-4o-mini"  # Model that does the optimization work
```

### Using Environment Variables

```python
import asyncio
from pagans import PAGANSOptimizer

async def main():
    # Configuration loaded from environment variables
    optimizer = PAGANSOptimizer()

    result = await optimizer.optimize(
        prompt="Explain quantum computing",
        target_model="claude-3.5-sonnet"  # What you're optimizing FOR
    )

    print(f"Optimized for: {result.target_model}")
    print(f"Optimization: {result.optimized}")
    await optimizer.close()

asyncio.run(main())
```

### Context Manager (Recommended)

```python
import asyncio
from pagans import PAGANSOptimizer

async def main():
    async with PAGANSOptimizer() as optimizer:
        result = await optimizer.optimize(
            prompt="Create a REST API in Node.js",
            target_model="gemini-2.5-pro"
        )
        print(result.optimized)
```

## 🎯 Supported Models

### Short Model Names (Recommended)
Use these convenient short names with PAGANS:

| Family | Short Names | Available via OpenRouter |
|--------|-------------|-------------------------|
| **OpenAI** | `gpt-5.4`, `gpt-5.4-pro`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4o` | ✅ |
| **Anthropic** | `claude-opus-4.6`, `claude-sonnet-4.6`, `claude-haiku-4.5`, `claude-opus-4.1`, `claude-sonnet-4` | ✅ |
| **xAI** | `grok-4.20-beta`, `grok-4`, `grok-4-fast`, `grok-code-fast-1` | ✅ |
| **Google** | `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` | ✅ |

### Full Model Names
You can also use full OpenRouter model names:

| Family | Full Model Names |
|--------|------------------|
| **OpenAI** | `openai/gpt-5.4`, `openai/gpt-5.4-pro`, `openai/gpt-5.4-mini`, `openai/gpt-5.4-nano`, `openai/gpt-4.1`, `openai/gpt-4o` |
| **Anthropic** | `anthropic/claude-opus-4.6`, `anthropic/claude-sonnet-4.6`, `anthropic/claude-haiku-4.5`, `anthropic/claude-opus-4.1` |
| **xAI** | `x-ai/grok-4.20-beta`, `x-ai/grok-4`, `x-ai/grok-4-fast`, `x-ai/grok-code-fast-1` |
| **Google** | `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`, `google/gemini-2.5-pro`, `google/gemini-2.5-flash` |

> **✨ Key Feature**: PAGANS automatically detects the model family and applies the appropriate optimization strategy for optimal results!

## 📚 Examples

See the `examples/` directory for complete working examples:

### Basic PAGANS Example
```bash
# Set your OpenRouter API key (get one at https://openrouter.ai/)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Run the basic example
uv run examples/openai_example.py
```

### Model Family Examples
```bash
# Set your OpenRouter API key (get one at https://openrouter.ai/)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Run examples for different model families
uv run examples/anthropic_example.py  # Claude models
uv run examples/google_example.py    # Gemini models
```

### Advanced Examples
```bash
# Provider factory example (legacy)
uv run examples/provider_factory_example.py
```

### Notebook Example
```bash
# Launch Jupyter and open the quickstart notebook
uv sync --dev
uv run jupyter lab
```

Open: `notebooks/pagans_quickstart.ipynb`

## 🔧 Advanced Usage

### Batch Processing

```python
import asyncio
from pagans import PAGANSOptimizer

async def batch_optimize():
    prompts = [
        "Write a Python function",
        "Explain machine learning",
        "Create a React component"
    ]

    async with PAGANSOptimizer() as optimizer:
        results = await optimizer.optimize_multiple(
            prompts=prompts,
            target_model="gpt-4o",
            max_concurrent=3
        )

        for result in results:
            print(f"Optimized: {result.optimized}")
```

### Compare Across Model Families

```python
import asyncio
from pagans import PAGANSOptimizer

async def compare_models():
    prompt = "Write a function to reverse a string"

    async with PAGANSOptimizer() as optimizer:
        results = await optimizer.compare_optimizations(
            prompt=prompt,
            target_models=[
                "gpt-4o",           # OpenAI family
                "claude-3.5-sonnet", # Anthropic family
                "gemini-2.5-pro"     # Google family
            ]
        )

        for model, result in results.items():
            print(f"{model}: {result.optimized[:50]}...")
```

### Custom Configuration

```python
import asyncio
from pagans import PAGANSOptimizer

async def custom_config():
    optimizer = PAGANSOptimizer(
        api_key="your-api-key",
        base_url="https://custom.openrouter.url",
        timeout=30.0,
        max_retries=5,
        retry_delay=2.0
    )

    result = await optimizer.optimize(
        prompt="Your prompt here",
        target_model="gpt-4o"
    )

    await optimizer.close()
```

## 🏗️ API Reference

### PAGANSOptimizer

#### `__init__(api_key=None, base_url=None, optimizer_model=None, timeout=30.0, max_retries=3, retry_delay=1.0)`

Initialize the PAGANSOptimizer.

**Parameters:**
- `api_key` (str, optional): OpenRouter API key
- `base_url` (str, optional): OpenRouter base URL
- `optimizer_model` (str, optional): Model that does the optimization work
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
- `provider` (str, optional): Provider used (always OpenRouter)
- `optimization_notes` (str, optional): Optimization notes
- `tokens_used` (int, optional): Tokens used in optimization
- `optimization_time` (float, optional): Time taken in seconds

## ⚙️ Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `PAGANS_OPTIMIZER_MODEL`: The model that does the optimization work (optional)
- `OPENROUTER_BASE_URL`: Custom OpenRouter base URL (optional)

### .env File Support

Create a `.env` file in your project root:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PAGANS_OPTIMIZER_MODEL=openai/gpt-4o-mini
```

**Important Environment Variables:**
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `PAGANS_OPTIMIZER_MODEL`: The model that does the optimization work (optional, defaults to `openai/gpt-4o-mini`)
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
1. **🤖 Extended Model Families** - Support for Meta Llama, Mistral, Cohere, and Perplexity models via OpenRouter
2. **⚡ Advanced Caching** - Redis-based distributed caching with semantic similarity matching
3. **🎨 Custom Optimization Strategies** - Allow users to define their own optimization approaches

### 🎯 **Planned Features**
- **🔄 A/B Testing Framework** - Compare optimization strategies and automatically select the best approach
- **📊 Performance Analytics** - Track optimization success rates, cost savings, and performance metrics
- **🛠️ CLI Tool** - Command-line interface for batch processing and automation
- **🌐 Web Dashboard** - Simple web UI for testing and managing optimizations
- **🔌 Framework Integrations** - Native support for LangChain, LlamaIndex, and Haystack
- **📱 IDE Extensions** - VS Code extension for in-editor prompt optimization

Want to contribute to any of these features? Check out our [agents.md](agents.md) for developer guidelines!

## 🤝 Contributing

Contributions are welcome.

- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Code of Conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Support: [`SUPPORT.md`](SUPPORT.md)
- Developer architecture notes: [`agents.md`](agents.md)

Quick flow:

1. Fork and clone the repository.
2. Create a feature branch.
3. Run tests and lint locally.
4. Open a pull request with context and tests.

## 📦 Publishing

PyPI publishing guide is available at:

- [`docs/PYPI_PUBLISHING.md`](docs/PYPI_PUBLISHING.md)

Automated publishing workflow:

- [`.github/workflows/publish.yml`](.github/workflows/publish.yml)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
