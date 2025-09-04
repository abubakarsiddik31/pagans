# Prompt Optimizer

[![PyPI version](https://badge.fury.io/py/prompt-optimizer.svg)](https://pypi.org/project/prompt-optimizer/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/prompt-optimizer/badge/?version=latest)](https://prompt-optimizer.readthedocs.io/en/latest/?badge=latest)

A Python package for optimizing prompts across different LLM model families using OpenRouter API. This tool helps developers improve their prompts for specific model architectures by leveraging family-specific optimization techniques.

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
pip install prompt-optimizer
```

### From Source

```bash
git clone https://github.com/abubakarsiddik31/prompt-optimizer.git
cd prompt-optimizer
pip install .
```

### Development Installation

```bash
git clone https://github.com/abubakarsiddik31/prompt-optimizer.git
cd prompt-optimizer
pip install -e ".[dev]"
```

## 🔧 Quick Start

### Basic Usage

```python
import asyncio
from prompt_optimizer import PromptOptimizer

async def main():
    # Initialize optimizer with your OpenRouter API key
    optimizer = PromptOptimizer(api_key="your-openrouter-api-key")

    # Your original prompt
    prompt = "Write a Python function to calculate fibonacci numbers"

    # Optimize for a specific model
    result = await optimizer.optimize(
        prompt=prompt,
        target_model="openai/gpt-4o"
    )

    print(f"Original: {result.original}")
    print(f"Optimized: {result.optimized}")
    print(f"Family: {result.target_family.value}")
    print(f"Time: {result.optimization_time:.2f}s")

    await optimizer.close()

asyncio.run(main())
```

### Using Environment Variables

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

```python
import asyncio
from prompt_optimizer import PromptOptimizer

async def main():
    # API key will be loaded from environment variable
    optimizer = PromptOptimizer()

    result = await optimizer.optimize(
        prompt="Explain quantum computing",
        target_model="anthropic/claude-3.5-sonnet"
    )

    print(result.optimized)
    await optimizer.close()

asyncio.run(main())
```

### Context Manager (Recommended)

```python
import asyncio
from prompt_optimizer import PromptOptimizer

async def main():
    async with PromptOptimizer() as optimizer:
        result = await optimizer.optimize(
            prompt="Create a REST API in Node.js",
            target_model="google/gemini-2.5-pro"
        )
        print(result.optimized)
```

## 🎯 Supported Models

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | `gpt-5`, `gpt-4.1`, `gpt-4o` | ✅ Fully Supported |
| **Anthropic** | `claude-4`, `claude-4.1`, `claude-3.5-sonnet` | ✅ Fully Supported |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash` | ✅ Fully Supported |

## 📚 Examples

See the `examples/` directory for complete working examples:

### OpenAI GPT Example
```python
# examples/openai_example.py
python examples/openai_example.py
```

### Anthropic Claude Example
```python
# examples/anthropic_example.py
python examples/anthropic_example.py
```

### Google Gemini Example
```python
# examples/google_example.py
python examples/google_example.py
```

## 🔧 Advanced Usage

### Batch Processing

```python
import asyncio
from prompt_optimizer import PromptOptimizer

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
from prompt_optimizer import PromptOptimizer

async def compare_models():
    prompt = "Write a function to reverse a string"

    async with PromptOptimizer() as optimizer:
        results = await optimizer.compare_optimizations(
            prompt=prompt,
            target_models=[
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.5-pro"
            ]
        )

        for model, result in results.items():
            print(f"{model}: {result.optimized[:50]}...")
```

### Custom Configuration

```python
import asyncio
from prompt_optimizer import PromptOptimizer

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
- `OPENROUTER_BASE_URL`: Custom OpenRouter base URL (optional)
- `DEFAULT_OPTIMIZER_MODEL`: Default model for optimization (optional)

### .env File Support

Create a `.env` file in your project root:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_OPTIMIZER_MODEL=anthropic/claude-3.5-sonnet
```

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=prompt_optimizer --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


