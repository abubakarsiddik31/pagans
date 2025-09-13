# PAGANS Development Guide

## Project Overview

**PAGANS** (Prompts Aligned to Guidelines and Normalization System) is a comprehensive Python package that optimizes prompts across different Large Language Model (LLM) families. The project aims to solve the critical problem of prompt portability and effectiveness across various AI providers.

### 🎯 **Mission Statement**
To provide developers with a unified, intelligent system that automatically adapts prompts to work optimally with different LLM architectures, reducing the complexity of multi-provider AI integration.

### 🔧 **Core Value Proposition**
- **Universal Compatibility**: One prompt, optimized for any model family
- **Performance Optimization**: Model-specific prompt engineering for maximum effectiveness
- **Developer Experience**: Simple API that abstracts away provider complexities
- **Cost Efficiency**: Smart routing and optimization to minimize API costs

## Architecture Philosophy

### **Design Principles**

1. **Async-First Architecture**
   - All operations are asynchronous by default
   - Non-blocking I/O for high-throughput scenarios
   - Proper resource management with context managers

2. **Provider Agnostic Design**
   - Unified interface across all LLM providers
   - Easy addition of new providers without breaking changes
   - Fallback mechanisms for provider failures

3. **Type Safety & Validation**
   - Comprehensive type hints throughout the codebase
   - Pydantic models for data validation
   - Runtime type checking in development mode

4. **Modular & Extensible**
   - Plugin architecture for optimization strategies
   - Easy customization of prompt templates
   - Clear separation of concerns

## Developer Responsibilities

### 🚀 **As a Core Developer, You Should:**

#### **1. Code Quality & Standards**
- **Follow PEP 8** and project-specific style guidelines
- **Write comprehensive tests** for all new features (aim for >90% coverage)
- **Use type hints** consistently across all functions and classes
- **Document all public APIs** with clear docstrings and examples
- **Follow semantic versioning** for all releases

#### **2. Architecture & Design**
- **Maintain async-first approach** - never block the event loop
- **Design for extensibility** - new providers should be easy to add
- **Implement proper error handling** - graceful degradation and clear error messages
- **Consider performance implications** - optimize for speed and memory usage
- **Plan for scalability** - design components that can handle enterprise workloads

#### **3. Testing & Quality Assurance**
```python
# Example test structure
async def test_optimization_with_caching():
    """Test that caching works correctly for repeated optimizations."""
    async with PromptOptimizer() as optimizer:
        # First optimization
        result1 = await optimizer.optimize(
            prompt="Test prompt",
            target_model="openai/gpt-4o-mini"
        )
        
        # Second optimization (should use cache)
        result2 = await optimizer.optimize(
            prompt="Test prompt", 
            target_model="openai/gpt-4o-mini"
        )
        
        assert result1.optimized == result2.optimized
        assert optimizer.get_cache_size() == 1
```

#### **4. Documentation & Examples**
- **Write clear, executable examples** for all major features
- **Maintain up-to-date API documentation**
- **Create troubleshooting guides** for common issues
- **Document performance characteristics** and limitations

#### **5. Community & Collaboration**
- **Review pull requests thoroughly** - focus on design, tests, and documentation
- **Respond to issues promptly** with helpful guidance
- **Maintain backward compatibility** whenever possible
- **Communicate breaking changes clearly** with migration guides

## Development Workflow

### **Setting Up Development Environment**

This project uses `uv` for fast, reliable Python package management.

```bash
# Clone the repository
git clone https://github.com/abubakarsiddik31/pagans.git
cd pagans

# Install with development dependencies using uv
uv sync --dev

# Install the package in editable mode
uv pip install -e .

# Set up your OpenRouter API key (get one at https://openrouter.ai/)
export OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Set up pre-commit hooks
uv run pre-commit install

# Run tests to verify setup
uv run pytest

# Run examples to test functionality
uv run examples/anthropic_example.py
```

### **Working with uv**

All commands in this project should be run through `uv` for consistency:

```bash
# Run Python scripts
uv run python script.py
uv run examples/openai_example.py

# Run tests
uv run pytest
uv run pytest --cov=pagans

# Run linting and formatting
uv run ruff check
uv run black .
uv run mypy src/

# Install new dependencies
uv add httpx  # Add runtime dependency
uv add --dev pytest  # Add development dependency

# Update dependencies
uv sync  # Sync with lockfile
uv lock --upgrade  # Update lockfile
```

### **Branch Strategy**
- **`main`**: Production-ready code, protected branch
- **`develop`**: Integration branch for new features
- **`feature/*`**: Individual feature development
- **`hotfix/*`**: Critical bug fixes for production

### **Code Review Process**
1. **Create feature branch** from `develop`
2. **Implement feature** with comprehensive tests
3. **Update documentation** and examples
4. **Submit pull request** with detailed description
5. **Address review feedback** promptly
6. **Merge after approval** and CI passes

## Key Components Deep Dive

### **1. Core Optimizer (`core.py`)**
```python
class PromptOptimizer:
    """
    Main orchestrator for prompt optimization.
    
    Responsibilities:
    - Model family detection
    - Provider selection and routing
    - Caching and performance optimization
    - Error handling and retry logic
    """
```

**Developer Guidelines:**
- Always validate inputs before processing
- Implement proper caching strategies
- Handle provider failures gracefully
- Log performance metrics for monitoring

### **2. Provider Clients (`client.py`)**
```python
class BaseProviderClient:
    """
    Abstract base class for all provider clients.
    
    Must implement:
    - authenticate()
    - optimize_prompt()
    - validate_model()
    - handle_errors()
    """
```

**Developer Guidelines:**
- Implement consistent error handling across providers
- Use exponential backoff for retries
- Respect rate limits and quotas
- Provide detailed error messages

### **3. Optimization Prompts (`optimizer_prompts/`)**
```python
class OptimizationPromptManager:
    """
    Manages family-specific optimization strategies.
    
    Features:
    - Jinja2 template rendering
    - Dynamic prompt generation
    - A/B testing support
    """
```

**Developer Guidelines:**
- Research model-specific best practices thoroughly
- Test prompts across different use cases
- Maintain template versioning for A/B testing
- Document optimization strategies clearly

### **4. Model Registry (`models.py`)**
```python
def detect_model_family(model_name: str) -> ModelFamily:
    """
    Smart model family detection with fuzzy matching.
    
    Should handle:
    - Official model names
    - Provider-prefixed names
    - Version variations
    - Aliases and shortcuts
    """
```

**Developer Guidelines:**
- Keep model mappings up-to-date
- Handle edge cases and ambiguous names
- Provide clear error messages for unsupported models
- Test with real-world model name variations

## Testing Strategy

### **Test Categories**

1. **Unit Tests** (`test_*.py`)
   - Test individual functions and classes
   - Mock external dependencies
   - Focus on edge cases and error conditions

2. **Integration Tests** (`test_integration.py`)
   - Test real API interactions
   - Verify end-to-end workflows
   - Use test API keys and sandbox environments

3. **Performance Tests** (`test_performance.py`)
   - Measure optimization speed
   - Test concurrent operations
   - Validate memory usage

4. **Edge Case Tests** (`test_edge_cases.py`)
   - Test error conditions
   - Validate input sanitization
   - Test provider failure scenarios

### **Testing Best Practices**

```python
# Good: Comprehensive test with clear assertions
async def test_batch_optimization_with_mixed_models():
    """Test batch optimization with different model families."""
    prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
    models = ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.5-pro"]
    
    async with PromptOptimizer() as optimizer:
        results = await optimizer.optimize_multiple(
            prompts=prompts,
            target_models=models,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all(result.optimization_time < 10.0 for result in results)
        assert len(set(result.target_family for result in results)) == 3
```

## Performance Guidelines

### **Optimization Targets**
- **Response Time**: < 5 seconds for single optimization
- **Throughput**: > 100 concurrent optimizations
- **Memory Usage**: < 100MB for typical workloads
- **Cache Hit Rate**: > 80% for repeated prompts

### **Performance Best Practices**

1. **Use Connection Pooling**
```python
# Good: Reuse HTTP connections
self.client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)
```

2. **Implement Smart Caching**
```python
# Good: Semantic caching with TTL
cache_key = hashlib.sha256(f"{prompt}:{model}:{version}".encode()).hexdigest()
if cache_key in self._cache and not self._is_expired(cache_key):
    return self._cache[cache_key]
```

3. **Optimize for Concurrency**
```python
# Good: Use semaphores for rate limiting
semaphore = asyncio.Semaphore(max_concurrent)
async with semaphore:
    return await self._optimize_single(prompt)
```

## Security Considerations

### **API Key Management**
- Never log API keys or sensitive data
- Use environment variables for configuration
- Implement key rotation support
- Validate API key formats before use

### **Input Validation**
```python
def validate_prompt(prompt: str) -> str:
    """Validate and sanitize user input."""
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty")
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt too long: {len(prompt)} > {MAX_PROMPT_LENGTH}")
    
    # Sanitize potential injection attempts
    return prompt.strip()
```

### **Error Handling**
- Never expose internal errors to users
- Log detailed errors for debugging
- Implement rate limiting for error scenarios
- Use structured logging for security events

## Release Process

### **Version Management**
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Tag releases with detailed changelog
- Maintain backward compatibility in minor versions
- Provide migration guides for breaking changes

### **Release Checklist**
- [ ] All tests pass: `uv run pytest`
- [ ] Documentation is up-to-date
- [ ] Examples work with new version: `uv run examples/openai_example.py`
- [ ] Performance benchmarks are acceptable
- [ ] Security review completed
- [ ] Changelog updated
- [ ] Version bumped in `pyproject.toml`

## Monitoring & Observability

### **Key Metrics to Track**
- **Optimization Success Rate**: % of successful optimizations
- **Average Response Time**: Mean time for optimization
- **Provider Availability**: Uptime of each provider
- **Cache Hit Rate**: Efficiency of caching system
- **Error Rate**: Frequency and types of errors

### **Logging Standards**
```python
import logging

logger = logging.getLogger(__name__)

# Good: Structured logging with context
logger.info(
    "Optimization completed",
    extra={
        "model": target_model,
        "family": family.value,
        "duration": optimization_time,
        "cache_hit": cache_hit,
        "tokens_used": tokens_used
    }
)
```

## Contributing Guidelines

### **Before You Start**
1. **Check existing issues** to avoid duplicate work
2. **Discuss major changes** in GitHub issues first
3. **Read the codebase** to understand patterns and conventions
4. **Set up development environment** properly

### **Pull Request Requirements**
- [ ] **Clear description** of changes and motivation
- [ ] **Comprehensive tests** for new functionality
- [ ] **Updated documentation** for API changes
- [ ] **Performance impact** assessment
- [ ] **Backward compatibility** consideration
- [ ] **Security implications** review

### **Code Review Focus Areas**
1. **Correctness**: Does the code work as intended?
2. **Performance**: Are there any performance regressions?
3. **Security**: Are there any security vulnerabilities?
4. **Maintainability**: Is the code easy to understand and modify?
5. **Testing**: Are edge cases and error conditions covered?

## Getting Help

### **Resources**
- **Documentation**: [https://pagans.readthedocs.io/](https://pagans.readthedocs.io/)
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord/Slack**: Real-time developer chat (coming soon)

### **Common Questions**
- **"How do I add a new provider?"** - See `client.py` and follow the `BaseProviderClient` pattern
- **"How do I optimize prompts for a new model family?"** - Add to `optimizer_prompts/` with research-backed strategies
- **"How do I debug optimization failures?"** - Enable debug logging and check provider-specific error handling

---

**Remember**: PAGANS is more than just a library - it's a platform that empowers developers to build better AI applications. Every contribution should advance this mission while maintaining the highest standards of quality, performance, and developer experience.
