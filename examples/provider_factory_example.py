"""
Comprehensive example demonstrating PAGANS unified optimization capabilities.

This example showcases PAGANS' ability to optimize prompts across different model families
using a unified OpenRouter API, demonstrating the core value proposition of seamless
cross-model optimization.
"""

import asyncio
import os

from dotenv import load_dotenv

from pagans import PAGANSOptimizer, create_optimizer


async def demonstrate_provider_factory():
    """Demonstrate the provider factory pattern capabilities."""

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not found!")
        print("\nTo run this example:")
        print("1. Get an API key from https://openrouter.ai/")
        print("2. Set the environment variable:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        print("3. Run the example again")
        return

    print("🚀 PAGANS Unified Optimization Demonstration")
    print("=" * 50)

    # Example 1: Using the convenience function
    print("\n📦 Example 1: Using create_optimizer() convenience function")
    optimizer = create_optimizer(api_key=api_key)

    # Example 2: Direct PAGANSOptimizer usage
    print("\n🎯 Example 2: Direct PAGANSOptimizer usage")
    print("PAGANS uses OpenRouter as the unified API for all model families")

    # Example 3: Cross-model family optimization
    print("\n🔄 Example 3: Cross-model family optimization comparison")

    original_prompt = """
    Create a Python function that implements a binary search algorithm
    with proper error handling and documentation.
    """

    # Test different models across model families
    test_models = [
        "gpt-4o",           # OpenAI family
        "claude-3.5-sonnet", # Anthropic family
        "gemini-2.5-pro",   # Google family
    ]

    results = {}
    for model in test_models:
        try:
            print(f"\n🎯 Optimizing for {model}...")
            result = await optimizer.optimize(
                prompt=original_prompt,
                target_model=model,
            )
            results[model] = result
            print(f"✅ Success: {result.target_family.value} family, {result.optimization_time:.2f}s")
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[model] = None

    # Example 4: Model family detection demonstration
    print("\n🔍 Example 4: Model family detection capabilities")
    print("PAGANS automatically detects model families and applies appropriate optimization strategies")

    # Show supported models by family
    print("\n📋 Supported models by family:")
    for family, models in optimizer.get_supported_models().items():
        print(f"  {family.value}: {len(models)} models available")
        print(f"    Examples: {', '.join(models[:3])}")

    # Example 5: Cache demonstration
    print("\n💾 Example 5: Optimization caching")
    print(f"Cache size before: {optimizer.get_cache_size()}")

    # Optimize the same prompt again to demonstrate caching
    result2 = await optimizer.optimize(
        prompt=original_prompt,
        target_model="gpt-4o",
    )
    print(f"Cache size after: {optimizer.get_cache_size()}")
    print(f"Results identical: {results['gpt-4o'].optimized == result2.optimized}")

    # Example 6: Batch optimization
    print("\n⚡ Example 6: Batch optimization")
    prompts = [
        "Write a function to calculate Fibonacci numbers",
        "Create a REST API endpoint with FastAPI",
        "Implement a priority queue in Python",
    ]

    batch_results = await optimizer.optimize_multiple(
        prompts=prompts,
        target_model="claude-3.5-sonnet",
        max_concurrent=2
    )

    print(f"Batch optimization completed: {len(batch_results)} results")

    # Cleanup
    await optimizer.close()

    print("\n🎉 PAGANS unified optimization demonstration complete!")


async def demonstrate_error_handling():
    """Demonstrate error handling with PAGANS."""

    print("\n🛡️ Error Handling Demonstration")
    print("=" * 30)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ API key not found, skipping error handling demo")
        return

    optimizer = create_optimizer(api_key=api_key)

    # Test with invalid model
    try:
        result = await optimizer.optimize(
            prompt="Test prompt",
            target_model="invalid-model-name"
        )
    except Exception as e:
        print(f"✅ Expected error caught: {type(e).__name__}: {e}")

    # Test with empty prompt
    try:
        result = await optimizer.optimize(
            prompt="",
            target_model="gpt-4o"
        )
    except Exception as e:
        print(f"✅ Expected error caught: {type(e).__name__}: {e}")

    await optimizer.close()


async def main():
    """Main demonstration function."""
    await demonstrate_provider_factory()
    await demonstrate_error_handling()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())