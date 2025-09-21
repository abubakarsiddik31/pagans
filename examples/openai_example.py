"""
Simple example of using PAGANS with OpenAI GPT models.

This example demonstrates how to optimize prompts for OpenAI GPT models
using the PAGANS (Prompts Aligned to Guidelines and Normalization System) package.
"""

import asyncio
import os

from dotenv import load_dotenv

from pagans import PAGANSOptimizer

load_dotenv()


async def main():
    """Main example function for OpenAI GPT optimization using PAGANS."""

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

    # Create PAGANS optimizer with OpenRouter provider
    optimizer = PAGANSOptimizer(
        api_key=api_key,
        provider=Provider.OPENROUTER
    )

    original_prompt = """
    Write a Python function that calculates the factorial of a number.
    """

    # Use short model name - PAGANS will automatically detect the model family
    target_model = "gpt-4o"

    try:
        print("🚀 Optimizing prompt for GPT-4o using PAGANS...")
        print(f"Original prompt: {original_prompt.strip()}")

        # Demonstrate the new provider capabilities
        result = await optimizer.optimize(
            prompt=original_prompt,
            target_model=target_model,
        )

        print("\n✅ Optimization complete!")
        print(f"Target model: {result.target_model}")
        print(f"Provider: OpenRouter (Unified API)")
        print(f"Target family: {result.target_family.value}")
        print(f"Optimization time: {result.optimization_time:.2f}s")
        print(f"\nOptimized prompt:\n{result.optimized}")

        # Show optimization notes if available
        if result.optimization_notes:
            print(f"\nOptimization notes: {result.optimization_notes}")

        # Show tokens used if available
        if hasattr(result, 'tokens_used') and result.tokens_used:
            print(f"Tokens used: {result.tokens_used}")

        # Demonstrate model family detection capability
        print("\n🔍 Demonstrating model family detection...")
        print(f"Detected family for {result.target_model}: {result.target_family.value}")

        # Show supported models by family
        print("\n📋 Supported models by family:")
        for family, models in optimizer.get_supported_models().items():
            print(f"  {family.value}: {len(models)} models available")
            print(f"    Examples: {', '.join(models[:3])}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Always close the optimizer
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
