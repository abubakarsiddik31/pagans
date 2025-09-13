"""
Simple example of using Prompt Optimizer with Google Gemini models.

This example demonstrates how to optimize prompts for Google Gemini models
using the Prompt Optimizer package.
"""

import asyncio
import os

from dotenv import load_dotenv

from pagans import PromptOptimizer

load_dotenv()


async def main():
    """Main example function for Google Gemini optimization."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not found!")
        print("\nTo run this example:")
        print("1. Get an API key from https://openrouter.ai/")
        print("2. Set the environment variable:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        print("3. Run the example again")
        return

    optimizer = PromptOptimizer(api_key=api_key)

    original_prompt = """
    Create a step-by-step guide for setting up a basic web server using Node.js.
    """

    target_model = "google/gemini-2.5-pro"

    try:
        print("Optimizing prompt for Google Gemini-2.5-Pro...")
        print(f"Original prompt: {original_prompt.strip()}")

        result = await optimizer.optimize(
            prompt=original_prompt,
            target_model=target_model,
        )

        print("\n✅ Optimization complete!")
        print(f"Target model: {result.target_model}")
        print(f"Target family: {result.target_family.value}")
        print(f"Optimization time: {result.optimization_time:.2f}s")
        print(f"\nOptimized prompt:\n{result.optimized}")

        if result.optimization_notes:
            print(f"\nOptimization notes: {result.optimization_notes}")

        if result.tokens_used:
            print(f"Tokens used: {result.tokens_used}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
