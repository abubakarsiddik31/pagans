"""
Simple example of using Prompt Optimizer with OpenAI GPT models.

This example demonstrates how to optimize prompts for OpenAI GPT models
using the Prompt Optimizer package.
"""

import asyncio
import os

from dotenv import load_dotenv

from pagans import PromptOptimizer

load_dotenv()


async def main():
    """Main example function for OpenAI GPT optimization."""

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    optimizer = PromptOptimizer(api_key=api_key)

    original_prompt = """
    Write a Python function that calculates the factorial of a number.
    """

    target_model = "openai/gpt-4o"
    try:
        result = await optimizer.optimize(
            prompt=original_prompt,
            target_model=target_model,
        )

        print("\n✅ Optimization complete!")
        print(f"Target model: {result.target_model}")
        print(f"Target family: {result.target_family.value}")
        print(f"Optimization time: {result.optimization_time:.2f}s")
        print(f"\nOptimized prompt:\n{result.optimized}")

        # Show optimization notes if available
        if result.optimization_notes:
            print(f"\nOptimization notes: {result.optimization_notes}")

        # Show tokens used if available
        if result.tokens_used:
            print(f"Tokens used: {result.tokens_used}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Always close the optimizer
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
