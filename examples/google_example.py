#!/usr/bin/env python3
"""
Simple example of using Prompt Optimizer with Google Gemini models.

This example demonstrates how to optimize prompts for Google Gemini models
using the Prompt Optimizer package.
"""

import asyncio
import os
from dotenv import load_dotenv
from prompt_optimizer import PromptOptimizer

load_dotenv()


async def main():
    """Main example function for Google Gemini optimization."""

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    # Create optimizer instance
    optimizer = PromptOptimizer(api_key=api_key)

    # Example prompt to optimize
    original_prompt = """
    Create a step-by-step guide for setting up a basic web server using Node.js.
    """

    # Target Google model
    target_model = "google/gemini-2.5-pro"

    try:
        print("Optimizing prompt for Google Gemini-2.5-Pro...")
        print(f"Original prompt: {original_prompt.strip()}")

        # Optimize the prompt
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
