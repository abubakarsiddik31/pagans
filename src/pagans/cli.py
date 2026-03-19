"""Command-line interface for PAGANS."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .core import PAGANSOptimizer


def _read_prompt(arg_value: str | None, file_value: str | None) -> str:
    if arg_value and file_value:
        msg = "Use either --prompt or --prompt-file, not both."
        raise ValueError(msg)
    if file_value:
        return Path(file_value).read_text(encoding="utf-8")
    if arg_value:
        return arg_value
    msg = "A prompt is required. Provide --prompt or --prompt-file."
    raise ValueError(msg)


def _read_prompts_file(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        msg = "No prompts found in file."
        raise ValueError(msg)
    return prompts


def _to_dict(result: Any) -> dict[str, Any]:
    return {
        "original": result.original,
        "optimized": result.optimized,
        "target_model": result.target_model,
        "target_family": result.target_family.value,
        "optimization_notes": result.optimization_notes,
        "tokens_used": result.tokens_used,
        "optimization_time": result.optimization_time,
    }


def _print_result(result: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_to_dict(result), indent=2))
        return

    print(f"Target model: {result.target_model}")
    print(f"Target family: {result.target_family.value}")
    if result.optimization_time is not None:
        print(f"Optimization time: {result.optimization_time:.2f}s")
    print()
    print(result.optimized)


def _print_compare(results: dict[str, Any], as_json: bool) -> None:
    if as_json:
        payload = {}
        for model, result in results.items():
            if hasattr(result, "target_model"):
                payload[model] = _to_dict(result)
            else:
                payload[model] = {"error": str(result)}
        print(json.dumps(payload, indent=2))
        return

    for model, result in results.items():
        print(f"== {model} ==")
        if hasattr(result, "target_model"):
            print(result.optimized)
        else:
            print(f"ERROR: {result}")
        print()


async def _run_optimize(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args.prompt, args.prompt_file)
    async with PAGANSOptimizer(
        api_key=args.api_key,
        base_url=args.base_url,
        optimizer_model=args.optimizer_model,
    ) as optimizer:
        result = await optimizer.optimize(
            prompt=prompt,
            target_model=args.target_model,
            optimization_notes=args.notes,
            use_cache=not args.no_cache,
        )
    _print_result(result, args.json)
    return 0


async def _run_compare(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args.prompt, args.prompt_file)
    target_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not target_models:
        raise ValueError("At least one model is required via --models.")

    async with PAGANSOptimizer(
        api_key=args.api_key,
        base_url=args.base_url,
        optimizer_model=args.optimizer_model,
    ) as optimizer:
        results = await optimizer.compare_optimizations(
            prompt=prompt,
            target_models=target_models,
            optimization_notes=args.notes,
            use_cache=not args.no_cache,
        )
    _print_compare(results, args.json)
    return 0


async def _run_batch(args: argparse.Namespace) -> int:
    prompts = _read_prompts_file(args.prompts_file)

    async with PAGANSOptimizer(
        api_key=args.api_key,
        base_url=args.base_url,
        optimizer_model=args.optimizer_model,
    ) as optimizer:
        results = await optimizer.optimize_multiple(
            prompts=prompts,
            target_model=args.target_model,
            optimization_notes=args.notes,
            use_cache=not args.no_cache,
            max_concurrent=args.max_concurrent,
        )

    if args.json:
        print(json.dumps([_to_dict(result) for result in results], indent=2))
    else:
        for idx, result in enumerate(results, start=1):
            print(f"== Result {idx} ({result.target_model}) ==")
            print(result.optimized)
            print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pagans",
        description="PAGANS CLI for prompt optimization via OpenRouter.",
    )
    parser.add_argument(
        "--api-key", help="OpenRouter API key. Defaults to OPENROUTER_API_KEY."
    )
    parser.add_argument(
        "--base-url", help="OpenRouter base URL. Defaults to OPENROUTER_BASE_URL."
    )
    parser.add_argument(
        "--optimizer-model",
        help="Model used to perform optimization. Defaults to PAGANS_OPTIMIZER_MODEL or package default.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize one prompt for one target model."
    )
    optimize_parser.add_argument("--prompt", help="Prompt text.")
    optimize_parser.add_argument(
        "--prompt-file", help="Path to file containing the prompt text."
    )
    optimize_parser.add_argument(
        "--target-model", required=True, help="Target model to optimize for."
    )
    optimize_parser.add_argument("--notes", help="Additional optimization notes.")
    optimize_parser.add_argument("--json", action="store_true", help="Output JSON.")
    optimize_parser.add_argument(
        "--no-cache", action="store_true", help="Disable cache."
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare optimization across multiple target models."
    )
    compare_parser.add_argument("--prompt", help="Prompt text.")
    compare_parser.add_argument(
        "--prompt-file", help="Path to file containing the prompt text."
    )
    compare_parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated target models, e.g. 'gpt-5.4,claude-sonnet-4-20250514'.",
    )
    compare_parser.add_argument("--notes", help="Additional optimization notes.")
    compare_parser.add_argument("--json", action="store_true", help="Output JSON.")
    compare_parser.add_argument(
        "--no-cache", action="store_true", help="Disable cache."
    )

    batch_parser = subparsers.add_parser(
        "batch", help="Optimize multiple prompts from a file."
    )
    batch_parser.add_argument(
        "--prompts-file",
        required=True,
        help="Path to text file containing one prompt per line.",
    )
    batch_parser.add_argument(
        "--target-model", required=True, help="Target model to optimize for."
    )
    batch_parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Max concurrent optimizations."
    )
    batch_parser.add_argument("--notes", help="Additional optimization notes.")
    batch_parser.add_argument("--json", action="store_true", help="Output JSON.")
    batch_parser.add_argument("--no-cache", action="store_true", help="Disable cache.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "optimize":
            return asyncio.run(_run_optimize(args))
        if args.command == "compare":
            return asyncio.run(_run_compare(args))
        if args.command == "batch":
            return asyncio.run(_run_batch(args))
        parser.print_help()
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
