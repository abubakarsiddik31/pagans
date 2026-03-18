# Text Model Guidance Matrix

Last reviewed: **2026-03-18**

This file tracks official prompt-guidance sources and the text-model aliases currently maintained in PAGANS.

## OpenAI

Official docs used:
- https://developers.openai.com/api/docs/guides/prompt-engineering
- https://developers.openai.com/api/docs/guides/reasoning-best-practices
- https://developers.openai.com/api/docs/models

Primary optimization targets (latest text focus):
- `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`
- `gpt-5.2`, `gpt-5.2-mini`, `gpt-5.2-nano`, `gpt-5.1`

Compatibility aliases kept for routing:
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4o`

## Anthropic

Official docs used:
- https://docs.anthropic.com/en/docs/about-claude/models/overview
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags

Primary optimization targets (latest text focus):
- `claude-opus-4-20250514`, `claude-sonnet-4-20250514`
- `claude-opus-4-1-20250805`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
- `claude-3-7-sonnet-20250219`

Compatibility aliases kept for routing:
- `claude-opus-4.6`, `claude-sonnet-4.6`, `claude-haiku-4.5`, `claude-opus-4.1`, `claude-opus-4`, `claude-sonnet-4`, `claude-3.5-sonnet`

## Google Gemini

Official docs used:
- https://ai.google.dev/gemini-api/docs/models/gemini-v2
- https://ai.google.dev/gemini-api/docs/models/generative-models

Primary optimization targets (latest text focus):
- `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`

Compatibility aliases kept for routing:
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-flash-lite-preview-09-2025`, `gemini-2.0-flash`

## xAI Grok

Official docs used:
- https://docs.x.ai/docs/guides/deferred-chat-completions
- https://docs.x.ai/docs/api-reference/chat-completions

OpenRouter model index used:
- https://openrouter.ai/models?q=grok

Primary optimization targets (latest text focus):
- `grok-4-1-fast-reasoning`, `grok-4-1-fast-non-reasoning`
- `grok-4-fast-reasoning`, `grok-4-fast-non-reasoning`
- `grok-code-fast-1`

Compatibility aliases kept for routing:
- `grok-4`, `grok-4-fast`, `grok-4.20-beta`, `grok-4.20-multi-agent-beta`

## Update Process

When updating model aliases:
1. Confirm canonical model IDs in each provider's official docs.
2. Confirm OpenRouter provider prefixes/IDs for routed usage.
3. Update `src/pagans/models.py`:
   - `SHORT_MODEL_NAMES`
   - `OPENROUTER_MODEL_MAPPINGS`
   - `MODEL_MAPPINGS`
4. Add/update tests in `tests/test_models.py` and `tests/test_optimizer_prompts.py`.
5. Update this file's "Last reviewed" date.
