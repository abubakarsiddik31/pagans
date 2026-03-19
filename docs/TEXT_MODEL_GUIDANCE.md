# Text Model Guidance Matrix

Last reviewed: **2026-03-19**

This file tracks official prompt-guidance sources, provider API references, and the text-model aliases currently maintained in PAGANS.

## Optimizer Provider APIs

Official API docs used for provider client configuration:

- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat/create-chat-completion
- OpenAI Models API: https://platform.openai.com/docs/api-reference/models
- Google AI Studio OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
- Anthropic Models API: https://docs.anthropic.com/en/api/models
- Z.ai developer quick start: https://docs.z.ai/guides

Configured default base URLs:

- OpenRouter: `https://openrouter.ai/api/v1`
- OpenAI: `https://api.openai.com/v1`
- Google AI Studio: `https://generativelanguage.googleapis.com/v1beta/openai`
- Anthropic: `https://api.anthropic.com/v1`
- Z.ai: `https://api.z.ai/api/paas/v4`

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
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/openai

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
2. Confirm provider prefix/ID conventions for routed usage.
3. Update `src/pagans/models.py` and `src/pagans/models/__init__.py`:
   - `SHORT_MODEL_NAMES`
   - provider-specific model mappings
   - `MODEL_MAPPINGS`
4. Add/update tests in `tests/test_models.py` and `tests/test_optimizer_prompts.py`.
5. Update this file's "Last reviewed" date.
