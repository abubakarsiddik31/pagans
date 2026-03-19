# PAGANS CLI

The `pagans` command provides a terminal interface for prompt optimization across providers.

## Global Flags

- `--provider`: optimizer provider (`openrouter`, `openai`, `google`, `anthropic`, `zai`)
- `--api-key`: provider API key (optional if provider env key is set)
- `--base-url`: provider base URL (optional if provider env base URL is set)
- `--optimizer-model`: model used to perform optimization (optional if `PAGANS_OPTIMIZER_MODEL` is set)

Provider env vars:

- OpenRouter: `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`
- OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- Google AI Studio: `GOOGLE_API_KEY`, `GOOGLE_BASE_URL`
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`
- Z.ai: `ZAI_API_KEY`, `ZAI_BASE_URL`

## Commands

## `optimize`

Optimize a single prompt for one target model.

```bash
pagans --provider openrouter optimize \
  --prompt "Write a production-ready retry strategy for HTTP calls" \
  --target-model gpt-5.4
```

Use a prompt file:

```bash
pagans --provider anthropic optimize \
  --prompt-file ./prompt.txt \
  --target-model claude-sonnet-4-20250514
```

JSON output:

```bash
pagans --provider openai optimize \
  --prompt "Summarize this architecture decision" \
  --target-model gemini-3.1-pro-preview \
  --json
```

## `compare`

Optimize one prompt across multiple models.

```bash
pagans --provider google compare \
  --prompt "Design a scalable queue processing architecture" \
  --models "gpt-5.4,claude-sonnet-4-20250514,gemini-3.1-pro-preview,grok-4-1-fast-reasoning"
```

## `batch`

Optimize multiple prompts from a text file (`one prompt per line`).

```bash
pagans --provider zai batch \
  --prompts-file ./prompts.txt \
  --target-model grok-4-1-fast-reasoning \
  --max-concurrent 5
```

## Exit Codes

- `0`: success
- `1`: runtime error or validation failure
- `2`: invalid CLI usage
