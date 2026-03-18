# PAGANS CLI

The `pagans` command provides a terminal interface for prompt optimization through OpenRouter.

## Global Flags

- `--api-key`: OpenRouter API key (optional if `OPENROUTER_API_KEY` is set)
- `--base-url`: OpenRouter base URL (optional if `OPENROUTER_BASE_URL` is set)
- `--optimizer-model`: model used to perform optimization (optional if `PAGANS_OPTIMIZER_MODEL` is set)

## Commands

## `optimize`

Optimize a single prompt for one target model.

```bash
pagans optimize \
  --prompt "Write a production-ready retry strategy for HTTP calls" \
  --target-model gpt-5.4
```

Use a prompt file:

```bash
pagans optimize \
  --prompt-file ./prompt.txt \
  --target-model claude-sonnet-4-20250514
```

JSON output:

```bash
pagans optimize \
  --prompt "Summarize this architecture decision" \
  --target-model gemini-3.1-pro-preview \
  --json
```

## `compare`

Optimize one prompt across multiple models.

```bash
pagans compare \
  --prompt "Design a scalable queue processing architecture" \
  --models "gpt-5.4,claude-sonnet-4-20250514,gemini-3.1-pro-preview,grok-4-1-fast-reasoning"
```

## `batch`

Optimize multiple prompts from a text file (`one prompt per line`).

```bash
pagans batch \
  --prompts-file ./prompts.txt \
  --target-model grok-4-1-fast-reasoning \
  --max-concurrent 5
```

## Exit Codes

- `0`: success
- `1`: runtime error or validation failure
- `2`: invalid CLI usage
