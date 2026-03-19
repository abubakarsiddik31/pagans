# Notebooks

This folder contains runnable Jupyter notebooks for PAGANS.

- `pagans_quickstart.ipynb`: end-to-end quickstart for single prompt optimization,
  cross-family comparison, and batch optimization.
- `pagans_openrouter_optimizer.ipynb`: OpenRouter optimizer provider setup and run.
- `pagans_openai_optimizer.ipynb`: OpenAI optimizer provider setup and run.
- `pagans_google_optimizer.ipynb`: Google AI Studio optimizer provider setup and run.
- `pagans_anthropic_optimizer.ipynb`: Anthropic optimizer provider setup and run.
- `pagans_zai_optimizer.ipynb`: Z.ai optimizer provider setup and run.

## Run locally

```bash
uv sync --dev
uv run jupyter lab
```

Then open any notebook in `notebooks/`.
