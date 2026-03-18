"""Unit tests for PAGANS CLI."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.pagans.cli import main
from src.pagans.models import ModelFamily, OptimizationResult


def _mk_result(target_model: str = "gpt-5.4") -> OptimizationResult:
    return OptimizationResult(
        original="Original",
        optimized="Optimized",
        target_model=target_model,
        target_family=ModelFamily.OPENAI,
        optimization_notes=None,
        optimization_time=0.25,
    )


def test_cli_optimize_json_output(capsys):
    mock_optimizer = AsyncMock()
    mock_optimizer.__aenter__.return_value = mock_optimizer
    mock_optimizer.optimize.return_value = _mk_result("gpt-5.4")

    with patch("src.pagans.cli.PAGANSOptimizer", return_value=mock_optimizer):
        code = main(
            [
                "optimize",
                "--prompt",
                "Test prompt",
                "--target-model",
                "gpt-5.4",
                "--json",
            ]
        )

    out = capsys.readouterr().out
    assert code == 0
    assert '"target_model": "gpt-5.4"' in out


def test_cli_compare_json_output(capsys):
    mock_optimizer = AsyncMock()
    mock_optimizer.__aenter__.return_value = mock_optimizer
    mock_optimizer.compare_optimizations.return_value = {
        "gpt-5.4": _mk_result("gpt-5.4"),
        "gemini-3.1-pro-preview": _mk_result("gemini-3.1-pro-preview"),
    }

    with patch("src.pagans.cli.PAGANSOptimizer", return_value=mock_optimizer):
        code = main(
            [
                "compare",
                "--prompt",
                "Test prompt",
                "--models",
                "gpt-5.4,gemini-3.1-pro-preview",
                "--json",
            ]
        )

    out = capsys.readouterr().out
    assert code == 0
    assert '"gpt-5.4"' in out
    assert '"gemini-3.1-pro-preview"' in out


def test_cli_batch_json_output(tmp_path: Path, capsys):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("Prompt one\nPrompt two\n", encoding="utf-8")

    mock_optimizer = AsyncMock()
    mock_optimizer.__aenter__.return_value = mock_optimizer
    mock_optimizer.optimize_multiple.return_value = [
        _mk_result("grok-4-1-fast-reasoning"),
        _mk_result("grok-4-1-fast-reasoning"),
    ]

    with patch("src.pagans.cli.PAGANSOptimizer", return_value=mock_optimizer):
        code = main(
            [
                "batch",
                "--prompts-file",
                str(prompts_file),
                "--target-model",
                "grok-4-1-fast-reasoning",
                "--json",
            ]
        )

    out = capsys.readouterr().out
    assert code == 0
    assert out.count('"target_model": "grok-4-1-fast-reasoning"') == 2


def test_cli_optimize_requires_prompt(capsys):
    code = main(["optimize", "--target-model", "gpt-5.4"])
    err = capsys.readouterr().err

    assert code == 1
    assert "A prompt is required" in err
