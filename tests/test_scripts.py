from __future__ import annotations

import json
from pathlib import Path

import pytest

from btmcp.eval.ladder import load_ladder
from btmcp.server.app import AppConfig, Services
from scripts.estimate_cost import EXPERIMENTS, Profile, euros
from scripts.estimate_cost import main as estimate_main
from scripts.token_audit import audit, render


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    services = Services(AppConfig(surface="a", runs_root=tmp_path_factory.mktemp("runs")))
    return audit(services)


def test_x0_covers_every_read_and_inspection_tool(report: dict) -> None:
    assert set(report["per_tool"]) == {
        "list_symbols",
        "describe_series",
        "get_bars",
        "get_news",
        "run_backtest",
        "get_run",
        "explain_trades",
        "compare_runs",
    }
    assert report["n_cases"] >= 40


def test_x0_finds_no_tool_over_the_budget(report: dict) -> None:
    """The claim the repo makes about token budgeting, measured rather than asserted."""
    assert report["over_budget_fraction"] == 0.0
    assert report["global_max"] <= report["budget_tokens"]
    for tool, stats in report["per_tool"].items():
        assert stats["max"] <= report["budget_tokens"], f"{tool} exceeded the budget"


def test_x0_sweeps_the_documented_grid(report: dict) -> None:
    bars = [r for r in report["rows"] if r["tool"] == "get_bars"]
    assert {r["args"]["n_symbols"] for r in bars} == {1, 5, 20}
    assert {r["args"]["years"] for r in bars} == {1, 3, 8}


def test_x0_renders_a_markdown_table(report: dict) -> None:
    text = render(report)
    assert "| tool | n | p50 | p99 | max | over budget |" in text
    assert "static token audit" in text


def test_x0_needs_no_network_or_api_key() -> None:
    import ast

    tree = ast.parse(Path("scripts/token_audit.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"anthropic", "urllib", "requests", "httpx", "httpx2"}


def test_committed_x0_result_matches_a_fresh_run(report: dict) -> None:
    committed = json.loads(Path("results/x0/token_audit.json").read_text())
    assert committed["per_tool"] == report["per_tool"]
    assert committed["global_max"] == report["global_max"]


def test_cost_model_prices_cache_reads_at_a_tenth() -> None:
    rung = load_ladder().get("cheap")
    profile = Profile(1_000_000, 0, cached_fraction=0.0, requests_per_run=1, basis="test")
    cached = Profile(1_000_000, 0, cached_fraction=1.0, requests_per_run=1, basis="test")
    assert euros(rung, profile, 1) == pytest.approx(2.0 / 1.08)
    assert euros(rung, cached, 1) == pytest.approx(0.2 / 1.08)


def test_every_planned_experiment_is_priced(capsys) -> None:
    assert set(EXPERIMENTS) == {"x0", "x1", "x2", "x3", "x4", "x5"}
    assert estimate_main(["--model", "cheap", "--results", "results/does-not-exist"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "Within the 40 EUR ceiling" in out


def test_estimate_fails_when_a_sweep_would_break_the_ceiling(capsys) -> None:
    assert estimate_main(["--experiment", "x5", "--model", "strong", "--ceiling-eur", "1"]) == 1
    assert "OVER the 1 EUR ceiling" in capsys.readouterr().out


NO_PRIORS = ["--results", "results/does-not-exist"]
"""Pins the estimate to the stated prior.

Without it these assertions read whatever sweeps happen to be on disk, so they start
failing the day a real sweep lands under `results/`.
"""


def test_free_rungs_are_reported_against_quota_not_euros(capsys) -> None:
    """A free provider is still constrained; the currency is tokens and requests."""
    code = estimate_main(["--experiment", "x2", "--model", "groq-llama-70b", *NO_PRIORS])
    out = capsys.readouterr().out
    assert "EUR" not in out
    assert "tokens" in out and "requests" in out
    assert "token/day quota" in out
    assert code == 1, "x2 exceeds one day of the free quota and should say so"


def test_a_free_sweep_within_quota_passes(capsys) -> None:
    args = ["--experiment", "x4", "--model", "groq-llama-70b", "--runs", "10", *NO_PRIORS]
    assert estimate_main(args) == 0
    assert "Fits within one day's quota" in capsys.readouterr().out


def test_a_paid_rung_is_costed_in_euros(capsys) -> None:
    """Cerebras moved to pay-as-you-go, so it reports against the ceiling now."""
    assert estimate_main(["--experiment", "x2", "--model", "cerebras-oss-120b", *NO_PRIORS]) == 0
    out = capsys.readouterr().out
    assert "EUR" in out and "Within the 40 EUR ceiling" in out


def test_free_report_names_the_rate_limit_floor(capsys) -> None:
    estimate_main(["--experiment", "x1", "--model", "groq-llama-70b", *NO_PRIORS])
    assert "requests/min" in capsys.readouterr().out


def test_every_ladder_rung_is_priced_or_quota_bound() -> None:
    for rung in load_ladder():
        if rung.provider == "null":
            continue
        priced = rung.input_per_mtok is not None and rung.output_per_mtok is not None
        assert priced or rung.quota is not None, f"{rung.name} has neither a price nor a quota"


def test_unknown_model_is_reported_not_raised(capsys) -> None:
    assert estimate_main(["--model", "no-such-rung"]) == 1
    assert "unknown model" in capsys.readouterr().out
