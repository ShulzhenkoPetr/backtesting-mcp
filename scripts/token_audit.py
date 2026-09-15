from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from btmcp.core.costs import CostModel
from btmcp.server.app import AppConfig, Services
from btmcp.server.shaping import count_tokens, serialise

END = dt.date(2025, 12, 31)
SYMBOL_GRID = (1, 5, 20)
YEAR_GRID = (1, 3, 8)
FIELD_SETS: tuple[list[str] | None, ...] = (
    None,
    ["close"],
    ["open", "high", "low", "close", "volume"],
)


def _window(years: int) -> tuple[dt.date, dt.date]:
    return dt.date(END.year - years + 1, 1, 1), END


def _cases(services: Services) -> list[tuple[str, dict[str, Any], Any]]:
    universe = services.bars.universe
    cases: list[tuple[str, dict[str, Any], Any]] = [
        ("list_symbols", {"as_of": END}, services.list_symbols(END)),
    ]

    for symbol in (universe[0], "SYN-08", "SYN-17"):
        cases.append(("describe_series", {"symbol": symbol}, services.describe_series(symbol, END)))

    for n_symbols, years, fields in itertools.product(SYMBOL_GRID, YEAR_GRID, FIELD_SETS):
        symbols = universe[:n_symbols]
        start, end = _window(years)
        summary, links = services.get_bars(symbols, start, end, END, fields)
        cases.append(
            (
                "get_bars",
                {"n_symbols": n_symbols, "years": years, "fields": len(fields or [])},
                services.shaped(summary, links).payload,
            )
        )

    for years in YEAR_GRID:
        start, end = _window(years)
        cases.append(("get_news", {"years": years}, services.get_news("SYN-04", start, end, END)))

    costs = CostModel(fee_bps=5.0, slippage_bps=2.0)
    spec = {"primitive": "sma_cross", "params": {"fast": 5, "slow": 10}}
    summary, links = services.run_backtest(spec, "SYN-04", dt.date(2019, 1, 1), dt.date(2021, 12, 31), costs)
    cases.append(("run_backtest", {}, services.shaped(summary, links).payload))
    cases.append(("get_run", {}, services.get_run(summary.run_id)))

    for trade_filter in ("worst_5", "best_5", "largest_drawdown", "all_in_window"):
        cases.append(
            (
                "explain_trades",
                {"filter": trade_filter},
                services.explain_trades(summary.run_id, trade_filter),  # type: ignore[arg-type]
            )
        )

    second, _ = services.run_backtest(
        {"primitive": "buy_and_hold", "params": {}},
        "SYN-04",
        dt.date(2019, 1, 1),
        dt.date(2021, 12, 31),
        costs,
    )
    cases.append(("compare_runs", {}, services.compare_runs([summary.run_id, second.run_id])))
    return cases


def audit(services: Services) -> dict[str, Any]:
    budget = services.config.budget_tokens
    per_tool: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []

    for tool, args, payload in _cases(services):
        tokens = count_tokens(serialise(services.shaped(payload).payload))
        per_tool.setdefault(tool, []).append(tokens)
        rows.append({"tool": tool, "args": args, "tokens": tokens, "over_budget": tokens > budget})

    summary = {}
    for tool, values in sorted(per_tool.items()):
        ordered = sorted(values)
        summary[tool] = {
            "n": len(values),
            "p50": int(statistics.median(ordered)),
            "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
            "max": max(ordered),
            "over_budget": sum(v > budget for v in values),
        }

    all_tokens = [r["tokens"] for r in rows]
    return {
        "budget_tokens": budget,
        "n_cases": len(rows),
        "over_budget_fraction": sum(r["over_budget"] for r in rows) / len(rows),
        "global_max": max(all_tokens),
        "per_tool": summary,
        "rows": rows,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "## X0 — static token audit",
        "",
        f"{report['n_cases']} legal argument combinations, tokenized offline. "
        f"Budget {report['budget_tokens']} tokens.",
        "",
        "| tool | n | p50 | p99 | max | over budget |",
        "|---|---|---|---|---|---|",
    ]
    for tool, stats in report["per_tool"].items():
        lines.append(
            f"| `{tool}` | {stats['n']} | {stats['p50']} | {stats['p99']} | "
            f"{stats['max']} | {stats['over_budget']} |"
        )
    lines += [
        "",
        f"Largest result across the whole grid: **{report['global_max']} tokens** "
        f"(budget {report['budget_tokens']}).",
        f"Fraction of requests over budget: **{report['over_budget_fraction']:.0%}**.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    services = Services(AppConfig(surface="a", runs_root=args.runs))
    report = audit(services)
    text = render(report)
    print(text)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "token_audit.json").write_text(json.dumps(report, indent=2, default=str))
        (args.out / "token_audit.md").write_text(text + "\n")
        print(f"\nwrote {args.out}/token_audit.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
