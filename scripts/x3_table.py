from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from btmcp.eval.metrics import Aggregate, aggregate, by_seed_spread, load_runs

SPLIT = "dev"


def runs_for(root: Path, surface: str | None, split: str | None) -> list[dict[str, Any]]:
    runs = load_runs(root)
    if surface:
        runs = [r for r in runs if r["surface"] == surface]
    if split:
        runs = [r for r in runs if r["split"] == split]
    return runs


def delta(now: float | None, before: float | None) -> str:
    if now is None or before is None:
        return "-"
    return f"{now - before:+.0f}" if abs(now - before) >= 10 else f"{now - before:+.2f}"


def pct(value: float) -> str:
    return f"{value:.0%}"


def row(name: str, agg: Aggregate, base: Aggregate | None, prev: Aggregate | None, spread: float) -> str:
    success_delta = "-" if base is None else f"{(agg.task_success - base.task_success) * 100:+.1f}pp"
    step_delta = "-" if prev is None else f"{(agg.task_success - prev.task_success) * 100:+.1f}pp"
    verdict = "baseline" if base is None else _reading(agg, base, spread)
    d_calls = delta(agg.calls_to_completion, base.calls_to_completion if base else None)
    d_tokens = delta(agg.total_tokens, base.total_tokens if base else None)
    return (
        f"| {name} | {agg.n} | {pct(agg.task_success)} | {success_delta} | {step_delta} | "
        f"{agg.calls_to_completion:.2f} | {d_calls} | "
        f"{agg.total_tokens:,.0f} | {d_tokens} | "
        f"{agg.exhausted} | {verdict} |"
    )


def _reading(agg: Aggregate, base: Aggregate, spread: float) -> str:
    """A suggested reading, not a decision.

    The seed spread is the smallest difference this design can distinguish from noise,
    so a delta inside it is reported as unresolved rather than as a small effect.
    """
    change = agg.task_success - base.task_success
    if abs(change) < spread:
        return f"inside noise (spread {spread:.0%})"
    return "improves" if change > 0 else "regresses"


def report(args: argparse.Namespace) -> int:
    base_runs = {s: runs_for(args.baseline, s, SPLIT) for s in ("a", "b")}
    if not any(base_runs.values()):
        print(f"no {SPLIT}-split runs under {args.baseline}")
        return 1

    print(f"# X3 change table\n\nBaseline: `{args.baseline}` filtered to the {SPLIT} split.\n")
    for surface in ("a", "b"):
        base = aggregate(base_runs[surface], f"baseline/{surface}")
        spread = by_seed_spread(base_runs[surface])["spread"]
        print(f"\n## Surface {surface.upper()}\n")
        print(f"Seed spread on the baseline: **{spread:.1%}** - the noise floor for this column.\n")
        print(
            "| change | n | success | Δ base | Δ step | calls | Δ | tokens | Δ | exh | reading |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )
        print(row("00-baseline", base, None, None, spread))
        previous = base
        for path in args.changes:
            runs = runs_for(path, surface, SPLIT)
            if not runs:
                print(f"| {path.name} | 0 | - | - | - | - | - | - | - | - | no runs |")
                continue
            agg = aggregate(runs, f"{path.name}/{surface}")
            print(row(path.name, agg, base, previous, spread))
            previous = agg

    print(
        "\n*Δ base* is against the baseline, *Δ step* against the previous change. "
        "A delta smaller than the seed spread is not a result."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("changes", nargs="+", type=Path, help="change sweep directories, in order")
    parser.add_argument("--baseline", type=Path, required=True, help="the X2 sweep directory")
    args = parser.parse_args(argv)
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
