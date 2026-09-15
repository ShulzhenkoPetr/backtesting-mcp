from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from btmcp.eval.ladder import Rung, load_ladder, resolve
from btmcp.eval.metrics import load_runs

USD_PER_EUR = 1.08

EXPERIMENTS: dict[str, tuple[int, str]] = {
    "x0": (0, "static token audit, no API calls"),
    "x1": (0, "no separate sweep: X1 is X2's surface-A arm (see EXPERIMENTS.md)"),
    "x2": (240, "surfaces A and B x 40 scenarios x 3 seeds; carries X1"),
    "x3": (192, "3 changes x 12 dev scenarios x 3 seeds, plus one 84-run holdout"),
    "x4": (80, "fencing on/off x surface A/B x 4 injections x 5 seeds"),
    "x5": (360, "2 surfaces x 3 capability tiers x 20 scenarios x 3 seeds"),
}

PRIOR = (12_000.0, 900.0, 0.55, 3.0)
"""Stated prior: input tokens, output tokens, cached fraction, tool calls per run."""


@dataclass(frozen=True)
class Profile:
    input_tokens: float
    output_tokens: float
    cached_fraction: float
    requests_per_run: float
    basis: str

    def tokens(self, runs: int) -> float:
        return runs * (self.input_tokens + self.output_tokens)

    def requests(self, runs: int) -> float:
        return runs * self.requests_per_run


def measure(results: Path) -> Profile | None:
    runs = [r for r in load_runs(results) if r.get("turns")]
    if not runs:
        return None
    inputs = [sum(t["input_tokens"] for t in r["turns"]) for r in runs]
    outputs = [sum(t["output_tokens"] for t in r["turns"]) for r in runs]
    cached = sum(sum(t.get("cache_read_tokens", 0) for t in r["turns"]) for r in runs)
    total_input = sum(inputs)
    return Profile(
        input_tokens=statistics.mean(inputs),
        output_tokens=statistics.mean(outputs),
        cached_fraction=(cached / total_input) if total_input else 0.0,
        requests_per_run=statistics.mean([len(r["turns"]) for r in runs]),
        basis=f"measured from {len(runs)} prior run(s) under {results}",
    )


def euros(rung: Rung, profile: Profile, runs: int) -> float:
    assert rung.input_per_mtok is not None and rung.output_per_mtok is not None
    cached = profile.input_tokens * profile.cached_fraction
    fresh = profile.input_tokens - cached
    usd_per_run = (
        fresh * rung.input_per_mtok / 1e6
        + cached * rung.input_per_mtok * 0.1 / 1e6
        + profile.output_tokens * rung.output_per_mtok / 1e6
    )
    return runs * usd_per_run / USD_PER_EUR


def _paid_report(rung: Rung, profile: Profile, targets: list[tuple[str, int]], ceiling: float) -> int:
    print(f"{'experiment':<12}{'runs':>7}{'EUR':>10}   notes")
    total = 0.0
    for name, runs in targets:
        cost = euros(rung, profile, runs)
        total += cost
        print(f"{name:<12}{runs:>7}{cost:>10.2f}   {EXPERIMENTS[name][1]}")

    print(f"\n{'TOTAL':<12}{'':>7}{total:>10.2f} EUR")
    if total > ceiling:
        print(f"OVER the {ceiling:.0f} EUR ceiling by {total - ceiling:.2f} EUR.")
        return 1
    print(f"Within the {ceiling:.0f} EUR ceiling.")
    return 0


def _free_report(rung: Rung, profile: Profile, targets: list[tuple[str, int]]) -> int:
    quota = rung.quota
    daily_tokens = quota.daily_tokens if quota else None
    daily_requests = quota.daily_requests if quota else None

    print(f"{'experiment':<12}{'runs':>7}{'tokens':>12}{'requests':>10}{'days':>8}   notes")
    total_tokens = 0.0
    total_requests = 0.0
    worst_days = 0.0

    for name, runs in targets:
        tokens = profile.tokens(runs)
        requests = profile.requests(runs)
        total_tokens += tokens
        total_requests += requests
        days = _days(tokens, requests, daily_tokens, daily_requests)
        worst_days = max(worst_days, days)
        print(
            f"{name:<12}{runs:>7}{tokens:>12,.0f}{requests:>10,.0f}"
            f"{_fmt_days(days):>8}   {EXPERIMENTS[name][1]}"
        )

    print(f"\n{'TOTAL':<12}{'':>7}{total_tokens:>12,.0f}{total_requests:>10,.0f}")
    if daily_tokens:
        share = total_tokens / daily_tokens
        print(
            f"{share:.0%} of the {daily_tokens:,} token/day quota "
            f"({_fmt_days(_days(total_tokens, total_requests, daily_tokens, daily_requests))} of budget)"
        )
    if quota and quota.per_minute_requests:
        minutes = total_requests / quota.per_minute_requests
        print(
            f"At {quota.per_minute_requests} requests/min the sweep floor is "
            f"{minutes:.0f} minutes of wall clock."
        )
    if not quota or (daily_tokens is None and daily_requests is None):
        print("No quota recorded for this rung; spend is unmetered but not unlimited.")
        return 0

    if _days(total_tokens, total_requests, daily_tokens, daily_requests) > 1.0:
        print("Does not fit in one day's quota. Split the sweep; run_eval.py is resumable.")
        return 1
    print("Fits within one day's quota.")
    return 0


def _days(tokens: float, requests: float, daily_tokens: int | None, daily_requests: int | None) -> float:
    spans = [
        tokens / daily_tokens if daily_tokens else 0.0,
        requests / daily_requests if daily_requests else 0.0,
    ]
    return max(spans)


def _fmt_days(days: float) -> str:
    return "-" if days == 0 else (f"{days:.2f}" if days < 100 else ">99")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default=None)
    parser.add_argument("--runs", type=int, default=None, help="override the run count")
    parser.add_argument("--model", default="cheap", help="a ladder rung name or provider:model")
    parser.add_argument("--ladder", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ceiling-eur", type=float, default=40.0)
    args = parser.parse_args(argv)

    ladder = load_ladder(args.ladder) if args.ladder else load_ladder()
    try:
        rung = resolve(args.model, ladder)
    except KeyError as exc:
        print(exc)
        return 1

    profile = measure(args.results) or Profile(*PRIOR, basis="stated prior (no prior sweeps found)")
    targets = (
        [(args.experiment, args.runs or EXPERIMENTS[args.experiment][0])]
        if args.experiment
        else list((name, count) for name, (count, _) in EXPERIMENTS.items())
    )

    kind = "free" if rung.free else "paid"
    print(f"rung {rung.name} ({rung.provider}/{rung.model}, {kind}) | {profile.basis}")
    print(
        f"per run: {profile.input_tokens:,.0f} input, {profile.output_tokens:,.0f} output, "
        f"{profile.requests_per_run:.1f} requests, {profile.cached_fraction:.0%} of input cached\n"
    )

    if rung.free:
        return _free_report(rung, profile, targets)
    return _paid_report(rung, profile, targets, args.ceiling_eur)


if __name__ == "__main__":
    raise SystemExit(main())
