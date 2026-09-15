from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from btmcp.eval.judge import Judge
from btmcp.eval.ladder import load_ladder
from btmcp.eval.metrics import aggregate, load_runs
from btmcp.eval.models import build_model
from btmcp.eval.runner import outcome_exists, run_scenario, write_outcome
from btmcp.eval.scenario import load_scenarios
from btmcp.server.cli import load_surface


async def sweep(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(
        Path(args.scenarios),
        split=None if args.split == "all" else args.split,
        ids=args.ids.split(",") if args.ids else None,
    )
    if not scenarios:
        print("no scenarios matched")
        return 1

    if args.split == "holdout" and not args.confirm_holdout:
        print(
            "Refusing to run the holdout split without --confirm-holdout.\n"
            "Holdout is for one confirming run on a final configuration. Iterating "
            "against it turns the dev/holdout split into schema overfitting with extra steps."
        )
        return 1

    ladder = load_ladder(args.ladder) if args.ladder else load_ladder()

    if args.judge and not args.judge_model:
        print(
            "--judge needs --judge-model: grading with whichever rung happens to be "
            "the default silently changes what the rubric scores mean.\n"
            f"Available rungs: {', '.join(ladder.names())}"
        )
        return 1

    build = load_surface(args.surface)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = Judge(build_model(args.judge_model, ladder), out / "judge_cache.json") if args.judge else None

    planned = len(scenarios) * args.seeds
    print(
        f"{planned} runs: {len(scenarios)} scenarios x {args.seeds} seeds, "
        f"surface {args.surface}, model {args.model}"
    )
    if args.dry_run:
        for scenario in scenarios:
            print(f"  {scenario.id:<12} {scenario.category:<15} {scenario.split}")
        return 0

    done = 0
    for seed in range(args.seeds):
        for scenario in scenarios:
            if outcome_exists(scenario.id, args.surface, seed, out):
                done += 1
                continue
            model = build_model(args.model, ladder)
            outcome = await run_scenario(
                scenario,
                build,
                model,
                seed=seed,
                fencing=args.fencing == "on",
                runs_root=Path(args.runs),
                surface_name=args.surface,
            )
            if judge and scenario.rubric:
                grade = await judge.grade(scenario.rubric, outcome.final_answer)
                outcome.rubric_score = grade.score
            write_outcome(outcome, out)
            done += 1
            flag = "ok " if outcome.task_success else "FAIL"
            print(f"  [{done}/{planned}] {flag} {scenario.id} seed={seed} calls={outcome.total_calls}")

    summary = aggregate(load_runs(out), label=f"{args.surface}/{args.model}/{args.split}")
    (out / "summary.json").write_text(json.dumps(summary.as_row(), indent=2))
    print(f"\ntask_success {summary.task_success:.0%} over {summary.n} runs -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--surface", choices=["a", "b"], default="a")
    parser.add_argument(
        "--model",
        default="null",
        help=(
            "a ladder rung name (see btmcp/eval/ladder.yaml), a raw provider:model pair, "
            "or 'null' to smoke-test the sweep path at zero cost"
        ),
    )
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--scenarios", default="btmcp/eval/scenarios")
    parser.add_argument("--ids", default=None, help="comma-separated scenario ids")
    parser.add_argument("--fencing", choices=["on", "off"], default="on")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge", action="store_true", help="grade rubrics (costs one call per scenario)")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="ladder rung used for rubric grading; required with --judge, and stated rather than defaulted",
    )
    parser.add_argument("--ladder", type=Path, default=None, help="override btmcp/eval/ladder.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-holdout",
        action="store_true",
        help="required to run --split holdout; use once, on a final configuration",
    )
    args = parser.parse_args(argv)

    return asyncio.run(sweep(args))


if __name__ == "__main__":
    raise SystemExit(main())
