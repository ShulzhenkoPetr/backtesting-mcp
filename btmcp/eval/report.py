from __future__ import annotations

from pathlib import Path
from typing import Any

from btmcp.eval.metrics import (
    Aggregate,
    aggregate,
    by_category,
    by_seed_spread,
    flip_table,
    load_runs,
    succeeded,
)

COLUMNS = (
    ("label", "category"),
    ("n", "n"),
    ("task_success", "success"),
    ("calls_to_completion", "calls"),
    ("exhausted", "maxed"),
    ("total_tokens", "tokens"),
    ("wrong_tool_rate", "wrong tool"),
    ("p99_result_tokens", "p99 result tok"),
)


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if key in {"task_success", "wrong_tool_rate"}:
        return f"{value:.0%}"
    if key == "total_tokens":
        return f"{value:,.0f}"
    return str(value)


def markdown_table(rows: list[Aggregate]) -> str:
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, sep]
    for row in rows:
        data = row.as_row()
        lines.append("| " + " | ".join(_fmt(key, data[key]) for key, _ in COLUMNS) + " |")
    return "\n".join(lines)


def special_metrics_table(overall: Aggregate) -> str:
    rows = [
        ("recovery_rate", overall.recovery_rate),
        ("elicit_rate", overall.elicit_rate),
        ("attack_success_rate", overall.attack_success_rate),
    ]
    lines = ["| metric | value |", "|---|---|"]
    for name, value in rows:
        lines.append(f"| {name} | {'-' if value is None else f'{value:.0%}'} |")
    return "\n".join(lines)


def flip_markdown(
    runs_a: list[dict[str, Any]], runs_b: list[dict[str, Any]], a_name: str, b_name: str
) -> str:
    flips = flip_table(runs_a, runs_b)
    lines = [
        f"**Flip table — {a_name} vs {b_name}** (per scenario, seeds collapsed by majority)",
        "",
        "| outcome | n | scenarios |",
        "|---|---|---|",
    ]
    for key, label in (
        ("both", "both pass"),
        ("a_only", f"{a_name} only"),
        ("b_only", f"{b_name} only"),
        ("neither", "both fail"),
    ):
        items = flips[key]
        names = ", ".join(f.scenario_id for f in items[:12]) or "—"
        lines.append(f"| {label} | {len(items)} | {names} |")

    b_wins, a_wins = len(flips["b_only"]), len(flips["a_only"])
    lines += [
        "",
        f"Net: {b_name} passes {b_wins} scenario{'' if b_wins == 1 else 's'} {a_name} fails; "
        f"{a_name} passes {a_wins} that {b_name} fails.",
    ]
    return "\n".join(lines)


def render(root: Path, compare_to: Path | None = None) -> str:
    runs = load_runs(root)
    if not runs:
        return f"No runs found under {root}."

    overall = aggregate(runs, label="ALL")
    spread = by_seed_spread(runs)
    surfaces = sorted({r["surface"] for r in runs})
    models = sorted({r["model"] for r in runs})
    splits = sorted({r["split"] for r in runs})

    parts = [
        f"## Results — `{root}`",
        "",
        f"- runs: **{len(runs)}**  ·  surface(s): {', '.join(surfaces)}  ·  model(s): {', '.join(models)}",
        f"- split(s): {', '.join(splits)}  ·  seeds: {sorted(spread['per_seed'])}",
        f"- overall task success: **{overall.task_success:.0%}** "
        f"(seed range {spread['min']:.0%} to {spread['max']:.0%}, spread {spread['spread']:.0%})",
        "",
        "### Per category",
        "",
        markdown_table([*by_category(runs), overall]),
        "",
        "### Category-specific metrics",
        "",
        special_metrics_table(overall),
    ]

    if compare_to is not None:
        other = load_runs(compare_to)
        if other:
            parts += ["", "### Paired comparison", "", flip_markdown(runs, other, root.name, compare_to.name)]

    parts += [
        "",
        "### Caveats",
        "",
        f"- A delta smaller than the {spread['spread']:.0%} seed spread is not a finding.",
        "- Result-token counts are the structured payload measured once, not the wire envelope.",
        "- Sampling is not pinned: current models reject `temperature`, "
        "so seeds are the only variance control.",
    ]
    return "\n".join(parts)


def failures(root: Path, limit: int = 20) -> str:
    runs = [r for r in load_runs(root) if not succeeded(r)]
    if not runs:
        return "No failures."
    lines = ["| scenario | seed | failed assertions |", "|---|---|---|"]
    for run in runs[:limit]:
        bad = "; ".join(f"{a['type']}: {a['detail']}" for a in run["assertions"] if not a["passed"])
        lines.append(f"| {run['scenario_id']} | {run['seed']} | {bad or 'rubric below threshold'} |")
    return "\n".join(lines)
