from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(records: list[dict[str, Any]]) -> str:
    if not records:
        return "empty audit log"
    lines = [
        f"{'#':>3}  {'tool':<22} {'tier':<9} {'outcome':<7} {'ms':>8} {'tokens':>7}  error",
        "-" * 78,
    ]
    for i, r in enumerate(records, 1):
        lines.append(
            f"{i:>3}  {r['tool']:<22} {r['tier']:<9} {r['outcome']:<7} "
            f"{r['latency_ms']:>8.1f} {r['result_tokens']:>7}  {r.get('error_code') or ''}"
        )

    total_ms = sum(r["latency_ms"] for r in records)
    total_tokens = sum(r["result_tokens"] for r in records)
    errors = [r for r in records if r["outcome"] != "ok"]
    tokens = sorted(r["result_tokens"] for r in records)
    p99 = tokens[min(len(tokens) - 1, int(len(tokens) * 0.99))]

    lines += [
        "-" * 78,
        f"{len(records)} calls, {len(errors)} errors, {total_ms:.0f} ms, {total_tokens} result tokens",
        f"p99 result tokens: {p99}",
        f"surface: {records[0]['surface']}  dataset: {records[0]['dataset_id']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("audit", type=Path, help="path to an audit.jsonl")
    args = parser.parse_args()
    print(render(load(args.audit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
