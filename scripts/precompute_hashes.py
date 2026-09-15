from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import yaml

from btmcp.core.costs import CostModel
from btmcp.core.spec import parse_spec
from btmcp.core.store import RunStore
from btmcp.data.manifest import DATASET_ID

SCENARIOS = Path("btmcp/eval/scenarios")


def expected_hash(spec_block: dict[str, Any]) -> str:
    return RunStore.run_id(
        spec=parse_spec(spec_block["spec"]),
        symbol=spec_block["symbol"],
        start=dt.date.fromisoformat(spec_block["start"]),
        end=dt.date.fromisoformat(spec_block["end"]),
        costs=CostModel.model_validate(spec_block["costs"]),
        dataset_id=spec_block.get("dataset_id", DATASET_ID),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=SCENARIOS)
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting")
    args = parser.parse_args()

    drifted: list[str] = []
    updated = 0
    for path in sorted(args.root.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        changed = False
        for assertion in data.get("assertions", []):
            if assertion.get("type") != "config_hash_reached" or "spec" not in assertion:
                continue
            value = expected_hash(assertion["spec"])
            if assertion.get("value") != value:
                if args.check:
                    drifted.append(
                        f"{data['id']}: expected {value[:12]}, file has {str(assertion.get('value'))[:12]}"
                    )
                else:
                    assertion["value"] = value
                    changed = True
        if changed:
            path.write_text(yaml.safe_dump(data, sort_keys=False, width=100, allow_unicode=True))
            updated += 1

    if drifted:
        print("config hash drift:\n  " + "\n  ".join(drifted))
        return 1
    total = len(list(args.root.glob("*.yaml")))
    print(f"{'checked' if args.check else 'updated'} {total if args.check else updated} scenario(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
