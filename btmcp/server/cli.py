from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from btmcp.server.app import AppConfig
from btmcp.server.guardrails.auth import TIERS
from btmcp.server.guardrails.fencing import FencingConfig
from btmcp.server.shaping import DEFAULT_BUDGET_TOKENS


def build_config(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        prog="btmcp-serve", description="Run the backtesting MCP server over stdio."
    )
    parser.add_argument("--surface", choices=["a", "b"], default="a")
    parser.add_argument("--tier", choices=list(TIERS), default="simulate")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--audit", type=Path, default=None, help="JSONL audit log path")
    parser.add_argument("--fencing", choices=["on", "off"], default="on", help="X4 ablation switch")
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    args = parser.parse_args(argv)

    return AppConfig(
        surface=args.surface,
        tier=args.tier,
        dataset_root=args.dataset,
        runs_root=args.runs,
        audit_path=args.audit,
        fencing=FencingConfig(enabled=args.fencing == "on"),
        budget_tokens=args.budget_tokens,
    )


def load_surface(surface: str) -> Callable[[AppConfig], tuple[Any, Any]]:
    if surface == "a":
        from btmcp.server.surface_a import build

        return build

    from btmcp.server.surface_b import build as build_b

    return build_b


def main(argv: list[str] | None = None) -> int:
    config = build_config(argv)
    mcp, _ = load_surface(config.surface)(config)
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
