from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from mcp import Client

from btmcp.eval.rendering import anthropic_tools, openai_tools
from btmcp.eval.runner import tool_specs
from btmcp.eval.schema import DIALECTS, is_empty_object
from btmcp.server.app import AppConfig
from btmcp.server.cli import load_surface
from btmcp.server.guardrails.fencing import FencingConfig


def walk(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    """Every property-less object in the rendered schema, with its path."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, list):
        for index, item in enumerate(node):
            found += walk(item, f"{path}[{index}]")
    elif isinstance(node, dict):
        if is_empty_object(node):
            found.append((path, node))
        for key, value in node.items():
            found += walk(value, f"{path}.{key}")
    return found


async def probe(args: argparse.Namespace) -> int:
    config = AppConfig(
        surface=args.surface,
        tier=args.tier,
        runs_root=Path(args.runs),
        fencing=FencingConfig(enabled=True),
    )
    mcp, _ = load_surface(args.surface)(config)
    async with Client(mcp, mode="legacy") as client:
        specs = tool_specs((await client.list_tools()).tools)

    dialect = DIALECTS[args.dialect]
    render = anthropic_tools if args.dialect == "anthropic" else openai_tools
    rendered = render(specs, dialect)

    print(f"surface {args.surface} | dialect {dialect.name} | {len(rendered)} tools\n")
    print(f"  {'':<5}{'tool':<24}{'schema':>8}{'descr':>8}  (chars)")
    problems = 0
    schema_chars = descr_chars = 0
    for tool, payload in zip(specs, rendered, strict=True):
        schema = payload.get("input_schema") or payload["function"]["parameters"]
        empties = walk(schema, f"{tool.name}")
        strict = payload.get("function", {}).get("strict", False)
        flag = "FAIL" if empties else "ok  "
        schema_chars += len(json.dumps(schema))
        descr_chars += len(tool.description)
        print(
            f"  {flag} {tool.name:<24}{len(json.dumps(schema)):>8}{len(tool.description):>8}  strict={strict}"
        )
        for path, node in empties:
            problems += 1
            print(f"         property-less object at {path}: {json.dumps(node)}")

    # Descriptions are re-sent on every turn, so a change to them has a per-turn token
    # price. X3 moves exactly this number; printing it makes the cost column of the
    # change table measurable offline rather than inferred from a sweep.
    print(
        f"\n  {'TOTAL':<29}{schema_chars:>8}{descr_chars:>8}"
        f"   ~{(schema_chars + descr_chars) // 4} tokens of tool block per turn"
    )

    if args.show:
        chosen = next((p for s, p in zip(specs, rendered, strict=True) if s.name == args.show), None)
        if chosen is None:
            print(f"\nno tool named {args.show!r}; known: {', '.join(s.name for s in specs)}")
            return 1
        print(f"\n--- {args.show} as sent ---")
        print(json.dumps(chosen, indent=2))

    if problems:
        print(
            f"\n{problems} property-less object(s). Cerebras rejects these with "
            "\"Object fields require at least one of: 'properties' or 'anyOf'\"."
        )
        return 1
    print("\nNothing obviously unsendable.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--surface", choices=["a", "b"], default="a")
    parser.add_argument("--dialect", choices=sorted(DIALECTS), default="cerebras")
    parser.add_argument("--tier", choices=["read", "simulate", "act"], default="simulate")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--show", default=None, help="dump one tool's full rendered payload")
    args = parser.parse_args(argv)
    return asyncio.run(probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
