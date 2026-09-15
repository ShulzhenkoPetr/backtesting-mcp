from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolBinding:
    tool: str
    where: dict[str, Any] = field(default_factory=dict)

    def matches(self, tool: str, arguments: dict[str, Any]) -> bool:
        if tool != self.tool:
            return False
        return all(arguments.get(key) == value for key, value in self.where.items())


def _identity(*names: str) -> dict[str, ToolBinding]:
    return {name: ToolBinding(name) for name in names}


SURFACE_A = _identity(
    "list_symbols",
    "describe_series",
    "get_bars",
    "get_news",
    "validate_strategy",
    "run_backtest",
    "get_run",
    "compare_runs",
    "explain_trades",
    "publish_run_report",
)

SURFACE_B: dict[str, ToolBinding] = {
    "list_symbols": ToolBinding("explore_data", {"mode": "symbols"}),
    "describe_series": ToolBinding("explore_data", {"mode": "describe"}),
    "get_bars": ToolBinding("explore_data", {"mode": "bars"}),
    "get_news": ToolBinding("explore_data", {"mode": "news"}),
    "validate_strategy": ToolBinding("run_backtest"),
    "run_backtest": ToolBinding("run_backtest"),
    "get_run": ToolBinding("inspect_run", {"view": "summary"}),
    "compare_runs": ToolBinding("compare_runs"),
    "explain_trades": ToolBinding("inspect_run"),
    "publish_run_report": ToolBinding("publish_run_report"),
}

BINDINGS: dict[str, dict[str, ToolBinding]] = {"a": SURFACE_A, "b": SURFACE_B}


def resolve(surface: str, capability: str) -> ToolBinding:
    """Maps a scenario's capability name onto the tool that implements it on a surface.

    Scenarios are written once, in Surface A's vocabulary, and run unmodified against
    both surfaces. Without this, every consolidated tool on B would fail assertions
    that name A's tools and X2 would measure nothing but the rename.
    """
    return BINDINGS.get(surface, SURFACE_A).get(capability, ToolBinding(capability))
