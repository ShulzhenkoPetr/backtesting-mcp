from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.shared.exceptions import MCPError

from btmcp.core.errors import TierDenied

TIERS: tuple[str, ...] = ("read", "simulate", "act")
TIER_RANK = {tier: i for i, tier in enumerate(TIERS)}

TOOL_TIERS: dict[str, str] = {
    # read
    "list_symbols": "read",
    "describe_series": "read",
    "get_bars": "read",
    "get_news": "read",
    "explore_data": "read",
    "get_run": "read",
    "compare_runs": "read",
    "explain_trades": "read",
    "inspect_run": "read",
    # simulate
    "validate_strategy": "simulate",
    "run_backtest": "simulate",
    # act
    "publish_run_report": "act",
}


@dataclass(frozen=True)
class TierPolicy:
    granted: str = "simulate"

    def allows(self, tool: str) -> bool:
        required = TOOL_TIERS.get(tool, "act")
        return TIER_RANK[required] <= TIER_RANK[self.granted]

    def required_for(self, tool: str) -> str:
        return TOOL_TIERS.get(tool, "act")


def denial(tool: str, policy: TierPolicy) -> TierDenied:
    required = policy.required_for(tool)
    return TierDenied(
        cause=(
            f"tool '{tool}' requires the '{required}' tier but this server was started at '{policy.granted}'"
        ),
        suggested_action=(
            f"restart the server with --tier {required}, or use a tool available at "
            f"'{policy.granted}'. Tiers escalate read -> simulate -> act."
        ),
        required_tier=required,
        granted_tier=policy.granted,
        tool=tool,
    )


class TierMiddleware:
    """Enforces permission tiers before parameter validation.

    Implemented as middleware rather than as a decorator on each tool so that a new
    tool cannot be added without a tier: an unmapped name defaults to `act`, the
    tier almost nothing is granted, so forgetting to classify a tool fails closed.
    """

    def __init__(self, policy: TierPolicy) -> None:
        self.policy = policy

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        if ctx.method == "tools/call":
            tool = _tool_name(ctx.params)
            if tool and not self.policy.allows(tool):
                error = denial(tool, self.policy)
                raise MCPError(code=-32003, message=error.cause, data=error.to_payload())
        return await call_next(ctx)


def _tool_name(params: Any) -> str | None:
    if isinstance(params, dict):
        name = params.get("name")
        return str(name) if name else None
    return getattr(params, "name", None)


__all__ = ["TIERS", "TOOL_TIERS", "TierMiddleware", "TierPolicy", "denial", "replace"]
