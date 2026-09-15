from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import ElicitResult

from btmcp.eval.scenario import load_scenarios
from btmcp.eval.surfaces import BINDINGS, resolve
from btmcp.server.app import AppConfig
from btmcp.server.surface_a import build as build_a
from btmcp.server.surface_b import build as build_b

AS_OF = "2021-12-31"
SMA = {"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}}
COSTS = {"fee_bps": 5, "slippage_bps": 2}
RUN_ARGS = {"spec": SMA, "symbol": "SYN-04", "start": "2019-01-01", "end": "2021-12-31", "costs": COSTS}


def config(tmp_path: Path, **kw) -> AppConfig:
    return AppConfig(surface="b", runs_root=tmp_path / "runs", audit_path=tmp_path / "audit.jsonl", **kw)


async def _accept_nothing(context, params) -> ElicitResult:
    return ElicitResult(action="decline")


@asynccontextmanager
async def surface(tmp_path: Path, **kw):
    mcp, services = build_b(config(tmp_path, **kw))
    async with Client(mcp, elicitation_callback=_accept_nothing) as c:
        yield c, services


async def test_surface_b_exposes_four_tools_plus_the_act_tool(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        tools = {t.name for t in (await c.list_tools()).tools}
    assert tools == {"explore_data", "run_backtest", "inspect_run", "compare_runs", "publish_run_report"}


async def test_both_surfaces_expose_the_same_resources_and_prompts(tmp_path: Path) -> None:
    mcp_a, _ = build_a(AppConfig(surface="a", runs_root=tmp_path / "a"))
    mcp_b, _ = build_b(AppConfig(surface="b", runs_root=tmp_path / "b"))
    async with Client(mcp_a) as ca, Client(mcp_b) as cb:
        assert {str(r.uri) for r in (await ca.list_resources()).resources} == {
            str(r.uri) for r in (await cb.list_resources()).resources
        }
        assert {p.name for p in (await ca.list_prompts()).prompts} == {
            p.name for p in (await cb.list_prompts()).prompts
        }


@pytest.mark.parametrize(
    ("mode", "args"),
    [
        ("symbols", {}),
        ("describe", {"symbol": "SYN-04"}),
        ("bars", {"symbols": ["SYN-04"], "start": "2019-01-01", "end": "2021-12-31"}),
        ("news", {"symbol": "SYN-04", "start": "2018-01-01", "end": "2021-12-31"}),
    ],
)
async def test_explore_data_covers_every_read(tmp_path: Path, mode: str, args: dict) -> None:
    async with surface(tmp_path) as (c, _):
        result = await c.call_tool("explore_data", {"mode": mode, "as_of": AS_OF, **args})
    assert result.structured_content


async def test_explore_data_names_the_arguments_a_mode_needs(tmp_path: Path) -> None:
    """Consolidation moves this error from the schema into the handler; it must stay actionable."""
    async with surface(tmp_path) as (c, _):
        with pytest.raises(MCPError) as exc:
            await c.call_tool("explore_data", {"mode": "bars", "as_of": AS_OF})
    data = exc.value.error.data
    assert data["code"] == "INVALID_SPEC"
    assert "symbols" in data["cause"] and "start" in data["cause"]
    assert "mode='bars'" in data["suggested_action"]


async def test_as_of_is_required_on_explore_data(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        tool = next(t for t in (await c.list_tools()).tools if t.name == "explore_data")
    assert "as_of" in tool.input_schema.get("required", [])


async def test_run_backtest_elicits_rather_than_erroring_on_missing_costs(tmp_path: Path) -> None:
    """The designed difference from Surface A, and the thing X2 exists to measure.

    Legacy mode: elicitation needs a back-channel for server-initiated requests, which
    only the handshake protocol provides.
    """
    elicited: list[str] = []

    async def record(context, params) -> ElicitResult:
        elicited.append(params.message)
        return ElicitResult(action="decline")

    mcp, _ = build_b(config(tmp_path))
    async with Client(mcp, mode="legacy", elicitation_callback=record) as c:
        assert c.protocol_version == "2025-11-25"
        result = await c.call_tool(
            "run_backtest", {"spec": SMA, "symbol": "SYN-04", "start": "2019-01-01", "end": "2021-12-31"}
        )
    assert elicited and "cost model" in elicited[0]
    assert result.structured_content["ok"] is False
    assert result.structured_content["code"] == "MISSING_COST_MODEL"


async def test_run_backtest_elicits_for_a_missing_period(tmp_path: Path) -> None:
    elicited: list[str] = []

    async def record(context, params) -> ElicitResult:
        elicited.append(params.message)
        return ElicitResult(action="decline")

    mcp, _ = build_b(config(tmp_path))
    async with Client(mcp, mode="legacy", elicitation_callback=record) as c:
        await c.call_tool("run_backtest", {"spec": SMA, "symbol": "SYN-04", "costs": COSTS})
    assert elicited and "window" in elicited[0]


async def test_elicitation_degrades_to_a_typed_error_without_a_back_channel(tmp_path: Path) -> None:
    """The stateless 2026-07-28 envelope cannot carry server-initiated requests.

    A server may not assume its client offers a back-channel, so the unavailable case
    must become the same actionable error, not a NoBackChannelError escaping the tool.
    """
    mcp, _ = build_b(config(tmp_path))
    async with Client(mcp) as c:
        assert c.protocol_version == "2026-07-28"
        result = await c.call_tool(
            "run_backtest", {"spec": SMA, "symbol": "SYN-04", "start": "2019-01-01", "end": "2021-12-31"}
        )
    payload = result.structured_content
    assert not result.is_error
    assert payload["code"] == "MISSING_COST_MODEL"
    assert "pass costs explicitly" in payload["suggested_action"]
    assert "cannot prompt" in payload["suggested_action"]


async def test_surface_a_publish_confirmation_also_needs_a_back_channel(tmp_path: Path) -> None:
    """E4-T1's confirmation gate, exercised rather than only denied by tier."""
    asked: list[str] = []

    async def decline(context, params) -> ElicitResult:
        asked.append(params.message)
        return ElicitResult(action="decline")

    mcp, _ = build_a(AppConfig(surface="a", tier="act", runs_root=tmp_path / "runs"))
    async with Client(mcp, mode="legacy", elicitation_callback=decline) as c:
        run = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
        result = await c.call_tool("publish_run_report", {"run_id": run["run_id"]})

    assert asked and "Publish the report" in asked[0]
    assert result.structured_content["published"] is False
    assert result.structured_content["reason"] == "confirmation not granted"


async def test_surface_a_publishes_only_on_explicit_confirmation(tmp_path: Path) -> None:
    async def accept(context, params) -> ElicitResult:
        return ElicitResult(action="accept", content={"confirm": True, "audience": "internal"})

    mcp, _ = build_a(AppConfig(surface="a", tier="act", runs_root=tmp_path / "runs"))
    async with Client(mcp, mode="legacy", elicitation_callback=accept) as c:
        run = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
        result = await c.call_tool("publish_run_report", {"run_id": run["run_id"]})
    assert result.structured_content["published"] is True
    assert result.structured_content["audience"] == "internal"


async def test_validation_is_folded_into_run_backtest(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        with pytest.raises(MCPError) as exc:
            await c.call_tool(
                "run_backtest", {**RUN_ARGS, "spec": {"primitive": "sma_crossover", "params": {}}}
            )
    assert exc.value.error.data["code"] == "INVALID_SPEC"


async def test_both_surfaces_produce_the_same_run_id(tmp_path: Path) -> None:
    """Same core, so an identical configuration must be the same content-addressed run."""
    mcp_a, _ = build_a(AppConfig(surface="a", runs_root=tmp_path / "shared"))
    mcp_b, _ = build_b(AppConfig(surface="b", runs_root=tmp_path / "shared"))
    async with Client(mcp_a) as ca:
        a = (await ca.call_tool("run_backtest", RUN_ARGS)).structured_content
    async with Client(mcp_b, elicitation_callback=_accept_nothing) as cb:
        b = (await cb.call_tool("run_backtest", RUN_ARGS)).structured_content
    assert a["run_id"] == b["run_id"]
    assert a["metrics"]["sharpe"] == b["metrics"]["sharpe"]
    assert b["cached"] is True, "the second surface should hit the shared cache"


async def test_inspect_run_views(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        run = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
        summary = (await c.call_tool("inspect_run", {"run_id": run["run_id"]})).structured_content
        trades = (
            await c.call_tool("inspect_run", {"run_id": run["run_id"], "view": "trades", "filter": "best_5"})
        ).structured_content
        drawdowns = (
            await c.call_tool("inspect_run", {"run_id": run["run_id"], "view": "drawdowns"})
        ).structured_content
    assert summary["metrics"]["sharpe"] > 1.0
    assert trades["filter"] == "best_5"
    assert drawdowns["filter"] == "largest_drawdown"


async def test_tiers_and_fencing_apply_identically_on_b(tmp_path: Path) -> None:
    async with surface(tmp_path, tier="read") as (c, _):
        with pytest.raises(MCPError) as exc:
            await c.call_tool("run_backtest", RUN_ARGS)
    assert exc.value.error.data["code"] == "TIER_DENIED"

    async with surface(tmp_path) as (c, _):
        news = (
            await c.call_tool(
                "explore_data",
                {
                    "mode": "news",
                    "symbol": "SYN-04",
                    "start": "2018-01-01",
                    "end": "2025-12-31",
                    "as_of": AS_OF,
                },
            )
        ).structured_content
    assert news["fenced"] is True and "UNTRUSTED_DATA" in news["text"]


async def test_surface_b_stays_within_the_token_budget(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, services):
        await c.call_tool(
            "explore_data",
            {
                "mode": "bars",
                "symbols": [f"SYN-{i:02d}" for i in range(1, 21)],
                "start": "2018-01-01",
                "end": "2025-12-31",
                "as_of": "2025-12-31",
            },
        )
        await c.call_tool("run_backtest", RUN_ARGS)
        records = list(services.audit.records)
        budget = services.config.budget_tokens
    for record in records:
        assert record["result_tokens"] <= budget, f"{record['tool']} returned {record['result_tokens']}"


# ---- E6-T1 acceptance -----------------------------------------------------


def test_every_capability_a_scenario_asserts_is_bound_on_both_surfaces() -> None:
    """E6-T1 acceptance: all 40 scenarios must be runnable against B unmodified."""
    asserted = {
        assertion.tool
        for scenario in load_scenarios()
        for assertion in scenario.assertions
        if getattr(assertion, "tool", None)
    }
    for surface_name in ("a", "b"):
        unbound = [cap for cap in asserted if cap not in BINDINGS[surface_name]]
        assert not unbound, f"surface {surface_name} has no binding for {unbound}"


async def test_bindings_point_at_tools_that_actually_exist(tmp_path: Path) -> None:
    for surface_name, build in (("a", build_a), ("b", build_b)):
        mcp, _ = build(AppConfig(surface=surface_name, runs_root=tmp_path / surface_name))
        async with Client(mcp) as c:
            live = {t.name for t in (await c.list_tools()).tools}
        for capability, binding in BINDINGS[surface_name].items():
            assert binding.tool in live, f"{surface_name}: {capability} -> missing tool {binding.tool}"


def test_binding_discriminates_consolidated_modes() -> None:
    bars = resolve("b", "get_bars")
    news = resolve("b", "get_news")
    assert bars.tool == news.tool == "explore_data"
    assert bars.matches("explore_data", {"mode": "bars"})
    assert not bars.matches("explore_data", {"mode": "news"})
    assert news.matches("explore_data", {"mode": "news"})
    assert resolve("a", "get_bars").matches("get_bars", {})


async def test_a_vocabulary_assertions_pass_against_b_tool_calls(tmp_path: Path) -> None:
    """The binding is what makes X2 a comparison of granularity rather than of naming."""
    from btmcp.eval.events import ToolCall, Turn
    from btmcp.eval.models import ScriptedModel
    from btmcp.eval.runner import run_scenario
    from btmcp.eval.scenario import Scenario

    scenario = Scenario.model_validate(
        {
            "id": "bind-01",
            "category": "happy_path",
            "split": "dev",
            "tier": "simulate",
            "prompt": "x",
            "assertions": [
                {"type": "tool_called", "tool": "get_bars"},
                {"type": "arg_matches", "tool": "get_bars", "path": "as_of", "equals": "2021-12-31"},
                {"type": "no_tool_called", "tool": "get_news"},
            ],
        }
    )
    model = ScriptedModel(
        [
            Turn(
                tool_calls=[
                    ToolCall(
                        "t1",
                        "explore_data",
                        {
                            "mode": "bars",
                            "symbols": ["SYN-04"],
                            "start": "2019-01-01",
                            "end": "2021-12-31",
                            "as_of": "2021-12-31",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            Turn(text="done"),
        ]
    )
    outcome = await run_scenario(scenario, build_b, model, runs_root=tmp_path / "runs", surface_name="b")
    assert outcome.task_success, outcome.assertions


async def test_the_binding_does_not_confuse_two_modes_of_one_tool(tmp_path: Path) -> None:
    from btmcp.eval.events import ToolCall, Turn
    from btmcp.eval.models import ScriptedModel
    from btmcp.eval.runner import run_scenario
    from btmcp.eval.scenario import Scenario

    scenario = Scenario.model_validate(
        {
            "id": "bind-02",
            "category": "happy_path",
            "split": "dev",
            "tier": "simulate",
            "prompt": "x",
            "assertions": [{"type": "tool_called", "tool": "get_bars"}],
        }
    )
    # The model reads news, not bars. A binding that matched on tool name alone would
    # score this as a pass because both are `explore_data`.
    model = ScriptedModel(
        [
            Turn(
                tool_calls=[
                    ToolCall(
                        "t1",
                        "explore_data",
                        {
                            "mode": "news",
                            "symbol": "SYN-04",
                            "start": "2019-01-01",
                            "end": "2021-12-31",
                            "as_of": "2021-12-31",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            Turn(text="done"),
        ]
    )
    outcome = await run_scenario(scenario, build_b, model, runs_root=tmp_path / "runs", surface_name="b")
    assert not outcome.task_success
