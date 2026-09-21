from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from btmcp.data.barstore import BarStore
from btmcp.server.app import AppConfig
from btmcp.server.guardrails.fencing import FencingConfig
from btmcp.server.surface_a import SYMBOL_UNIVERSE, build

AS_OF = "2021-12-31"
WINDOW = {"start": "2019-01-01", "end": "2021-12-31"}
SMA = {"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}}
HOLD = {"primitive": "buy_and_hold", "params": {}}
COSTS = {"fee_bps": 5, "slippage_bps": 2}
RUN_ARGS = {"spec": SMA, "symbol": "SYN-04", **WINDOW, "costs": COSTS}


def config(tmp_path: Path, **kw) -> AppConfig:
    return AppConfig(surface="a", runs_root=tmp_path / "runs", audit_path=tmp_path / "audit.jsonl", **kw)


@asynccontextmanager
async def surface(tmp_path: Path, **kw):
    """Inline rather than a pytest fixture.

    The Client's task group must be entered and exited in the same task, which an
    async-generator fixture does not guarantee; doing so raises an anyio cancel-scope
    error at teardown.
    """
    mcp, services = build(config(tmp_path, **kw))
    async with Client(mcp) as c:
        yield c, services


async def test_negotiates_the_modern_protocol_and_lists_the_surface(tmp_path: Path) -> None:
    """Asserted, not assumed: a legacy-mode client silently negotiates 2025-11-25 instead."""
    async with surface(tmp_path) as (c, _):
        assert c.protocol_version == "2026-07-28"
        tools = {t.name for t in (await c.list_tools()).tools}
    assert tools == {
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
    }


async def test_every_tool_declares_a_structured_output_schema(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        for tool in (await c.list_tools()).tools:
            assert tool.output_schema is not None, f"{tool.name} has no output schema"
            assert tool.description


async def test_resources_and_prompts_are_listed(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        resources = {str(r.uri) for r in (await c.list_resources()).resources}
        prompts = {p.name for p in (await c.list_prompts()).prompts}
    assert len(resources) == 3
    assert prompts == {"evaluate-strategy-idea", "diagnose-a-bad-backtest"}


async def test_as_of_is_required_on_every_data_tool(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        for tool in (await c.list_tools()).tools:
            if tool.name in {"list_symbols", "describe_series", "get_bars", "get_news", "validate_strategy"}:
                assert "as_of" in tool.input_schema.get("required", []), f"{tool.name} lets as_of be omitted"


async def test_get_bars_never_returns_rows_and_respects_as_of(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        result = await c.call_tool(
            "get_bars",
            {"symbols": ["SYN-04"], "start": "2019-01-01", "end": "2021-12-31", "as_of": "2020-06-30"},
        )
    payload = result.structured_content
    assert payload["n_rows"] > 0
    assert payload["per_symbol"][0]["last"] <= "2020-06-30"
    assert "rows" not in payload
    assert payload["resources"]


async def test_run_backtest_returns_summary_and_links(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        payload = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
    assert payload["metrics"]["sharpe"] > 1.0
    assert len(payload["resources"]) == 2
    assert payload["cached"] is False


async def test_identical_run_is_served_from_cache(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        first = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
        second = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
    assert first["cached"] is False and second["cached"] is True
    assert first["run_id"] == second["run_id"]


async def test_costs_cannot_be_omitted(tmp_path: Path) -> None:
    """The SDK rejects the call before the handler runs, so this never reaches the engine."""
    async with surface(tmp_path) as (c, _):
        result = await c.call_tool("run_backtest", {"spec": SMA, "symbol": "SYN-04", **WINDOW})
    assert result.is_error
    assert "costs" in str(result.content)


async def test_invalid_spec_returns_an_actionable_typed_error(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        with pytest.raises(MCPError) as exc:
            await c.call_tool(
                "run_backtest", {**RUN_ARGS, "spec": {"primitive": "sma_crossover", "params": {}}}
            )
    data = exc.value.error.data
    assert data["code"] == "INVALID_SPEC"
    assert "sma_cross" in data["suggested_action"]
    assert "Traceback" not in str(data)


async def test_news_is_fenced_and_passed_through_intact(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        payload = (
            await c.call_tool(
                "get_news", {"symbol": "SYN-04", "start": "2018-01-01", "end": "2025-12-31", "as_of": AS_OF}
            )
        ).structured_content
    assert payload["fenced"] is True
    assert "UNTRUSTED_DATA" in payload["text"]
    assert "not instructions" in payload["text"]
    assert "publish_run_report" in payload["text"], "payload must pass through intact, not be stripped"


async def test_fencing_ablation_disables_the_wrapper(tmp_path: Path) -> None:
    async with surface(tmp_path, fencing=FencingConfig(enabled=False)) as (c, _):
        payload = (
            await c.call_tool(
                "get_news", {"symbol": "SYN-04", "start": "2018-01-01", "end": "2025-12-31", "as_of": AS_OF}
            )
        ).structured_content
    assert payload["fenced"] is False
    assert "UNTRUSTED_DATA" not in payload["text"]


async def test_read_tier_refuses_run_backtest_with_an_actionable_message(tmp_path: Path) -> None:
    async with surface(tmp_path, tier="read") as (c, _):
        assert await c.call_tool("list_symbols", {"as_of": AS_OF})
        with pytest.raises(MCPError) as exc:
            await c.call_tool("run_backtest", RUN_ARGS)
    data = exc.value.error.data
    assert data["code"] == "TIER_DENIED"
    assert data["context"]["required_tier"] == "simulate"
    assert data["context"]["granted_tier"] == "read"


async def test_publish_is_denied_below_the_act_tier(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        with pytest.raises(MCPError) as exc:
            await c.call_tool("publish_run_report", {"run_id": "x" * 64})
    assert exc.value.error.data["code"] == "TIER_DENIED"


async def test_audit_log_records_every_call(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, services):
        await c.call_tool("list_symbols", {"as_of": AS_OF})
        await c.call_tool("describe_series", {"symbol": "SYN-04", "as_of": AS_OF})
        # "NOPE" is outside the symbol enum, so the SDK rejects it before the handler
        # runs and reports an `is_error` result rather than raising MCPError.
        rejected = await c.call_tool(
            "get_bars", {"symbols": ["NOPE"], "start": "2019-01-01", "end": "2020-01-01", "as_of": AS_OF}
        )
        assert rejected.is_error
        records = list(services.audit.records)
        audit_path = services.audit.path

    assert [r["tool"] for r in records] == ["list_symbols", "describe_series", "get_bars"]
    assert records[0]["outcome"] == "ok"
    assert records[-1]["outcome"] == "error"
    assert all(set(r) >= {"ts", "tool", "surface", "tier", "arg_hash", "result_tokens"} for r in records)
    assert audit_path.exists()


async def test_every_result_stays_within_the_token_budget(tmp_path: Path) -> None:
    """E3-T1 acceptance, measured on the wire over the largest legal request."""
    async with surface(tmp_path) as (c, services):
        await c.call_tool(
            "get_bars",
            {
                "symbols": [f"SYN-{i:02d}" for i in range(1, 21)],
                "start": "2018-01-01",
                "end": "2025-12-31",
                "as_of": "2025-12-31",
            },
        )
        await c.call_tool(
            "get_news", {"symbol": "SYN-04", "start": "2018-01-01", "end": "2025-12-31", "as_of": AS_OF}
        )
        await c.call_tool("run_backtest", RUN_ARGS)
        await c.call_tool("list_symbols", {"as_of": AS_OF})
        records = list(services.audit.records)
        budget = services.config.budget_tokens

    for record in records:
        assert record["result_tokens"] <= budget, (
            f"{record['tool']} returned {record['result_tokens']} tokens"
        )


async def test_compare_runs_uses_the_cache_without_recomputing(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, services):
        a = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
        b = (await c.call_tool("run_backtest", {**RUN_ARGS, "spec": HOLD})).structured_content
        before = len(list(services.runs.root.iterdir()))
        diff = (await c.call_tool("compare_runs", {"run_ids": [a["run_id"], b["run_id"]]})).structured_content
        after = len(list(services.runs.root.iterdir()))
    assert after == before, "compare_runs recomputed a run"
    assert len(diff["rows"]) == 2
    assert diff["best_by_sharpe"] in {a["run_id"], b["run_id"]}


@pytest.mark.parametrize("trade_filter", ["worst_5", "best_5", "largest_drawdown", "all_in_window"])
async def test_explain_trades_filters(tmp_path: Path, trade_filter: str) -> None:
    async with surface(tmp_path) as (c, _):
        run = (await c.call_tool("run_backtest", RUN_ARGS)).structured_content
        payload = (
            await c.call_tool("explain_trades", {"run_id": run["run_id"], "filter": trade_filter})
        ).structured_content
    assert payload["filter"] == trade_filter
    assert payload["n_selected"] >= 0


async def test_unknown_run_id_is_a_typed_error(tmp_path: Path) -> None:
    async with surface(tmp_path) as (c, _):
        with pytest.raises(MCPError) as exc:
            await c.call_tool("get_run", {"run_id": "0" * 64})
    assert exc.value.error.data["code"] == "RUN_NOT_FOUND"


async def test_lookahead_cannot_be_obtained_through_any_tool(tmp_path: Path) -> None:
    """The headline guarantee, exercised across the whole surface rather than the store."""
    async with surface(tmp_path) as (c, _):
        bars = (
            await c.call_tool(
                "get_bars",
                {"symbols": ["SYN-04"], "start": "2019-01-01", "end": "2025-12-31", "as_of": "2020-01-02"},
            )
        ).structured_content
        profile = (
            await c.call_tool("describe_series", {"symbol": "SYN-04", "as_of": "2020-01-02"})
        ).structured_content
        news = (
            await c.call_tool(
                "get_news",
                {"symbol": "SYN-04", "start": "2018-01-01", "end": "2025-12-31", "as_of": "2020-01-02"},
            )
        ).structured_content

    assert bars["per_symbol"][0]["last"] <= "2020-01-02"
    assert profile["last"] <= "2020-01-02"
    assert "2020-03-12" not in news["text"], "a later-dated headline leaked past as_of"


def test_the_symbol_enum_matches_the_dataset() -> None:
    """X3 change 2 turned `symbol` into an enum read from the pinned manifest.

    The read has a `str` fallback for a tree with no generated dataset. That fallback
    would silently disarm the change rather than fail, and a sweep run against it
    would look like a null result instead of a broken one - so assert the universe is
    both non-empty and exactly the store's.
    """
    assert SYMBOL_UNIVERSE, "symbol enum is empty; the manifest was not read"
    assert list(SYMBOL_UNIVERSE) == sorted(BarStore().universe)
