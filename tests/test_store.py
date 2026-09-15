from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from btmcp.core.costs import FRICTIONLESS, CostModel
from btmcp.core.engine import Engine
from btmcp.core.errors import RunNotFound
from btmcp.core.spec import parse_spec
from btmcp.core.store import RunStore
from btmcp.data.barstore import BarStore

START, END = dt.date(2019, 1, 1), dt.date(2021, 12, 31)
COSTS = CostModel(fee_bps=5.0, slippage_bps=2.0)
SPEC = {"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}}


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(BarStore())


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


def run(
    store: RunStore,
    engine: Engine,
    spec: dict | None = None,
    costs: CostModel = COSTS,
    symbol: str = "SYN-04",
):
    return store.execute(engine, parse_spec(spec or SPEC), symbol, START, END, costs)


def test_second_identical_run_is_cached_and_fast(store: RunStore, engine: Engine) -> None:
    """E2-T5 acceptance: a repeat config returns from cache in under 50 ms, identically."""
    first = run(store, engine)
    second = run(store, engine)
    assert first.cached is False and second.cached is True
    assert second.elapsed_ms < 50.0
    assert first.run_id == second.run_id
    assert first.metrics.model_dump() == second.metrics.model_dump()


def test_run_id_is_sensitive_to_every_input(store: RunStore, engine: Engine) -> None:
    base = run(store, engine).run_id
    assert run(store, engine, costs=CostModel(fee_bps=6.0, slippage_bps=2.0)).run_id != base
    assert (
        run(store, engine, spec={"primitive": "sma_cross", "params": {"fast": 21, "slow": 50}}).run_id != base
    )
    assert run(store, engine, symbol="SYN-01").run_id != base
    assert run(store, engine, costs=FRICTIONLESS).run_id != base


def test_artifacts_are_persisted_and_retrievable(store: RunStore, engine: Engine) -> None:
    summary = run(store, engine)
    equity = store.equity(summary.run_id)
    trades = store.trades(summary.run_id)
    assert equity.height == summary.metrics.n_bars + 1
    assert trades.height == summary.metrics.trade_count
    assert equity.columns == ["timestamp", "equity", "weight"]
    assert summary.equity_uri.endswith("/equity") and summary.blotter_uri.endswith("/blotter")


def test_summary_never_inlines_rows(store: RunStore, engine: Engine) -> None:
    """The summary is what crosses the tool boundary, so it must stay small."""
    summary = run(store, engine)
    payload = summary.model_dump_json()
    assert len(payload) < 6000, f"summary is {len(payload)} bytes"
    assert "equity" not in summary.model_dump(exclude={"equity_uri"})


def test_unknown_run_id_raises_typed_error(store: RunStore) -> None:
    with pytest.raises(RunNotFound) as exc:
        store.get("0" * 64)
    assert exc.value.code == "RUN_NOT_FOUND"
    assert exc.value.suggested_action


def test_cache_survives_a_new_store_instance(tmp_path: Path, engine: Engine) -> None:
    first = run(RunStore(tmp_path / "runs"), engine)
    second = run(RunStore(tmp_path / "runs"), engine)
    assert second.cached is True
    assert second.run_id == first.run_id


def test_run_id_is_stable_across_equivalent_spec_spellings(store: RunStore, engine: Engine) -> None:
    explicit = parse_spec({**SPEC, "warmup_bars": 0, "rebalance": "daily", "sizing": "full"})
    implicit = parse_spec(SPEC)
    a = store.execute(engine, explicit, "SYN-04", START, END, COSTS)
    b = store.execute(engine, implicit, "SYN-04", START, END, COSTS)
    assert a.run_id == b.run_id
