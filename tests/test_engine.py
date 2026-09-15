from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from btmcp.core.costs import FRICTIONLESS, CostModel
from btmcp.core.engine import BarView, Engine, TargetPosition
from btmcp.core.errors import InsufficientHistory, LookaheadError
from btmcp.core.spec import parse_spec
from btmcp.data.barstore import BarStore

START, END = dt.date(2019, 1, 1), dt.date(2021, 12, 31)
COSTS = CostModel(fee_bps=5.0, slippage_bps=2.0)


@pytest.fixture(scope="module")
def store() -> BarStore:
    return BarStore()


@pytest.fixture(scope="module")
def engine(store: BarStore) -> Engine:
    return Engine(store)


def sma(fast: int = 20, slow: int = 50):
    return parse_spec({"primitive": "sma_cross", "params": {"fast": fast, "slow": slow}})


def hold():
    return parse_spec({"primitive": "buy_and_hold", "params": {}})


class PeekingStrategy:
    """Actively tries to defeat the point-in-time rule from inside the engine."""

    def __init__(self) -> None:
        self.attempted = False

    def decide(self, t: dt.date, view: BarView) -> TargetPosition:
        self.attempted = True
        view.bars(as_of=t + dt.timedelta(days=1))
        return TargetPosition(1.0)


def test_a_strategy_reaching_past_its_decision_date_is_refused(engine: Engine, store: BarStore) -> None:
    frame = store.bars(["SYN-04"], START, END, as_of=END)
    t = frame["timestamp"][100]
    view = BarView(frame.head(101), t)
    strategy = PeekingStrategy()
    with pytest.raises(LookaheadError) as exc:
        strategy.decide(t, view)
    assert strategy.attempted
    assert exc.value.code == "LOOKAHEAD_REFUSED"
    assert exc.value.context["decision_date"] == t.isoformat()


def test_barview_never_exposes_a_bar_after_its_timestamp(store: BarStore) -> None:
    frame = store.bars(["SYN-04"], START, END, as_of=END)
    t = frame["timestamp"][200]
    view = BarView(frame.head(201), t)
    assert view.bars()["timestamp"].max() == t
    assert view.as_of == t
    assert len(view.closes(lookback=10)) == 10
    assert view.bars(as_of=t - dt.timedelta(days=30))["timestamp"].max() <= t


def test_fills_never_use_the_deciding_bars_prices(engine: Engine, store: BarStore) -> None:
    """E2-T2 acceptance: execution is at the next bar's open, so a same-bar fill is impossible."""
    result = engine.run(sma(), "SYN-04", START, END, FRICTIONLESS)
    frame = store.bars(["SYN-04"], START, END, as_of=END)
    lookup = dict(zip(frame["timestamp"].to_list(), frame["open"].to_list(), strict=True))
    closes = dict(zip(frame["timestamp"].to_list(), frame["close"].to_list(), strict=True))

    assert result.trades
    for trade in result.trades:
        assert trade.price == pytest.approx(lookup[trade.timestamp]), "fill was not at the bar open"
        assert trade.price != pytest.approx(closes[trade.timestamp]), "fill used a same-bar close"


def test_strict_and_fast_paths_agree_exactly(store: BarStore) -> None:
    """The fast path prefetches the window; this proves it is an optimisation, not a shortcut."""
    fast = Engine(store, strict=False).run(sma(), "SYN-04", START, END, COSTS)
    strict = Engine(store, strict=True).run(sma(), "SYN-04", START, END, COSTS)
    assert fast.equity.equals(strict.equity)
    assert [t.timestamp for t in fast.trades] == [t.timestamp for t in strict.trades]
    assert [t.price for t in fast.trades] == [t.price for t in strict.trades]


def test_run_is_deterministic(engine: Engine) -> None:
    a = engine.run(sma(), "SYN-04", START, END, COSTS)
    b = engine.run(sma(), "SYN-04", START, END, COSTS)
    assert a.equity.equals(b.equity)


def test_buy_and_hold_tracks_the_underlying(engine: Engine, store: BarStore) -> None:
    result = engine.run(hold(), "SYN-01", START, END, FRICTIONLESS)
    frame = store.bars(["SYN-01"], START, END, as_of=END)
    expected = frame["close"][-1] / frame["open"][1]
    assert result.equity["equity"][-1] == pytest.approx(expected, rel=1e-9)
    assert len(result.trades) == 1


def test_costs_reduce_returns_on_a_high_turnover_strategy(engine: Engine) -> None:
    free = engine.run(sma(5, 10), "SYN-04", START, END, FRICTIONLESS)
    charged = engine.run(sma(5, 10), "SYN-04", START, END, CostModel(fee_bps=5.0, slippage_bps=0.0))
    assert len(charged.trades) > 20
    assert charged.equity["equity"][-1] < free.equity["equity"][-1]
    drag = 1 - charged.equity["equity"][-1] / free.equity["equity"][-1]
    assert drag > 0.005, f"5bp on {len(charged.trades)} trades should be material, got {drag:.4%}"


def test_warmup_is_enforced_against_available_history(engine: Engine) -> None:
    with pytest.raises(InsufficientHistory):
        engine.run(sma(20, 50), "SYN-04", dt.date(2019, 1, 1), dt.date(2019, 2, 1), COSTS)


def test_dead_symbol_cannot_be_backtested_past_its_last_bar(engine: Engine) -> None:
    with pytest.raises(InsufficientHistory):
        engine.run(sma(), "SYN-17", dt.date(2022, 1, 1), dt.date(2023, 1, 1), COSTS)


def test_weights_are_long_or_flat_only(engine: Engine) -> None:
    result = engine.run(sma(), "SYN-04", START, END, COSTS)
    weights = result.equity["weight"]
    assert weights.min() >= 0.0 and weights.max() <= 1.0


def test_fixed_fraction_sizing_caps_exposure(engine: Engine) -> None:
    spec = parse_spec(
        {"primitive": "buy_and_hold", "params": {}, "sizing": "fixed_fraction", "fraction": 0.5}
    )
    result = engine.run(spec, "SYN-01", START, END, FRICTIONLESS)
    assert result.equity["weight"].max() == pytest.approx(0.5)


def test_rebalance_cadence_reduces_trade_count(engine: Engine) -> None:
    daily = engine.run(sma(5, 10), "SYN-04", START, END, COSTS)
    monthly_spec = parse_spec(
        {"primitive": "sma_cross", "params": {"fast": 5, "slow": 10}, "rebalance": "monthly"}
    )
    monthly = engine.run(monthly_spec, "SYN-04", START, END, COSTS)
    assert len(monthly.trades) < len(daily.trades)


def test_target_position_rejects_leverage_and_shorts() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TargetPosition(1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TargetPosition(-0.2)


def test_equity_curve_is_aligned_and_finite(engine: Engine) -> None:
    result = engine.run(sma(), "SYN-04", START, END, COSTS)
    assert result.equity.height == result.equity["timestamp"].n_unique()
    assert result.equity["equity"].is_finite().all()
    assert result.equity["timestamp"].is_sorted()
    assert result.equity["equity"].gt(0).all()
    assert result.equity.select(pl.col("timestamp").max()).item() <= END
