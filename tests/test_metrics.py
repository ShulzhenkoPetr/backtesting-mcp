from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from btmcp.core.costs import FRICTIONLESS, CostModel
from btmcp.core.engine import Engine
from btmcp.core.metrics import TRADING_DAYS, compute
from btmcp.core.spec import parse_spec
from btmcp.data.barstore import BarStore

START, END = dt.date(2019, 1, 1), dt.date(2021, 12, 31)
FULL = (dt.date(2018, 1, 1), dt.date(2025, 12, 31))


@pytest.fixture(scope="module")
def store() -> BarStore:
    return BarStore()


@pytest.fixture(scope="module")
def engine(store: BarStore) -> Engine:
    return Engine(store)


def test_buy_and_hold_matches_a_hand_computed_reference(engine: Engine, store: BarStore) -> None:
    """E2-T4 acceptance: metrics reproduce an independent numpy computation to 1e-9."""
    spec = parse_spec({"primitive": "buy_and_hold", "params": {}})
    result = engine.run(spec, "SYN-01", START, END, FRICTIONLESS)
    metrics = compute(result)

    frame = store.bars(["SYN-01"], START, END, as_of=END)
    opens, closes = frame["open"].to_numpy(), frame["close"].to_numpy()
    # Frictionless buy-and-hold: all capital enters at the second bar's open and never leaves.
    equity = np.concatenate([[1.0], closes[1:] / opens[1]])
    returns = np.diff(equity) / equity[:-1]

    assert metrics.total_return == pytest.approx(equity[-1] - 1.0, abs=1e-9)
    assert metrics.annualised_vol == pytest.approx(returns.std(ddof=1) * np.sqrt(TRADING_DAYS), abs=1e-9)
    assert metrics.sharpe == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS), abs=1e-9
    )
    assert metrics.hit_rate == pytest.approx(float((returns > 0).mean()), abs=1e-9)
    assert metrics.max_drawdown == pytest.approx(
        float((equity / np.maximum.accumulate(equity) - 1.0).min()), abs=1e-9
    )
    years = len(returns) / TRADING_DAYS
    assert metrics.annualised_return == pytest.approx(equity[-1] ** (1 / years) - 1.0, abs=1e-9)
    assert metrics.risk_free_rate == 0.0


@pytest.mark.parametrize(
    "spec_payload",
    [
        {"primitive": "buy_and_hold", "params": {}},
        {"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}},
        {"primitive": "sma_cross", "params": {"fast": 5, "slow": 10}},
        {"primitive": "threshold", "params": {"lookback": 20, "entry_pct": 2.0, "exit_pct": -2.0}},
    ],
)
def test_random_walk_symbol_yields_no_edge_under_any_primitive(engine: Engine, spec_payload: dict) -> None:
    """SYN-11 has no signal by construction. A Sharpe above 0.5 here means a bug in the engine."""
    result = engine.run(parse_spec(spec_payload), "SYN-11", *FULL, FRICTIONLESS)
    metrics = compute(result)
    assert abs(metrics.sharpe) < 0.5, (
        f"{spec_payload['primitive']} found edge in a random walk: {metrics.sharpe:.2f}"
    )


def test_planted_signal_shows_decay_across_years(engine: Engine) -> None:
    spec = parse_spec({"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}})
    in_window = compute(engine.run(spec, "SYN-04", START, END, FRICTIONLESS))
    out_window = compute(engine.run(spec, "SYN-04", dt.date(2022, 1, 1), dt.date(2025, 12, 31), FRICTIONLESS))
    assert in_window.sharpe > 1.0
    assert out_window.sharpe < 0.5


def test_drawdown_dates_are_ordered_and_real(engine: Engine) -> None:
    metrics = compute(
        engine.run(parse_spec({"primitive": "buy_and_hold", "params": {}}), "SYN-03", *FULL, FRICTIONLESS)
    )
    dd = metrics.drawdown
    assert dd.max_drawdown < 0
    assert dd.peak_date is not None and dd.trough_date is not None
    assert dd.peak_date < dd.trough_date
    if dd.recovery_date is not None:
        assert dd.recovery_date > dd.trough_date


def test_breakdowns_partition_the_sample(engine: Engine) -> None:
    metrics = compute(
        engine.run(
            parse_spec({"primitive": "buy_and_hold", "params": {}}), "SYN-01", START, END, FRICTIONLESS
        )
    )
    assert sum(b.n_bars for b in metrics.by_year) == metrics.n_bars
    assert sum(b.n_bars for b in metrics.by_volatility_regime) == metrics.n_bars
    assert [b.label for b in metrics.by_volatility_regime] == ["calm", "stressed"]
    assert {b.label for b in metrics.by_year} == {"2019", "2020", "2021"}


def test_stressed_regime_has_higher_volatility(engine: Engine) -> None:
    metrics = compute(
        engine.run(parse_spec({"primitive": "buy_and_hold", "params": {}}), "SYN-13", *FULL, FRICTIONLESS)
    )
    calm, stressed = metrics.by_volatility_regime
    assert stressed.annualised_vol > calm.annualised_vol


def test_exposure_and_holding_period_are_consistent(engine: Engine) -> None:
    held = compute(
        engine.run(
            parse_spec({"primitive": "buy_and_hold", "params": {}}), "SYN-01", START, END, FRICTIONLESS
        )
    )
    assert held.exposure_fraction == pytest.approx(1.0)
    timed = compute(
        engine.run(
            parse_spec({"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}}),
            "SYN-04",
            START,
            END,
            FRICTIONLESS,
        )
    )
    assert 0.0 < timed.exposure_fraction < 1.0
    assert timed.avg_holding_period_bars > 1.0


def test_costs_are_accounted_and_reported(engine: Engine) -> None:
    spec = parse_spec({"primitive": "sma_cross", "params": {"fast": 5, "slow": 10}})
    charged = compute(engine.run(spec, "SYN-04", START, END, CostModel(fee_bps=10.0, slippage_bps=5.0)))
    free = compute(engine.run(spec, "SYN-04", START, END, FRICTIONLESS))
    assert charged.total_costs > 0 and free.total_costs == 0
    assert charged.total_return < free.total_return
    assert charged.turnover == pytest.approx(free.turnover, rel=0.15)
