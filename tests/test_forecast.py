from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from btmcp.core.costs import FRICTIONLESS
from btmcp.core.engine import Engine
from btmcp.core.errors import InvalidSpec
from btmcp.core.metrics import compute
from btmcp.core.spec import parse_spec
from btmcp.core.store import RunStore
from btmcp.data.barstore import BarStore
from btmcp.forecast.base import Forecaster
from btmcp.forecast.baselines import REGISTRY, Drift, SeasonalNaive, _z

START, END = dt.date(2019, 1, 1), dt.date(2021, 12, 31)


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(BarStore())


def spec(forecaster_id: str = "seasonal_naive", **kw):
    params = {"forecaster_id": forecaster_id, "horizon": 5, "quantile": 0.5, "threshold": 0.0, **kw}
    return parse_spec({"primitive": "forecast_signal", "params": params})


def test_baselines_satisfy_the_protocol() -> None:
    assert isinstance(Drift(), Forecaster) and isinstance(SeasonalNaive(), Forecaster)
    assert set(REGISTRY) == {"drift", "seasonal_naive"}


def test_inverse_normal_approximation_is_accurate() -> None:
    assert _z(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _z(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert _z(0.025) == pytest.approx(-1.959964, abs=1e-4)
    assert _z(0.01) == pytest.approx(-2.326348, abs=1e-4)


@pytest.mark.parametrize("forecaster", [Drift(), SeasonalNaive()])
def test_forecast_shape_and_quantile_ordering(forecaster) -> None:
    context = np.cumsum(np.random.default_rng(0).standard_normal(200)) + 100
    out = forecaster.forecast(context, horizon=5, quantiles=[0.1, 0.5, 0.9])
    assert out.shape == (3, 5)
    assert np.all(out[0] <= out[1]) and np.all(out[1] <= out[2]), "quantiles must be ordered"


def test_forecaster_cannot_see_beyond_its_context() -> None:
    """The signature carries no dates, so a forecaster has nothing to look ahead with."""
    import inspect

    params = inspect.signature(Drift.forecast).parameters
    assert set(params) == {"self", "context", "horizon", "quantiles"}


def test_forecast_driven_backtest_differs_from_buy_and_hold(engine: Engine) -> None:
    """E7-T1 acceptance."""
    forecast = compute(engine.run(spec(), "SYN-04", START, END, FRICTIONLESS))
    hold = compute(
        engine.run(
            parse_spec({"primitive": "buy_and_hold", "params": {}}), "SYN-04", START, END, FRICTIONLESS
        )
    )
    assert forecast.total_return != pytest.approx(hold.total_return)
    assert forecast.exposure_fraction < 1.0


def test_unknown_forecaster_id_is_a_typed_error(engine: Engine) -> None:
    with pytest.raises(InvalidSpec) as exc:
        engine.run(spec("timesfm_xl"), "SYN-04", START, END, FRICTIONLESS)
    assert exc.value.context["offending_key"] == "params.forecaster_id"
    assert "seasonal_naive" in exc.value.suggested_action


def test_forecast_latency_is_reported_separately(engine: Engine, tmp_path) -> None:
    """E7-T2 acceptance: forecast cost is visible apart from engine time."""
    store = RunStore(tmp_path / "runs")
    summary = store.execute(engine, spec(), "SYN-04", START, END, FRICTIONLESS)
    assert summary.forecast.calls > 100
    assert summary.forecast.p50_ms > 0.0
    assert summary.forecast.p99_ms >= summary.forecast.p50_ms

    plain = store.execute(
        engine, parse_spec({"primitive": "buy_and_hold", "params": {}}), "SYN-04", START, END, FRICTIONLESS
    )
    assert plain.forecast.calls == 0


def test_optional_extra_is_not_required_by_the_core() -> None:
    """E7-T2 acceptance: the core suite passes without the [forecast] extra installed."""
    import importlib.util

    assert importlib.util.find_spec("timesfm") is None
    assert set(REGISTRY) == {"drift", "seasonal_naive"}
