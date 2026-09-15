from __future__ import annotations

import datetime as dt
import time
from typing import TYPE_CHECKING

from btmcp.core.errors import InvalidSpec
from btmcp.core.spec import ForecastParams
from btmcp.forecast.base import Forecaster
from btmcp.forecast.baselines import REGISTRY

if TYPE_CHECKING:
    from btmcp.core.engine import BarView, TargetPosition


class ForecastSignal:
    """Goes long when the forecast return at the chosen quantile clears the threshold.

    Latency is recorded per call so the run summary can report forecast cost
    separately from engine time (E7-T2): the bridge to the linearization work needs
    quality and latency deltas on the same axis.
    """

    def __init__(self, params: ForecastParams, registry: dict[str, object] | None = None) -> None:
        available: dict[str, Forecaster] = {**REGISTRY, **(registry or {})}  # type: ignore[dict-item]
        forecaster = available.get(params.forecaster_id)
        if forecaster is None:
            raise InvalidSpec(
                cause=f"unknown forecaster_id '{params.forecaster_id}'",
                suggested_action=f"use one of: {', '.join(sorted(available))}",
                offending_key="params.forecaster_id",
                valid_options=sorted(available),
            )
        self.params = params
        self.forecaster = forecaster
        self.latencies_ms: list[float] = []

    def decide(self, t: dt.date, view: BarView) -> TargetPosition:
        from btmcp.core.engine import TargetPosition

        context = view.closes()
        needed = 2 * self.params.horizon + 2
        if len(context) < needed:
            return TargetPosition(0.0)

        started = time.perf_counter()
        paths = self.forecaster.forecast(context, self.params.horizon, [self.params.quantile])
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)

        predicted = float(paths[0, -1])
        expected_return = (predicted / float(context[-1]) - 1.0) * 100.0
        return TargetPosition(1.0 if expected_return > self.params.threshold else 0.0)
