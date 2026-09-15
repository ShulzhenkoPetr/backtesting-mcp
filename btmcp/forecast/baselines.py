from __future__ import annotations

import numpy as np

from btmcp.forecast.base import Forecaster

NORMAL_QUANTILES_EPS = 1e-9


def _z(quantile: float) -> float:
    """Inverse standard normal via a rational approximation, to avoid a scipy dependency."""
    q = min(max(quantile, NORMAL_QUANTILES_EPS), 1.0 - NORMAL_QUANTILES_EPS)
    a = [
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.3577518672690,
        -30.66479806614716,
        2.506628277459239,
    ]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = np.sqrt(-2 * np.log(q))
        return float(
            (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
        )
    if q > phigh:
        r = np.sqrt(-2 * np.log(1 - q))
        return float(
            -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
        )
    r = q - 0.5
    t = r * r
    return float(
        (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5])
        * r
        / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)
    )


class Drift:
    """Random walk with drift: extrapolates the average per-bar change of the context."""

    id = "drift"

    def forecast(self, context: np.ndarray, horizon: int, quantiles: list[float]) -> np.ndarray:
        if len(context) < 2:
            level = float(context[-1]) if len(context) else 0.0
            return np.full((len(quantiles), horizon), level)
        slope = (float(context[-1]) - float(context[0])) / (len(context) - 1)
        steps = np.arange(1, horizon + 1)
        centre = float(context[-1]) + slope * steps
        sigma = float(np.std(np.diff(context), ddof=1))
        spread = sigma * np.sqrt(steps)
        return np.stack([centre + _z(q) * spread for q in quantiles])


class SeasonalNaive:
    """Repeats the value observed one season ago, with residual-based quantile spread."""

    id = "seasonal_naive"

    def __init__(self, period: int = 21) -> None:
        self.period = period

    def forecast(self, context: np.ndarray, horizon: int, quantiles: list[float]) -> np.ndarray:
        if len(context) < self.period + 1:
            return Drift().forecast(context, horizon, quantiles)
        season = context[-self.period :]
        centre = np.array([season[i % self.period] for i in range(horizon)], dtype=float)
        residuals = context[self.period :] - context[: -self.period]
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        steps = np.arange(1, horizon + 1)
        spread = sigma * np.sqrt(steps / self.period)
        return np.stack([centre + _z(q) * spread for q in quantiles])


REGISTRY: dict[str, Forecaster] = {f.id: f for f in (Drift(), SeasonalNaive())}
