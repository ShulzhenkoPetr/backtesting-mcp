from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from btmcp.core.engine import RunResult

TRADING_DAYS = 252
REGIME_WINDOW = 21


class DrawdownInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_drawdown: float
    peak_date: dt.date | None
    trough_date: dt.date | None
    recovery_date: dt.date | None


class PeriodBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    n_bars: int
    total_return: float
    annualised_vol: float
    sharpe: float


class Metrics(BaseModel):
    """Risk-free rate is zero throughout, and stated rather than assumed.

    Sharpe and Sortino are computed on daily arithmetic returns and scaled by
    sqrt(252); no rate is subtracted. Quoting a Sharpe without naming the rate is a
    common way to make a number look better than it is.
    """

    model_config = ConfigDict(extra="forbid")

    risk_free_rate: float = 0.0
    n_bars: int
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    drawdown: DrawdownInfo
    calmar: float
    hit_rate: float
    turnover: float
    avg_holding_period_bars: float
    exposure_fraction: float
    trade_count: int
    total_costs: float
    by_year: list[PeriodBreakdown]
    by_volatility_regime: list[PeriodBreakdown]


def _annualised_vol(returns: np.ndarray) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def _sharpe(returns: np.ndarray) -> float:
    vol = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else 0.0


def _sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dd = downside.std(ddof=1)
    return float(returns.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def _drawdown(equity: np.ndarray, stamps: list[dt.date]) -> DrawdownInfo:
    peaks = np.maximum.accumulate(equity)
    series = equity / peaks - 1.0
    trough = int(series.argmin())
    worst = float(series[trough])
    if worst == 0.0:
        return DrawdownInfo(max_drawdown=0.0, peak_date=None, trough_date=None, recovery_date=None)
    peak = int(equity[: trough + 1].argmax())
    after = np.nonzero(equity[trough:] >= equity[peak])[0]
    recovery = stamps[trough + int(after[0])] if len(after) else None
    return DrawdownInfo(
        max_drawdown=worst, peak_date=stamps[peak], trough_date=stamps[trough], recovery_date=recovery
    )


def _breakdown(label: str, returns: np.ndarray) -> PeriodBreakdown:
    total = float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0
    return PeriodBreakdown(
        label=label,
        n_bars=len(returns),
        total_return=total,
        annualised_vol=_annualised_vol(returns),
        sharpe=_sharpe(returns),
    )


def _volatility_regimes(returns: np.ndarray) -> np.ndarray:
    """Labels bars calm/stressed by trailing realised vol against the sample median.

    Derived from prices rather than read from the generator's hidden state, so the
    same breakdown is available on real data and is reproducible by a reader who
    only has the bars.
    """
    if len(returns) < REGIME_WINDOW:
        return np.zeros(len(returns), dtype=bool)
    frame = pl.DataFrame({"r": returns})
    trailing = frame.select(pl.col("r").rolling_std(REGIME_WINDOW, min_samples=2))["r"].to_numpy()
    trailing = np.nan_to_num(trailing, nan=float(np.nanmedian(trailing)))
    mask: np.ndarray = trailing > float(np.median(trailing))
    return mask


def compute(result: RunResult) -> Metrics:
    curve = result.equity
    equity = curve["equity"].to_numpy()
    stamps: list[dt.date] = curve["timestamp"].to_list()
    weights = curve["weight"].to_numpy()
    returns = np.diff(equity) / equity[:-1]
    return_stamps = stamps[1:]

    n_bars = len(returns)
    years = n_bars / TRADING_DAYS if n_bars else 0.0
    total_return = float(equity[-1] / equity[0] - 1.0)
    annualised = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 and total_return > -1 else 0.0
    )
    drawdown = _drawdown(equity, stamps)

    traded = sum(t.notional for t in result.trades)
    costs_paid = sum(t.cost for t in result.trades)
    held = weights[1:] > 0
    holding_runs = _run_lengths(held)

    by_year = [
        _breakdown(str(year), returns[[i for i, s in enumerate(return_stamps) if s.year == year]])
        for year in sorted({s.year for s in return_stamps})
    ]
    stressed = _volatility_regimes(returns)
    by_regime = [
        _breakdown("calm", returns[~stressed]),
        _breakdown("stressed", returns[stressed]),
    ]

    return Metrics(
        n_bars=n_bars,
        total_return=total_return,
        annualised_return=annualised,
        annualised_vol=_annualised_vol(returns),
        sharpe=_sharpe(returns),
        sortino=_sortino(returns),
        max_drawdown=drawdown.max_drawdown,
        drawdown=drawdown,
        calmar=float(annualised / abs(drawdown.max_drawdown)) if drawdown.max_drawdown < 0 else 0.0,
        hit_rate=float((returns > 0).mean()) if n_bars else 0.0,
        turnover=float(traded / equity[0]),
        avg_holding_period_bars=float(np.mean(holding_runs)) if holding_runs else 0.0,
        exposure_fraction=float(held.mean()) if len(held) else 0.0,
        trade_count=len(result.trades),
        total_costs=float(costs_paid),
        by_year=by_year,
        by_volatility_regime=by_regime,
    )


def _run_lengths(held: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in held:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs
