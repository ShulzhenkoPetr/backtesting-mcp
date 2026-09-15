from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from btmcp.core.costs import CostModel
from btmcp.core.errors import InsufficientHistory, LookaheadError
from btmcp.core.spec import ForecastParams, SmaCrossParams, StrategySpec, ThresholdParams
from btmcp.data.barstore import BarStore

EPSILON = 1e-12


@dataclass(frozen=True)
class TargetPosition:
    """Long/flat only, so weight is bounded to [0, 1]. Shorting is out of scope."""

    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")


class BarView:
    """The only thing a strategy ever sees. Cannot be made to yield a bar after `t`.

    The frame handed in is already truncated. `as_of` is exposed so a strategy can
    reason about its own horizon, and every accessor that takes an `as_of` clamps
    rather than trusts: asking for a later one raises instead of silently returning
    what the engine happens to be holding.
    """

    __slots__ = ("_frame", "_t")

    def __init__(self, frame: pl.DataFrame, t: dt.date) -> None:
        self._frame = frame
        self._t = t

    @property
    def as_of(self) -> dt.date:
        return self._t

    def __len__(self) -> int:
        return self._frame.height

    def _check(self, as_of: dt.date | None) -> None:
        if as_of is not None and as_of > self._t:
            raise LookaheadError(
                cause=f"strategy requested data as of {as_of} while deciding at {self._t}",
                suggested_action="a strategy may only read bars at or before its decision timestamp",
                requested=as_of.isoformat(),
                decision_date=self._t.isoformat(),
            )

    def bars(self, lookback: int | None = None, as_of: dt.date | None = None) -> pl.DataFrame:
        self._check(as_of)
        frame = self._frame
        if as_of is not None:
            frame = frame.filter(pl.col("timestamp") <= as_of)
        return frame.tail(lookback) if lookback else frame

    def closes(self, lookback: int | None = None, as_of: dt.date | None = None) -> np.ndarray:
        return self.bars(lookback=lookback, as_of=as_of)["close"].to_numpy()

    def last_close(self) -> float:
        return float(self._frame["close"][-1])


class Strategy(Protocol):
    def decide(self, t: dt.date, view: BarView) -> TargetPosition: ...


class BuyAndHold:
    def decide(self, t: dt.date, view: BarView) -> TargetPosition:
        return TargetPosition(1.0)


@dataclass
class SmaCross:
    params: SmaCrossParams

    def decide(self, t: dt.date, view: BarView) -> TargetPosition:
        closes = view.closes(lookback=self.params.slow)
        if len(closes) < self.params.slow:
            return TargetPosition(0.0)
        fast = float(closes[-self.params.fast :].mean())
        slow = float(closes.mean())
        return TargetPosition(1.0 if fast > slow else 0.0)


@dataclass
class Threshold:
    params: ThresholdParams
    _held: bool = False

    def decide(self, t: dt.date, view: BarView) -> TargetPosition:
        closes = view.closes(lookback=self.params.lookback + 1)
        if len(closes) < self.params.lookback + 1:
            return TargetPosition(0.0)
        trailing = (closes[-1] / closes[0] - 1.0) * 100.0
        if trailing >= self.params.entry_pct:
            self._held = True
        elif trailing < self.params.exit_pct:
            self._held = False
        return TargetPosition(1.0 if self._held else 0.0)


@dataclass
class Trade:
    timestamp: dt.date
    side: str
    price: float
    notional: float
    cost: float
    weight_before: float
    weight_after: float


@dataclass
class RunResult:
    equity: pl.DataFrame
    trades: list[Trade]
    spec: StrategySpec
    symbol: str
    costs: CostModel
    forecast_latency_ms: list[float] = field(default_factory=list)
    forecast_calls: int = 0


def build_strategy(spec: StrategySpec, forecaster_registry: dict[str, object] | None = None) -> Strategy:
    if isinstance(spec.params, SmaCrossParams):
        return SmaCross(spec.params)
    if isinstance(spec.params, ThresholdParams):
        return Threshold(spec.params)
    if isinstance(spec.params, ForecastParams):
        from btmcp.forecast.strategy import ForecastSignal

        strategy: Strategy = ForecastSignal(spec.params, forecaster_registry or {})
        return strategy
    return BuyAndHold()


def _is_rebalance_day(rebalance: str, day: dt.date, previous: dt.date | None) -> bool:
    if previous is None or rebalance == "daily":
        return True
    if rebalance == "weekly":
        return day.isocalendar()[:2] != previous.isocalendar()[:2]
    return (day.year, day.month) != (previous.year, previous.month)


class Engine:
    """Iterates decision timestamps, handing each one a `BarView` bounded by that date.

    `strict` re-queries the store at every decision timestamp, which is the literal
    reading of the point-in-time rule but costs one SQL round trip per bar. The
    default path fetches the run's window once and slices it per timestamp. The two
    are asserted equivalent in the test suite, so the fast path is an optimisation of
    the strict one rather than a weaker guarantee.
    """

    def __init__(self, store: BarStore, strict: bool = False) -> None:
        self.store = store
        self.strict = strict

    def run(
        self,
        spec: StrategySpec,
        symbol: str,
        start: dt.date,
        end: dt.date,
        costs: CostModel,
        forecaster_registry: dict[str, object] | None = None,
    ) -> RunResult:
        required = max(spec.min_bars, spec.warmup_bars)
        window = self.store.bars([symbol], start=start, end=end, as_of=end, min_rows=required + 1)

        n = window.height
        if n < required + 2:
            raise InsufficientHistory(
                cause=f"{symbol} has {n} bars in {start}..{end} but the spec needs {required + 2}",
                suggested_action="widen the window, lower warmup_bars, or choose a shorter primitive",
                required=required + 2,
                available=n,
            )

        strategy = build_strategy(spec, forecaster_registry)
        stamps: list[dt.date] = window["timestamp"].to_list()
        opens = window["open"].to_numpy()
        closes = window["close"].to_numpy()

        first_decision = max(required - 1, 0)
        weights = np.zeros(n)
        last_rebalance: dt.date | None = None
        current = 0.0

        for i in range(first_decision, n - 1):
            t = stamps[i]
            if _is_rebalance_day(spec.rebalance, t, last_rebalance):
                view = self._view(symbol, window, start, i, t)
                target = strategy.decide(t, view)
                current = target.weight * (
                    spec.fraction if spec.sizing == "fixed_fraction" and spec.fraction else 1.0
                )
                last_rebalance = t
            weights[i] = current

        equity, trades = self._simulate(stamps, opens, closes, weights, first_decision, costs)
        latencies: list[float] = list(getattr(strategy, "latencies_ms", []))
        # The curve carries its own base point so a reader never has to assume the start value.
        curve = pl.DataFrame(
            {
                "timestamp": stamps[first_decision:],
                "equity": np.concatenate([[1.0], equity]),
                "weight": np.concatenate([[0.0], weights[first_decision:-1]]),
            }
        )
        return RunResult(
            equity=curve,
            trades=trades,
            spec=spec,
            symbol=symbol,
            costs=costs,
            forecast_latency_ms=latencies,
            forecast_calls=len(latencies),
        )

    def _view(self, symbol: str, window: pl.DataFrame, start: dt.date, i: int, t: dt.date) -> BarView:
        if self.strict:
            return BarView(self.store.bars([symbol], start=start, end=t, as_of=t), t)
        return BarView(window.head(i + 1), t)

    @staticmethod
    def _simulate(
        stamps: list[dt.date],
        opens: np.ndarray,
        closes: np.ndarray,
        weights: np.ndarray,
        first_decision: int,
        costs: CostModel,
    ) -> tuple[np.ndarray, list[Trade]]:
        """Executes at the next bar's open, never the deciding bar's close."""
        cash, shares = 1.0, 0.0
        equity: list[float] = []
        trades: list[Trade] = []
        previous_weight = 0.0

        for i in range(first_decision + 1, len(stamps)):
            open_price = float(opens[i])
            close_price = float(closes[i])
            target_weight = float(weights[i - 1])

            equity_at_open = cash + shares * open_price
            target_value = target_weight * equity_at_open
            current_value = shares * open_price
            delta = target_value - current_value

            if abs(delta) > EPSILON * max(1.0, equity_at_open):
                cost = costs.trade_cost(delta)
                shares += delta / open_price
                cash -= delta + cost
                trades.append(
                    Trade(
                        timestamp=stamps[i],
                        side="buy" if delta > 0 else "sell",
                        price=open_price,
                        notional=abs(delta),
                        cost=cost,
                        weight_before=previous_weight,
                        weight_after=target_weight,
                    )
                )
                previous_weight = target_weight

            cash -= costs.carry_cost(shares * close_price, bars=1)
            equity.append(cash + shares * close_price)

        return np.asarray(equity), trades
