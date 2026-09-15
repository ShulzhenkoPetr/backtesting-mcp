from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from btmcp.core.costs import CostModel
from btmcp.core.engine import Engine, RunResult
from btmcp.core.errors import RunNotFound
from btmcp.core.hashing import ENGINE_VERSION, config_hash
from btmcp.core.metrics import Metrics, compute
from btmcp.core.spec import StrategySpec

EQUITY_FILE = "equity.parquet"
BLOTTER_FILE = "blotter.parquet"
SUMMARY_FILE = "summary.json"


class ForecastTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    p50_ms: float = 0.0
    p99_ms: float = 0.0


class RunSummary(BaseModel):
    """Compact run description. The rows live in parquet and are fetched by link.

    This is what crosses the tool boundary, so it is sized for a context window: the
    equity curve and the blotter are addressed, never inlined, regardless of how
    short the run happens to be.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset_id: str
    engine_version: str
    symbol: str
    spec: StrategySpec
    costs: CostModel
    start: dt.date
    end: dt.date
    metrics: Metrics
    cached: bool = False
    elapsed_ms: float = Field(default=0.0, exclude=True)
    equity_uri: str
    blotter_uri: str
    forecast: ForecastTiming = ForecastTiming()


class RunStore:
    """Content-addressed run cache. Identical configs resolve to the same directory."""

    def __init__(self, root: Path | str = Path("runs")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def run_id(
        spec: StrategySpec,
        symbol: str,
        start: dt.date,
        end: dt.date,
        costs: CostModel,
        dataset_id: str,
    ) -> str:
        return config_hash(
            spec=spec,
            dataset_id=dataset_id,
            costs=costs,
            engine_version=ENGINE_VERSION,
            symbol=symbol,
            start=start,
            end=end,
        )

    def _dir(self, run_id: str) -> Path:
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self._dir(run_id) / SUMMARY_FILE).exists()

    def get(self, run_id: str) -> RunSummary:
        path = self._dir(run_id) / SUMMARY_FILE
        if not path.exists():
            raise RunNotFound(
                cause=f"no run with id {run_id}",
                suggested_action="run_backtest first, or list prior runs; run ids are config hashes",
                run_id=run_id,
            )
        return RunSummary.model_validate_json(path.read_text())

    def equity(self, run_id: str) -> pl.DataFrame:
        self.get(run_id)
        return pl.read_parquet(self._dir(run_id) / EQUITY_FILE)

    def trades(self, run_id: str) -> pl.DataFrame:
        self.get(run_id)
        return pl.read_parquet(self._dir(run_id) / BLOTTER_FILE)

    def execute(
        self,
        engine: Engine,
        spec: StrategySpec,
        symbol: str,
        start: dt.date,
        end: dt.date,
        costs: CostModel,
        dataset_id: str | None = None,
        forecaster_registry: dict[str, object] | None = None,
    ) -> RunSummary:
        dataset = dataset_id or engine.store.dataset_id
        run_id = self.run_id(spec, symbol, start, end, costs, dataset)

        started = time.perf_counter()
        if self.exists(run_id):
            summary = self.get(run_id)
            return summary.model_copy(
                update={"cached": True, "elapsed_ms": (time.perf_counter() - started) * 1000.0}
            )

        result = engine.run(spec, symbol, start, end, costs, forecaster_registry=forecaster_registry)
        summary = self._persist(run_id, dataset, result, start, end)
        return summary.model_copy(update={"elapsed_ms": (time.perf_counter() - started) * 1000.0})

    def _persist(
        self, run_id: str, dataset_id: str, result: RunResult, start: dt.date, end: dt.date
    ) -> RunSummary:
        target = self._dir(run_id)
        target.mkdir(parents=True, exist_ok=True)
        result.equity.write_parquet(target / EQUITY_FILE, compression="zstd")
        _blotter(result).write_parquet(target / BLOTTER_FILE, compression="zstd")

        latencies = np.asarray(result.forecast_latency_ms)
        summary = RunSummary(
            run_id=run_id,
            dataset_id=dataset_id,
            engine_version=ENGINE_VERSION,
            symbol=result.symbol,
            spec=result.spec,
            costs=result.costs,
            start=start,
            end=end,
            metrics=compute(result),
            cached=False,
            equity_uri=f"btmcp://runs/{run_id}/equity",
            blotter_uri=f"btmcp://runs/{run_id}/blotter",
            forecast=ForecastTiming(
                calls=result.forecast_calls,
                p50_ms=float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
                p99_ms=float(np.percentile(latencies, 99)) if len(latencies) else 0.0,
            ),
        )
        (target / SUMMARY_FILE).write_text(summary.model_dump_json(indent=2))
        return summary


def _blotter(result: RunResult) -> pl.DataFrame:
    if not result.trades:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Date(),
                "side": pl.String(),
                "price": pl.Float64(),
                "notional": pl.Float64(),
                "cost": pl.Float64(),
                "weight_before": pl.Float64(),
                "weight_after": pl.Float64(),
            }
        )
    return pl.DataFrame([vars(t) for t in result.trades])
