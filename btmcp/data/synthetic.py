from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

GENERATOR_VERSION = "1.0.0"

TRADING_DAYS = 252
SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "timestamp": pl.Date(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
}


class SymbolParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    start: dt.date
    end: dt.date
    initial_price: float = Field(gt=0)
    annual_drift: float
    annual_vol_calm: float = Field(gt=0)
    annual_vol_stressed: float = Field(gt=0)
    p_stay_calm: float = Field(gt=0, lt=1)
    p_stay_stressed: float = Field(gt=0, lt=1)
    seasonal_period: Literal[0, 5, 21, 63, 252] = 0
    seasonal_amplitude: float = Field(default=0.0, ge=0)
    trend_strength: float = Field(default=0.0, ge=0)
    trend_dwell_days: int = Field(default=90, gt=0)
    trend_until: dt.date | None = None
    gap_prob: float = Field(default=0.0, ge=0, lt=0.1)
    note: str = ""


class SymbolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: SymbolParams
    n_rows: int
    first: dt.date
    last: dt.date
    sha256: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_version: str
    global_seed: int
    dataset_id: str
    symbols: list[SymbolRecord]


def _symbol_seed(global_seed: int, symbol: str) -> int:
    """Derives a per-symbol seed by hashing, so adding a symbol cannot shift another's stream."""
    digest = hashlib.sha256(f"{global_seed}:{symbol}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    days = pl.date_range(start, end, interval="1d", eager=True)
    return list(days.filter(days.dt.weekday() <= 5))


def generate_symbol(params: SymbolParams, global_seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(_symbol_seed(global_seed, params.symbol))
    dates = _trading_days(params.start, params.end)
    n = len(dates)

    vol_calm = params.annual_vol_calm / np.sqrt(TRADING_DAYS)
    vol_stressed = params.annual_vol_stressed / np.sqrt(TRADING_DAYS)
    base_drift = params.annual_drift / TRADING_DAYS

    regime = np.empty(n, dtype=np.int8)
    state = 0
    for i in range(n):
        stay = params.p_stay_calm if state == 0 else params.p_stay_stressed
        if rng.random() > stay:
            state = 1 - state
        regime[i] = state
    sigma = np.where(regime == 0, vol_calm, vol_stressed)

    trend = np.zeros(n)
    if params.trend_strength > 0:
        switch_p = 1.0 / params.trend_dwell_days
        sign = 1 if rng.random() < 0.5 else -1
        cutoff = params.trend_until or params.end
        for i, day in enumerate(dates):
            if rng.random() < switch_p:
                sign = -sign
            trend[i] = params.trend_strength * sign if day <= cutoff else 0.0

    seasonal = np.zeros(n)
    if params.seasonal_period and params.seasonal_amplitude:
        phase = rng.uniform(0, 2 * np.pi)
        idx = np.arange(n)
        seasonal = params.seasonal_amplitude * np.sin(2 * np.pi * idx / params.seasonal_period + phase)

    shocks = rng.standard_normal(n) * sigma
    log_returns = base_drift + trend + seasonal - 0.5 * sigma**2 + shocks
    close = params.initial_price * np.exp(np.cumsum(log_returns))

    prev_close = np.concatenate([[params.initial_price], close[:-1]])
    overnight = rng.standard_normal(n) * sigma * 0.35
    open_ = prev_close * np.exp(overnight)

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.standard_normal(n)) * sigma * 0.5)
    low = body_lo * np.exp(-np.abs(rng.standard_normal(n)) * sigma * 0.5)

    log_volume = 13.0 + 0.4 * regime + rng.standard_normal(n) * 0.35
    volume = np.exp(log_volume).astype(np.int64)

    frame = pl.DataFrame(
        {
            "symbol": [params.symbol] * n,
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        schema=SCHEMA,
    )

    if params.gap_prob > 0:
        keep = rng.random(n) >= params.gap_prob
        keep[0] = True
        keep[-1] = True
        frame = frame.filter(pl.Series(keep))

    return frame.with_columns(pl.col(["open", "high", "low", "close"]).round(6))


def write_symbol(frame: pl.DataFrame, out_dir: Path) -> Path:
    """Writes uncompressed parquet so regeneration is byte-identical across zstd versions."""
    path = out_dir / f"{frame['symbol'][0]}.parquet"
    frame.write_parquet(path, compression="uncompressed", statistics=False)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
