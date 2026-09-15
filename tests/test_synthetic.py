from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btmcp.data.manifest import GLOBAL_SEED, UNIVERSE, load_manifest
from btmcp.data.synthetic import generate_symbol, sha256_file

ROOT = Path("datasets/synthetic")
BY_SYMBOL = {p.symbol: p for p in UNIVERSE}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(ROOT)


def _close(symbol: str, start: dt.date | None = None, end: dt.date | None = None) -> np.ndarray:
    frame = pl.read_parquet(ROOT / f"{symbol}.parquet")
    if start is not None:
        frame = frame.filter(pl.col("timestamp") >= start)
    if end is not None:
        frame = frame.filter(pl.col("timestamp") <= end)
    return frame["close"].to_numpy()


def _sma_cross_sharpe(close: np.ndarray, fast: int = 20, slow: int = 50) -> float:
    fast_ma = np.convolve(close, np.ones(fast) / fast, mode="valid")
    slow_ma = np.convolve(close, np.ones(slow) / slow, mode="valid")
    fast_ma = fast_ma[len(fast_ma) - len(slow_ma) :]
    signal = (fast_ma > slow_ma).astype(float)
    prices = close[slow - 1 :]
    returns = np.diff(prices) / prices[:-1]
    strat = signal[:-1] * returns
    return float(strat.mean() / strat.std() * np.sqrt(252)) if strat.std() else 0.0


def _buy_hold_sharpe(close: np.ndarray) -> float:
    returns = np.diff(close) / close[:-1]
    return float(returns.mean() / returns.std() * np.sqrt(252))


def test_manifest_hashes_match_committed_files(manifest) -> None:
    assert len(manifest.symbols) == 20
    for record in manifest.symbols:
        path = ROOT / f"{record.params.symbol}.parquet"
        assert path.exists(), f"{path} missing"
        assert sha256_file(path) == record.sha256, f"{record.params.symbol} drifted from manifest"


def test_regeneration_is_byte_identical(tmp_path: Path, manifest) -> None:
    from btmcp.data.synthetic import write_symbol

    for record in manifest.symbols[:5]:
        frame = generate_symbol(record.params, manifest.global_seed)
        assert sha256_file(write_symbol(frame, tmp_path)) == record.sha256


def test_generation_is_reproducible_in_memory() -> None:
    params = BY_SYMBOL["SYN-04"]
    assert generate_symbol(params, GLOBAL_SEED).equals(generate_symbol(params, GLOBAL_SEED))


@pytest.mark.parametrize("symbol", [p.symbol for p in UNIVERSE])
def test_ohlc_invariants(symbol: str) -> None:
    frame = pl.read_parquet(ROOT / f"{symbol}.parquet")
    assert frame["high"].ge(frame["low"]).all()
    assert frame["high"].ge(frame["close"]).all() and frame["high"].ge(frame["open"]).all()
    assert frame["low"].le(frame["close"]).all() and frame["low"].le(frame["open"]).all()
    assert frame["close"].gt(0).all()
    assert frame["volume"].gt(0).all()
    assert frame["timestamp"].is_sorted() and frame["timestamp"].n_unique() == frame.height
    assert frame["timestamp"].dt.weekday().le(5).all(), "weekend bars present"


def test_syn04_signal_is_present_in_window_and_absent_after() -> None:
    """The planted ground truth the eval depends on: a real signal that genuinely decays."""
    in_window = _sma_cross_sharpe(_close("SYN-04", dt.date(2019, 1, 1), dt.date(2021, 12, 31)))
    out_window = _sma_cross_sharpe(_close("SYN-04", dt.date(2022, 1, 1), dt.date(2025, 12, 31)))
    assert in_window > 1.0, f"planted signal too weak in-window: {in_window:.2f}"
    assert out_window < 0.5, f"signal did not decay out-of-window: {out_window:.2f}"
    assert in_window - out_window > 1.0


def test_syn11_is_signal_free() -> None:
    close = _close("SYN-11")
    assert abs(_buy_hold_sharpe(close)) < 0.5
    assert abs(_sma_cross_sharpe(close)) < 0.5


def test_syn17_history_is_dead_after_mid_2021() -> None:
    frame = pl.read_parquet(ROOT / "SYN-17.parquet")
    assert frame["timestamp"][-1] == dt.date(2021, 6, 30)


def test_per_symbol_seeding_is_independent() -> None:
    """Symbol streams derive from hashed seeds, so the universe order cannot perturb them."""
    a = generate_symbol(BY_SYMBOL["SYN-01"], GLOBAL_SEED)
    b = generate_symbol(BY_SYMBOL["SYN-02"], GLOBAL_SEED)
    assert not np.allclose(a["close"].to_numpy()[:50], b["close"].to_numpy()[:50])
