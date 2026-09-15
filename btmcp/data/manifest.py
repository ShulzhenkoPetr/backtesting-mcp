from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from btmcp.data.synthetic import GENERATOR_VERSION, Manifest, SymbolParams

DATASET_ID = "syn-v1"
GLOBAL_SEED = 20260906
START = dt.date(2018, 1, 1)
END = dt.date(2025, 12, 31)

_SIGNAL_END = dt.date(2021, 12, 31)


def _sym(symbol: str, **kw: object) -> SymbolParams:
    base: dict[str, object] = {
        "symbol": symbol,
        "start": START,
        "end": END,
        "initial_price": 100.0,
        "annual_drift": 0.06,
        "annual_vol_calm": 0.18,
        "annual_vol_stressed": 0.36,
        "p_stay_calm": 0.99,
        "p_stay_stressed": 0.95,
    }
    base.update(kw)
    return SymbolParams(**base)


UNIVERSE: tuple[SymbolParams, ...] = (
    _sym("SYN-01", annual_drift=0.09, initial_price=42.0),
    _sym("SYN-02", annual_drift=0.03, annual_vol_calm=0.11, annual_vol_stressed=0.22, initial_price=88.0),
    _sym("SYN-03", annual_drift=0.12, annual_vol_calm=0.29, annual_vol_stressed=0.55, initial_price=17.5),
    _sym(
        "SYN-04",
        annual_drift=0.0,
        trend_strength=0.0018,
        trend_dwell_days=90,
        trend_until=_SIGNAL_END,
        initial_price=100.0,
        note="Planted 20/50 SMA cross signal through 2021; decays to noise afterwards.",
    ),
    _sym("SYN-05", annual_drift=0.05, seasonal_period=21, seasonal_amplitude=0.0009, initial_price=61.0),
    _sym("SYN-06", annual_drift=-0.04, initial_price=130.0),
    _sym("SYN-07", annual_drift=0.07, seasonal_period=252, seasonal_amplitude=0.0006, initial_price=75.0),
    _sym("SYN-08", annual_drift=0.02, gap_prob=0.012, initial_price=54.0, note="Sparse history with gaps."),
    _sym("SYN-09", annual_drift=0.15, annual_vol_calm=0.33, annual_vol_stressed=0.62, initial_price=9.4),
    _sym("SYN-10", annual_drift=0.04, seasonal_period=63, seasonal_amplitude=0.0007, initial_price=210.0),
    _sym(
        "SYN-11",
        annual_drift=0.0,
        annual_vol_calm=0.20,
        annual_vol_stressed=0.20,
        initial_price=100.0,
        note="Pure driftless random walk. Any strategy showing Sharpe > 1 here indicates a bug.",
    ),
    _sym("SYN-12", annual_drift=0.08, annual_vol_calm=0.14, annual_vol_stressed=0.26, initial_price=33.0),
    _sym("SYN-13", annual_drift=0.01, p_stay_calm=0.97, p_stay_stressed=0.90, initial_price=120.0),
    _sym("SYN-14", annual_drift=0.10, seasonal_period=5, seasonal_amplitude=0.0011, initial_price=27.0),
    _sym("SYN-15", annual_drift=0.06, gap_prob=0.005, initial_price=95.0),
    _sym("SYN-16", annual_drift=-0.02, annual_vol_calm=0.24, annual_vol_stressed=0.44, initial_price=68.0),
    _sym(
        "SYN-17",
        end=dt.date(2021, 6, 30),
        annual_drift=0.05,
        initial_price=140.0,
        note="History ends 2021-06-30. Exercises coverage and InsufficientHistory errors.",
    ),
    _sym("SYN-18", annual_drift=0.11, trend_strength=0.0009, trend_dwell_days=120, initial_price=48.0),
    _sym("SYN-19", annual_drift=0.03, annual_vol_calm=0.16, annual_vol_stressed=0.31, initial_price=155.0),
    _sym(
        "SYN-20",
        annual_drift=0.07,
        seasonal_period=21,
        seasonal_amplitude=0.0005,
        gap_prob=0.008,
        initial_price=82.0,
    ),
)


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def load_manifest(root: Path) -> Manifest:
    return Manifest.model_validate_json(manifest_path(root).read_text())


def save_manifest(manifest: Manifest, root: Path) -> None:
    payload = manifest.model_dump(mode="json")
    manifest_path(root).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def empty_manifest(symbols: list[object]) -> Manifest:
    return Manifest(
        generator_version=GENERATOR_VERSION,
        global_seed=GLOBAL_SEED,
        dataset_id=DATASET_ID,
        symbols=symbols,
    )
