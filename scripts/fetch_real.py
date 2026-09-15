from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import urllib.request
from pathlib import Path

import polars as pl

from btmcp.data.synthetic import SCHEMA, sha256_file

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
USER_AGENT = "btmcp-research/0.1 (personal research; not redistributed)"


def fetch_stooq(symbol: str, timeout: float = 30.0) -> pl.DataFrame:
    ticker = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    request = urllib.request.Request(STOOQ_URL.format(symbol=ticker), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return parse_stooq(symbol, raw)


def parse_stooq(symbol: str, raw: str) -> pl.DataFrame:
    """Normalises a stooq CSV onto the synthetic schema so both stores are interchangeable."""
    if not raw.strip() or raw.strip().lower().startswith("no data"):
        raise ValueError(f"stooq returned no data for {symbol}")
    frame = pl.read_csv(io.StringIO(raw), try_parse_dates=True)
    missing = {"Date", "Open", "High", "Low", "Close", "Volume"} - set(frame.columns)
    if missing:
        raise ValueError(f"unexpected stooq columns for {symbol}: missing {sorted(missing)}")
    return (
        frame.select(
            pl.lit(symbol).alias("symbol"),
            pl.col("Date").cast(pl.Date()).alias("timestamp"),
            pl.col("Open").cast(pl.Float64()).alias("open"),
            pl.col("High").cast(pl.Float64()).alias("high"),
            pl.col("Low").cast(pl.Float64()).alias("low"),
            pl.col("Close").cast(pl.Float64()).alias("close"),
            pl.col("Volume").cast(pl.Int64()).alias("volume"),
        )
        .drop_nulls()
        .sort("timestamp")
        .cast(SCHEMA)  # type: ignore[arg-type]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["stooq"], default="stooq")
    parser.add_argument("--symbols", required=True, help="comma-separated, e.g. AAPL,MSFT")
    parser.add_argument("--out", type=Path, default=Path("datasets/real"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    records = []
    for symbol in symbols:
        frame = fetch_stooq(symbol)
        path = args.out / f"{symbol}.parquet"
        frame.write_parquet(path, compression="uncompressed", statistics=False)
        records.append(
            {
                "symbol": symbol,
                "n_rows": frame.height,
                "first": str(frame["timestamp"][0]),
                "last": str(frame["timestamp"][-1]),
                "sha256": sha256_file(path),
            }
        )
        print(f"{symbol}: {frame.height} rows -> {path}")

    manifest = {
        "dataset_id": f"{args.source}-{dt.date.today().isoformat()}",
        "source": args.source,
        "pulled_at": dt.datetime.now(dt.UTC).isoformat(),
        "point_in_time": False,
        "survivorship_biased": True,
        "redistributable": False,
        "symbols": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
