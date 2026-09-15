from __future__ import annotations

import datetime as dt
from functools import cached_property
from itertools import pairwise
from pathlib import Path

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict

from btmcp.core.errors import InsufficientHistory, UnknownSymbol
from btmcp.data.manifest import load_manifest

_AS_OF_PREDICATE = "timestamp <= $as_of"


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    first: dt.date | None
    last: dt.date | None
    n_rows: int
    gaps: list[tuple[dt.date, dt.date]]

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0


class SymbolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    first: dt.date
    last: dt.date
    n_rows: int
    note: str = ""


class BarStore:
    """Point-in-time bar access. Every read is bounded by `as_of` inside the SQL.

    The lookahead guarantee is structural rather than validated: there is no public
    method that returns bars without an `as_of`, the parameter has no default, and
    the `timestamp <= ?` predicate is appended by `_query` rather than by callers.
    A caller cannot construct a request for tomorrow's bar and get it; the worst it
    can do is ask for a window that is entirely in the future and receive nothing.
    """

    def __init__(self, root: Path | str = Path("datasets/synthetic")) -> None:
        self.root = Path(root)
        self._con = duckdb.connect(":memory:")
        # The progress bar writes to stdout, which is the MCP stdio transport.
        self._con.execute("SET enable_progress_bar = false")
        self._manifest = load_manifest(self.root)
        self.dataset_id = self._manifest.dataset_id
        self._notes = {r.params.symbol: r.params.note for r in self._manifest.symbols}

    @cached_property
    def _known(self) -> dict[str, Path]:
        return {r.params.symbol: self.root / f"{r.params.symbol}.parquet" for r in self._manifest.symbols}

    def _resolve(self, symbols: list[str]) -> list[Path]:
        unknown = [s for s in symbols if s not in self._known]
        if unknown:
            raise UnknownSymbol(
                cause=f"symbol(s) not in dataset {self.dataset_id}: {', '.join(sorted(unknown))}",
                suggested_action="call list_symbols to see the available universe",
                unknown=sorted(unknown),
            )
        return [self._known[s] for s in symbols]

    def _query(
        self,
        paths: list[Path],
        as_of: dt.date,
        start: dt.date | None,
        end: dt.date | None,
        columns: str = "*",
    ) -> pl.DataFrame:
        """The one place bar SQL is built. The as_of predicate is not optional here."""
        clauses = [_AS_OF_PREDICATE]
        params: dict[str, object] = {"files": [str(p) for p in paths], "as_of": as_of}
        if start is not None:
            clauses.append("timestamp >= $start")
            params["start"] = start
        if end is not None:
            clauses.append("timestamp <= $end")
            params["end"] = end

        sql = (
            f"SELECT {columns} FROM read_parquet($files) "
            f"WHERE {' AND '.join(clauses)} ORDER BY symbol, timestamp"
        )
        return self._con.execute(sql, params).pl()

    def bars(
        self,
        symbols: list[str],
        start: dt.date,
        end: dt.date,
        as_of: dt.date,
        min_rows: int = 0,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame(schema={"symbol": pl.String(), "timestamp": pl.Date()})
        paths = self._resolve(symbols)
        frame = self._query(paths, as_of=as_of, start=start, end=end)

        if min_rows:
            counts = (
                frame.group_by("symbol").len().to_dict(as_series=False)
                if frame.height
                else {"symbol": [], "len": []}
            )
            available = dict(zip(counts["symbol"], counts["len"], strict=True))
            short = {s: available.get(s, 0) for s in symbols if available.get(s, 0) < min_rows}
            if short:
                detail = ", ".join(f"{s} has {n}" for s, n in sorted(short.items()))
                raise InsufficientHistory(
                    cause=(
                        f"{min_rows} bars required per symbol in {start}..{end} as of {as_of}, but {detail}"
                    ),
                    suggested_action=(
                        "widen the window, move as_of later, reduce warmup_bars, "
                        "or call describe_series to see actual coverage"
                    ),
                    required=min_rows,
                    available=short,
                    as_of=as_of.isoformat(),
                )
        return frame

    def coverage(self, symbol: str, as_of: dt.date) -> Coverage:
        paths = self._resolve([symbol])
        frame = self._query(paths, as_of=as_of, start=None, end=None, columns="timestamp")
        if frame.height == 0:
            return Coverage(symbol=symbol, first=None, last=None, n_rows=0, gaps=[])

        stamps = frame["timestamp"]
        gaps: list[tuple[dt.date, dt.date]] = []
        days = stamps.to_list()
        for prev, nxt in pairwise(days):
            business_gap = _business_days_between(prev, nxt)
            if business_gap > 1:
                gaps.append((prev, nxt))
        return Coverage(symbol=symbol, first=days[0], last=days[-1], n_rows=frame.height, gaps=gaps)

    def symbols(self, as_of: dt.date) -> list[SymbolInfo]:
        infos: list[SymbolInfo] = []
        for symbol in sorted(self._known):
            cov = self.coverage(symbol, as_of=as_of)
            if cov.first is None or cov.last is None:
                continue
            infos.append(
                SymbolInfo(
                    symbol=symbol,
                    first=cov.first,
                    last=cov.last,
                    n_rows=cov.n_rows,
                    note=self._notes.get(symbol, ""),
                )
            )
        return infos

    def note(self, symbol: str) -> str:
        return self._notes.get(symbol, "")

    @property
    def universe(self) -> list[str]:
        return sorted(self._known)


def _business_days_between(a: dt.date, b: dt.date) -> int:
    """Counts weekday steps, so a normal Friday-to-Monday step is not reported as a gap."""
    days = 0
    cursor = a
    while cursor < b:
        cursor += dt.timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days
