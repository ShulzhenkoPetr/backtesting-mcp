from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from btmcp.core.costs import CostModel
from btmcp.core.engine import Engine
from btmcp.core.errors import InsufficientHistory, RunNotFound, UnknownSymbol
from btmcp.core.hashing import ENGINE_VERSION
from btmcp.core.metrics import Metrics
from btmcp.core.spec import PRIMITIVES, StrategySpec, parse_spec
from btmcp.core.store import RunStore, RunSummary
from btmcp.data.barstore import BarStore
from btmcp.data.news import NewsStore
from btmcp.server.guardrails.audit import AuditLog
from btmcp.server.guardrails.auth import TierPolicy
from btmcp.server.guardrails.fencing import FencedText, FencingConfig, fence
from btmcp.server.shaping import DEFAULT_BUDGET_TOKENS, ResourceLink, Shaped, shape

TradeFilter = Literal["worst_5", "best_5", "largest_drawdown", "all_in_window"]


class SymbolList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    as_of: dt.date
    count: int
    symbols: list[dict[str, Any]]


class SeriesProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    as_of: dt.date
    first: dt.date | None
    last: dt.date | None
    n_rows: int
    gap_count: int
    gaps_sample: list[str]
    frequency: str
    last_close: float | None
    annualised_vol: float | None
    max_drawdown: float | None
    note: str


class BarSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str]
    as_of: dt.date
    start: dt.date
    end: dt.date
    n_rows: int
    per_symbol: list[dict[str, Any]]
    fields: list[str]
    note: str = "Rows are not inlined. Fetch the linked resource for the full frame."


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    cause: str
    suggested_action: str
    severity: Literal["error", "warning"] = "error"


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    symbol: str
    spec_echo: dict[str, Any] | None
    required_bars: int
    available_bars: int
    issues: list[ValidationIssue]


class MetricDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: list[str]
    metrics: list[str]
    rows: list[dict[str, Any]]
    best_by_sharpe: str | None


class TradeExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    filter: str
    n_selected: int
    trades: list[dict[str, Any]]
    commentary: str


@dataclass
class AppConfig:
    surface: str = "a"
    tier: str = "simulate"
    dataset_root: Path = field(default_factory=lambda: Path("datasets/synthetic"))
    runs_root: Path = field(default_factory=lambda: Path("runs"))
    audit_path: Path | None = None
    fencing: FencingConfig = field(default_factory=FencingConfig)
    budget_tokens: int = DEFAULT_BUDGET_TOKENS


class Services:
    """The layer both surfaces sit on, so A and B differ only in tool shape.

    X2 compares surfaces; if they did not share this object the comparison would be
    confounded by two implementations of the same behaviour.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.bars = BarStore(config.dataset_root)
        self.news = NewsStore(config.dataset_root)
        self.runs = RunStore(config.runs_root)
        self.engine = Engine(self.bars)
        self.policy = TierPolicy(config.tier)
        self.audit = AuditLog(config.audit_path, config.surface, config.tier, self.bars.dataset_id)

    # ---- read tier -------------------------------------------------------

    def list_symbols(self, as_of: dt.date) -> SymbolList:
        infos = self.bars.symbols(as_of=as_of)
        return SymbolList(
            dataset_id=self.bars.dataset_id,
            as_of=as_of,
            count=len(infos),
            symbols=[
                {
                    "symbol": i.symbol,
                    "first": i.first.isoformat(),
                    "last": i.last.isoformat(),
                    "n_rows": i.n_rows,
                }
                for i in infos
            ],
        )

    def describe_series(self, symbol: str, as_of: dt.date) -> SeriesProfile:
        import numpy as np

        coverage = self.bars.coverage(symbol, as_of=as_of)
        note = self.bars.note(symbol)
        if coverage.n_rows == 0:
            return SeriesProfile(
                symbol=symbol,
                as_of=as_of,
                first=None,
                last=None,
                n_rows=0,
                gap_count=0,
                gaps_sample=[],
                frequency="daily",
                last_close=None,
                annualised_vol=None,
                max_drawdown=None,
                note=note or "No bars visible at this as_of.",
            )

        frame = self.bars.bars([symbol], coverage.first, coverage.last, as_of=as_of)  # type: ignore[arg-type]
        closes = frame["close"].to_numpy()
        returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([])
        drawdown = float((closes / np.maximum.accumulate(closes) - 1.0).min()) if len(closes) else None
        return SeriesProfile(
            symbol=symbol,
            as_of=as_of,
            first=coverage.first,
            last=coverage.last,
            n_rows=coverage.n_rows,
            gap_count=len(coverage.gaps),
            gaps_sample=[f"{a}..{b}" for a, b in coverage.gaps[:5]],
            frequency="daily",
            last_close=float(closes[-1]),
            annualised_vol=float(returns.std(ddof=1) * (252**0.5)) if len(returns) > 1 else None,
            max_drawdown=drawdown,
            note=note,
        )

    def get_bars(
        self,
        symbols: list[str],
        start: dt.date,
        end: dt.date,
        as_of: dt.date,
        fields: list[str] | None = None,
    ) -> tuple[BarSummary, list[ResourceLink]]:
        chosen = fields or ["open", "high", "low", "close", "volume"]
        frame = self.bars.bars(symbols, start, end, as_of=as_of)
        per_symbol: list[dict[str, Any]] = []
        for symbol in symbols:
            part = frame.filter(frame["symbol"] == symbol)
            if part.height == 0:
                per_symbol.append({"symbol": symbol, "n_rows": 0})
                continue
            per_symbol.append(
                {
                    "symbol": symbol,
                    "n_rows": part.height,
                    "first": str(part["timestamp"][0]),
                    "last": str(part["timestamp"][-1]),
                    "first_close": round(float(part["close"][0]), 4),
                    "last_close": round(float(part["close"][-1]), 4),
                    "min_close": round(float(part["close"].min()), 4),
                    "max_close": round(float(part["close"].max()), 4),
                }
            )
        summary = BarSummary(
            symbols=symbols,
            as_of=as_of,
            start=start,
            end=end,
            n_rows=frame.height,
            per_symbol=per_symbol,
            fields=chosen,
        )
        link = ResourceLink(
            uri=f"btmcp://bars/{','.join(symbols)}/{start}/{end}/{as_of}",
            name="bars",
            description=f"{frame.height} daily bars for {len(symbols)} symbol(s) as of {as_of}",
        )
        return summary, [link]

    def get_news(self, symbol: str, start: dt.date, end: dt.date, as_of: dt.date) -> FencedText:
        items = self.news.get(symbol, start, end, as_of=as_of)
        blocks = [f"[{item.timestamp}] {item.headline}\n{item.body}" for item in items]
        return fence(blocks, source=f"news:{symbol}", config=self.config.fencing)

    # ---- simulate tier ---------------------------------------------------

    def validate_strategy(
        self, spec_payload: dict[str, Any], symbol: str, as_of: dt.date
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        spec: StrategySpec | None = None
        try:
            spec = parse_spec(spec_payload)
        except Exception as exc:
            payload = getattr(exc, "to_payload", lambda: {})()
            issues.append(
                ValidationIssue(
                    code=payload.get("code", "INVALID_SPEC"),
                    cause=payload.get("cause", str(exc)),
                    suggested_action=payload.get(
                        "suggested_action", f"valid primitives: {', '.join(PRIMITIVES)}"
                    ),
                )
            )

        available = 0
        required = 0
        try:
            coverage = self.bars.coverage(symbol, as_of=as_of)
            available = coverage.n_rows
        except UnknownSymbol as exc:
            issues.append(
                ValidationIssue(code=exc.code, cause=exc.cause, suggested_action=exc.suggested_action)
            )

        if spec is not None:
            required = max(spec.min_bars, spec.warmup_bars) + 2
            if available and available < required:
                issues.append(
                    ValidationIssue(
                        code="INSUFFICIENT_HISTORY",
                        cause=f"{symbol} has {available} bars as of {as_of}; the spec needs {required}",
                        suggested_action=(
                            "widen the window, lower warmup_bars, or use shorter moving averages"
                        ),
                    )
                )

        return ValidationReport(
            ok=not issues,
            symbol=symbol,
            spec_echo=spec.model_dump(mode="json") if spec else None,
            required_bars=required,
            available_bars=available,
            issues=issues,
        )

    def run_backtest(
        self,
        spec_payload: dict[str, Any],
        symbol: str,
        start: dt.date,
        end: dt.date,
        costs: CostModel,
    ) -> tuple[RunSummary, list[ResourceLink]]:
        spec = parse_spec(spec_payload)
        summary = self.runs.execute(self.engine, spec, symbol, start, end, costs)
        links = [
            ResourceLink(uri=summary.equity_uri, name="equity_curve", description="Daily equity curve."),
            ResourceLink(uri=summary.blotter_uri, name="blotter", description="Executed trades."),
        ]
        return summary, links

    # ---- inspection ------------------------------------------------------

    def get_run(self, run_id: str) -> RunSummary:
        return self.runs.get(run_id)

    def compare_runs(self, run_ids: list[str]) -> MetricDiff:
        if len(run_ids) < 2:
            raise RunNotFound(
                cause="compare_runs needs at least two run ids",
                suggested_action="pass the run_id from two or more completed backtests",
                given=run_ids,
            )
        summaries = [self.runs.get(rid) for rid in run_ids]
        names = [
            "total_return",
            "annualised_return",
            "annualised_vol",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "trade_count",
            "exposure_fraction",
        ]
        rows = [
            {
                "run_id": s.run_id[:12],
                "symbol": s.symbol,
                "primitive": s.spec.primitive,
                **{n: _round(getattr(s.metrics, n)) for n in names},
            }
            for s in summaries
        ]
        best = max(summaries, key=lambda s: s.metrics.sharpe)
        return MetricDiff(run_ids=run_ids, metrics=names, rows=rows, best_by_sharpe=best.run_id)

    def explain_trades(self, run_id: str, trade_filter: TradeFilter) -> TradeExplanation:
        summary = self.runs.get(run_id)
        blotter = self.runs.trades(run_id)
        equity = self.runs.equity(run_id)

        if blotter.height == 0:
            return TradeExplanation(
                run_id=run_id,
                filter=trade_filter,
                n_selected=0,
                trades=[],
                commentary="This run executed no trades.",
            )

        rows = blotter.to_dicts()
        for row in rows:
            row["timestamp"] = str(row["timestamp"])

        if trade_filter == "largest_drawdown":
            dd = summary.metrics.drawdown
            selected = (
                [
                    r
                    for r in rows
                    if dd.peak_date
                    and dd.trough_date
                    and str(dd.peak_date) <= r["timestamp"] <= str(dd.trough_date)
                ]
                if dd.peak_date
                else []
            )
            commentary = (
                f"Trades between the drawdown peak ({dd.peak_date}) "
                f"and trough ({dd.trough_date}), {dd.max_drawdown:.2%}."
            )
        elif trade_filter in {"worst_5", "best_5"}:
            ranked = _rank_trades(rows, equity)
            reverse = trade_filter == "best_5"
            selected = sorted(ranked, key=lambda r: r.get("pnl_pct", 0.0), reverse=reverse)[:5]
            label = "best" if reverse else "worst"
            commentary = f"The {len(selected)} {label} round trips by return."
        else:
            selected = rows[:50]
            commentary = f"First {len(selected)} of {len(rows)} trades in the run window."

        return TradeExplanation(
            run_id=run_id,
            filter=trade_filter,
            n_selected=len(selected),
            trades=selected,
            commentary=commentary,
        )

    # ---- shaping ---------------------------------------------------------

    def shaped(self, payload: Any, links: list[ResourceLink] | None = None) -> Shaped:
        return shape(payload, budget_tokens=self.config.budget_tokens, resource_factory=lambda: links or [])


def _round(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def _rank_trades(rows: list[dict[str, Any]], equity: Any) -> list[dict[str, Any]]:
    """Pairs each buy with the following sell and attributes the equity move between them."""
    curve = {
        str(t): e for t, e in zip(equity["timestamp"].to_list(), equity["equity"].to_list(), strict=True)
    }
    ranked: list[dict[str, Any]] = []
    open_row: dict[str, Any] | None = None
    for row in rows:
        if row["weight_after"] > row["weight_before"]:
            open_row = row
        elif open_row is not None:
            entry, exit_ = curve.get(open_row["timestamp"]), curve.get(row["timestamp"])
            if entry and exit_:
                merged = dict(open_row)
                merged["exit_timestamp"] = row["timestamp"]
                merged["pnl_pct"] = round((exit_ / entry - 1.0) * 100.0, 4)
                ranked.append(merged)
            open_row = None
    return ranked


__all__ = [
    "ENGINE_VERSION",
    "AppConfig",
    "BarSummary",
    "InsufficientHistory",
    "MetricDiff",
    "Metrics",
    "SeriesProfile",
    "Services",
    "SymbolList",
    "TradeExplanation",
    "TradeFilter",
    "ValidationIssue",
    "ValidationReport",
]
