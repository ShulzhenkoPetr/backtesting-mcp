from __future__ import annotations

import datetime as dt
from pathlib import Path

from btmcp.core.costs import CostModel
from btmcp.core.engine import Engine
from btmcp.core.spec import parse_spec
from btmcp.core.store import RunStore
from btmcp.data.barstore import BarStore

SYMBOL = "SYN-04"
IN_WINDOW = (dt.date(2019, 1, 1), dt.date(2021, 12, 31))
OUT_WINDOW = (dt.date(2022, 1, 1), dt.date(2025, 12, 31))
COSTS = CostModel(fee_bps=5.0, slippage_bps=2.0)


def _line(label: str, value: str) -> str:
    return f"  {label:<24} {value}"


def main() -> int:
    store = BarStore()
    engine = Engine(store)
    runs = RunStore(Path("runs"))
    spec = parse_spec({"primitive": "sma_cross", "params": {"fast": 20, "slow": 50}})

    print(f"btmcp demo — dataset {store.dataset_id}, {len(store.universe)} symbols, no network\n")

    print(f"20/50 SMA cross on {SYMBOL}, 5bp fees + 2bp slippage")
    for label, (start, end) in (("in-window", IN_WINDOW), ("after decay", OUT_WINDOW)):
        summary = runs.execute(engine, spec, SYMBOL, start, end, COSTS)
        m = summary.metrics
        print(f"\n  {label}  {start} .. {end}")
        print(_line("sharpe", f"{m.sharpe:+.2f}"))
        print(_line("annualised return", f"{m.annualised_return:+.2%}"))
        print(_line("max drawdown", f"{m.max_drawdown:.2%}"))
        print(_line("trades", str(m.trade_count)))
        print(_line("run_id", summary.run_id[:16]))

    print(
        "\nThe signal is planted through 2021 and decays afterwards, which is the point:\n"
        "the dataset has ground truth, so a wrong backtest is detectable.\n"
    )

    lookahead = store.bars([SYMBOL], *IN_WINDOW, as_of=dt.date(2020, 6, 30))
    print(f"Point-in-time check: asked for {IN_WINDOW[0]}..{IN_WINDOW[1]} as of 2020-06-30")
    print(_line("last bar returned", str(lookahead["timestamp"].max())))
    print("  No tool on this server can return a bar dated after its as_of.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
