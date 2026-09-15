## X0 — static token audit

41 legal argument combinations, tokenized offline. Budget 1500 tokens.

| tool | n | p50 | p99 | max | over budget |
|---|---|---|---|---|---|
| `compare_runs` | 1 | 344 | 344 | 344 | 0 |
| `describe_series` | 3 | 110 | 167 | 167 | 0 |
| `explain_trades` | 4 | 445 | 1222 | 1222 | 0 |
| `get_bars` | 27 | 362 | 499 | 499 | 0 |
| `get_news` | 3 | 273 | 642 | 642 | 0 |
| `get_run` | 1 | 741 | 741 | 741 | 0 |
| `list_symbols` | 1 | 644 | 644 | 644 | 0 |
| `run_backtest` | 1 | 874 | 874 | 874 | 0 |

Largest result across the whole grid: **1222 tokens** (budget 1500).
Fraction of requests over budget: **0%**.
