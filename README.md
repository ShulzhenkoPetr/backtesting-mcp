# btmcp — a point-in-time backtesting MCP server

Two tool surfaces over one backtesting core, built to measure how reliably language
models use each one. The backtester is deliberately modest. 

## Get started

```bash
git clone <repo> && cd backtesting-mcp && uv sync && uv run btmcp-demo
```

No API key, no network. Runs a real backtest on the committed synthetic dataset and
prints the point-in-time check.
Two surfaces sit on the same core: A exposes nine granular tools, B consolidates them into four.

```bash
uv run btmcp-serve --surface a --tier simulate
```

Measure what each tool puts in a context window, with no API key and no cost:

```bash
uv run btmcp-audit --out results/x0/
```

Smoke-test the whole eval sweep at zero cost, then read the paired report:

```bash
uv run btmcp-eval --surface a --model null --split dev --seeds 1 --out results/smoke/a
```

## Protocol

Built against **MCP Python SDK 2.1.1**. Protocol version: **`2026-07-28`**

## Scope

**In:** daily bars, ≤20 symbol universe, long/flat positions, declarative strategy
specs, one cost model, deterministic runs.

**Out:** intraday, shorting, portfolio optimisation, corporate actions beyond
adjusted closes, survivorship-bias correction, live data, order routing, arbitrary
user code.

This is a research harness. It is not a trading system and it is not investment advice.
