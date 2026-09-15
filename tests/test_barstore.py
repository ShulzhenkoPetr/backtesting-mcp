from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from btmcp.core.errors import InsufficientHistory, UnknownSymbol
from btmcp.data.barstore import BarStore

DATA_START = dt.date(2018, 1, 1)
DATA_END = dt.date(2025, 12, 31)


@pytest.fixture(scope="module")
def store() -> BarStore:
    return BarStore()


dates = st.dates(min_value=DATA_START, max_value=DATA_END)


@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    symbols=st.lists(
        st.sampled_from(["SYN-01", "SYN-04", "SYN-08", "SYN-11", "SYN-17"]),
        min_size=1,
        max_size=3,
        unique=True,
    ),
    a=dates,
    b=dates,
    as_of=dates,
)
def test_no_bar_ever_exceeds_as_of(
    store: BarStore, symbols: list[str], a: dt.date, b: dt.date, as_of: dt.date
) -> None:
    """The core guarantee, over 500 random (symbols, window, as_of) tuples."""
    start, end = min(a, b), max(a, b)
    frame = store.bars(symbols, start=start, end=end, as_of=as_of)
    if frame.height:
        assert frame["timestamp"].max() <= as_of
        assert frame["timestamp"].min() >= start
        assert frame["timestamp"].max() <= end
        assert set(frame["symbol"].unique()).issubset(set(symbols))


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(symbol=st.sampled_from(["SYN-01", "SYN-08", "SYN-17"]), as_of=dates)
def test_coverage_never_reports_beyond_as_of(store: BarStore, symbol: str, as_of: dt.date) -> None:
    cov = store.coverage(symbol, as_of=as_of)
    if cov.last is not None:
        assert cov.last <= as_of
        assert cov.first is not None and cov.first <= cov.last


def test_future_window_returns_empty_rather_than_leaking(store: BarStore) -> None:
    frame = store.bars(["SYN-04"], dt.date(2024, 1, 1), dt.date(2025, 1, 1), as_of=dt.date(2020, 1, 1))
    assert frame.height == 0


def test_as_of_truncates_a_window_that_extends_past_it(store: BarStore) -> None:
    frame = store.bars(["SYN-04"], dt.date(2019, 1, 1), dt.date(2021, 12, 31), as_of=dt.date(2020, 6, 30))
    assert frame.height > 0
    assert frame["timestamp"].max() == dt.date(2020, 6, 30)


def test_symbols_listing_respects_as_of(store: BarStore) -> None:
    early = {s.symbol for s in store.symbols(as_of=dt.date(2018, 1, 5))}
    late = {s.symbol for s in store.symbols(as_of=dt.date(2025, 12, 31))}
    assert early == late == set(store.universe)
    dead = next(s for s in store.symbols(as_of=dt.date(2025, 12, 31)) if s.symbol == "SYN-17")
    assert dead.last == dt.date(2021, 6, 30)


def test_unknown_symbol_names_the_offender(store: BarStore) -> None:
    with pytest.raises(UnknownSymbol) as exc:
        store.bars(["SYN-04", "NOT-A-SYMBOL"], DATA_START, DATA_END, as_of=DATA_END)
    assert "NOT-A-SYMBOL" in exc.value.cause
    assert exc.value.suggested_action


def test_insufficient_history_is_raised_against_min_rows(store: BarStore) -> None:
    with pytest.raises(InsufficientHistory) as exc:
        store.bars(
            ["SYN-04"], dt.date(2019, 1, 1), dt.date(2019, 1, 10), as_of=dt.date(2019, 1, 5), min_rows=50
        )
    assert exc.value.context["required"] == 50
    assert "SYN-04" in exc.value.context["available"]


def test_dead_symbol_yields_no_rows_after_its_last_bar(store: BarStore) -> None:
    frame = store.bars(["SYN-17"], dt.date(2022, 1, 1), dt.date(2023, 1, 1), as_of=dt.date(2023, 1, 1))
    assert frame.height == 0


def test_gaps_are_detected_for_the_sparse_symbol(store: BarStore) -> None:
    assert store.coverage("SYN-08", as_of=DATA_END).gaps
    assert not store.coverage("SYN-01", as_of=DATA_END).gaps


def test_every_bar_query_in_the_codebase_binds_as_of() -> None:
    """Structural guard: new SQL over bars that forgets as_of fails here, not in review.

    Scoped to SQL over parquet. Reads of run artefacts (equity curves, blotters) are
    not bar reads and have no as_of dimension, so matching them would only train the
    reader to ignore this test.
    """
    import ast

    offenders: list[str] = []
    for path in Path("btmcp").rglob("*.py"):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = ast.get_source_segment(source, node) or ""
            if "read_parquet" not in body or "SELECT" not in body.upper():
                continue
            if "$as_of" not in body and "_AS_OF_PREDICATE" not in body:
                offenders.append(f"{path}:{node.lineno} {node.name}")
    assert not offenders, "function queries bars without binding as_of:\n" + "\n".join(offenders)


def test_bar_data_is_only_reachable_through_the_store() -> None:
    """Nothing outside the data layer may open bar files itself.

    Checks actual reads, not configuration: a module is free to name the dataset
    directory as a default path, but only `btmcp/data` may query it.
    """
    import ast

    offenders: list[str] = []
    for path in Path("btmcp").rglob("*.py"):
        if path.parent.name == "data":
            continue
        source = path.read_text()
        if "duckdb" in source:
            offenders.append(f"{path}: imports duckdb outside the data layer")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = ast.get_source_segment(source, node) or ""
            if "read_parquet" in body and ("dataset_root" in body or "datasets/" in body):
                offenders.append(f"{path}:{node.lineno} {node.name} reads bar files directly")
    assert not offenders, "bar access outside btmcp/data:\n" + "\n".join(offenders)


def test_the_as_of_guard_test_can_actually_fail(tmp_path: Path) -> None:
    """Guards the guard: an unbounded query must be detectable, or the test above is theatre."""
    import ast

    bad = tmp_path / "bad.py"
    bad.write_text("def leak(con):\n    return con.execute('SELECT * FROM read_parquet($files)')\n")
    tree = ast.parse(bad.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    body = ast.get_source_segment(bad.read_text(), fn) or ""
    assert "read_parquet" in body
    assert "$as_of" not in body and "_AS_OF_PREDICATE" not in body


def test_barstore_exposes_no_unbounded_read(store: BarStore) -> None:
    """No public method may return bars without an as_of argument."""
    import inspect

    for name in ("bars", "coverage", "symbols"):
        params = inspect.signature(getattr(BarStore, name)).parameters
        assert "as_of" in params, f"{name} lacks as_of"
        assert params["as_of"].default is inspect.Parameter.empty, f"{name} gives as_of a default"
