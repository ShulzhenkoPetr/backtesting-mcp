from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from scripts.fetch_real import parse_stooq

SAMPLE = """Date,Open,High,Low,Close,Volume
2024-01-02,187.15,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
2024-01-04,182.15,183.09,180.88,181.91,71983600
"""


def test_parse_stooq_normalises_onto_the_synthetic_schema() -> None:
    frame = parse_stooq("AAPL", SAMPLE)
    assert frame.columns == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert frame.height == 3
    assert frame["timestamp"][0] == dt.date(2024, 1, 2)
    assert frame["symbol"].unique().to_list() == ["AAPL"]
    assert frame["timestamp"].is_sorted()


def test_parse_stooq_rejects_an_empty_response() -> None:
    with pytest.raises(ValueError, match="no data"):
        parse_stooq("NOPE", "No data\n")


def test_parse_stooq_rejects_unexpected_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_stooq("AAPL", "Date,Close\n2024-01-02,185.64\n")


def test_suite_does_not_depend_on_real_data() -> None:
    """E1-T4 acceptance: everything must pass with datasets/real/ absent."""
    assert not Path("datasets/real").exists() or not any(Path("datasets/real").glob("*.parquet"))
