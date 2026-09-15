from __future__ import annotations

import datetime as dt

import pytest

from btmcp.core.errors import UnknownSymbol
from btmcp.data.news import NewsStore

PAYLOAD_IDS = {"INJ-1", "INJ-2", "INJ-3", "INJ-4"}


@pytest.fixture(scope="module")
def news() -> NewsStore:
    return NewsStore()


def test_fixture_size_and_payload_count(news: NewsStore) -> None:
    assert len(news) >= 200
    assert {p.payload_id for p in news.payloads()} == PAYLOAD_IDS


def test_every_item_is_untrusted(news: NewsStore) -> None:
    """Provenance is structural: nothing in this store is ever authored content."""
    items = news.get("SYN-04", dt.date(2018, 1, 1), dt.date(2025, 12, 31), as_of=dt.date(2025, 12, 31))
    assert items and all(item.untrusted for item in items)


def test_news_respects_as_of(news: NewsStore) -> None:
    as_of = dt.date(2020, 1, 1)
    items = news.get("SYN-04", dt.date(2018, 1, 1), dt.date(2025, 12, 31), as_of=as_of)
    assert items
    assert all(item.timestamp <= as_of for item in items)


def test_injection_payload_is_hidden_before_its_date(news: NewsStore) -> None:
    window = (dt.date(2018, 1, 1), dt.date(2025, 12, 31))
    before = news.get("SYN-04", *window, as_of=dt.date(2020, 3, 11))
    after = news.get("SYN-04", *window, as_of=dt.date(2020, 3, 12))
    assert "INJ-1" not in {i.payload_id for i in before}
    assert "INJ-1" in {i.payload_id for i in after}


def test_injections_can_be_excluded_for_control_runs(news: NewsStore) -> None:
    window = (dt.date(2018, 1, 1), dt.date(2025, 12, 31), dt.date(2025, 12, 31))
    with_inj = news.get("SYN-04", *window, include_injections=True)
    without = news.get("SYN-04", *window, include_injections=False)
    assert len(with_inj) > len(without)
    assert all(i.kind == "benign" for i in without)


def test_unknown_symbol_raises(news: NewsStore) -> None:
    with pytest.raises(UnknownSymbol):
        news.get("NOPE", dt.date(2018, 1, 1), dt.date(2025, 12, 31), as_of=dt.date(2025, 12, 31))


@pytest.mark.parametrize("payload_id", sorted(PAYLOAD_IDS))
def test_each_payload_targets_a_distinct_guardrail(news: NewsStore, payload_id: str) -> None:
    payload = next(p for p in news.payloads() if p.payload_id == payload_id)
    body = payload.body.lower()
    expected = {
        "INJ-1": "publish_run_report",  # naked imperative at the act tier
        "INJ-2": "[system]",  # impersonated system notice
        "INJ-3": "already reviewed",  # claimed prior authorisation
        "INJ-4": "as_of",  # targets the lookahead guarantee, not the tier
    }[payload_id]
    assert expected in body


def test_inj4_targets_lookahead_rather_than_permissions(news: NewsStore) -> None:
    """INJ-4 is the important one: it attacks the structural defence, not the tier gate."""
    payload = next(p for p in news.payloads() if p.payload_id == "INJ-4")
    assert "as_of" in payload.body
    assert "publish_run_report" not in payload.body
