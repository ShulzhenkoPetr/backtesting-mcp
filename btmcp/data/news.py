from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from btmcp.core.errors import UnknownSymbol

NEWS_FILE = "news.jsonl"


class NewsItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    symbol: str
    timestamp: dt.date
    headline: str
    body: str
    kind: Literal["benign", "injection"] = "benign"
    payload_id: str | None = None

    @property
    def untrusted(self) -> bool:
        """Every item is untrusted. Provenance is not a function of content."""
        return True


class NewsStore:
    """Dated headlines, filtered by `as_of` on the same terms as bars.

    News carries the injection payloads, so it is the one input that is adversarial
    by construction. It is still point-in-time: an item dated after `as_of` is not
    visible, which is why `INJ-4` cannot smuggle a future bar in through a headline.
    """

    def __init__(self, root: Path | str = Path("datasets/synthetic")) -> None:
        self.root = Path(root)
        path = self.root / NEWS_FILE
        self._items: list[NewsItem] = [
            NewsItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
        ]
        self._symbols = {item.symbol for item in self._items}

    def get(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        as_of: dt.date,
        include_injections: bool = True,
    ) -> list[NewsItem]:
        if symbol not in self._symbols:
            raise UnknownSymbol(
                cause=f"no news coverage for symbol {symbol}",
                suggested_action="call list_symbols; news exists only for symbols in the dataset",
                unknown=[symbol],
            )
        return [
            item
            for item in self._items
            if item.symbol == symbol
            and start <= item.timestamp <= end
            and item.timestamp <= as_of
            and (include_injections or item.kind == "benign")
        ]

    def payloads(self) -> list[NewsItem]:
        return [item for item in self._items if item.kind == "injection"]

    def __len__(self) -> int:
        return len(self._items)


def write_fixture(items: list[NewsItem], root: Path) -> Path:
    path = root / NEWS_FILE
    lines = [json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in items]
    path.write_text("\n".join(lines) + "\n")
    return path
