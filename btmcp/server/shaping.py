from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict

DEFAULT_BUDGET_TOKENS = 1500
ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Offline token count.

    A BPE tokenizer, not the served model's own. Every number in this repo is a
    comparison between surfaces measured the same way, so a consistent proxy is
    sufficient; it is not claimed to equal a provider's billed count.
    """
    return len(_encoder().encode(text))


class ResourceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    name: str
    description: str
    mime_type: str = "application/vnd.apache.parquet"


class Shaped(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]
    links: list[ResourceLink]
    truncated: bool
    result_tokens: int


def serialise(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, default=str, separators=(",", ":"))


def shape(
    payload: Any,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    resource_factory: Callable[[], list[ResourceLink]] | None = None,
    summariser: Callable[[Any], dict[str, Any]] | None = None,
) -> Shaped:
    """The single chokepoint every tool result passes through.

    Row-bearing payloads never reach a caller in full: they are summarised and
    addressed by link regardless of size, so the budget cannot be defeated by a
    request that merely happens to be small today. Non-row payloads are returned
    whole unless they exceed the budget, in which case the summariser replaces them.
    """
    links = resource_factory() if resource_factory else []

    body = summariser(payload) if summariser else _as_dict(payload)
    if links:
        body = {**body, "resources": [link.model_dump() for link in links]}
    text = serialise(body)
    tokens = count_tokens(text)

    if tokens <= budget_tokens:
        return Shaped(payload=body, links=links, truncated=summariser is not None, result_tokens=tokens)

    trimmed = _fallback_summary(body, budget_tokens)
    return Shaped(
        payload=trimmed,
        links=links,
        truncated=True,
        result_tokens=count_tokens(serialise(trimmed)),
    )


def _as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _fallback_summary(body: dict[str, Any], budget_tokens: int) -> dict[str, Any]:
    """Drops the largest fields until the payload fits, naming what it dropped."""
    kept = dict(body)
    dropped: list[str] = []
    sizes = sorted(kept, key=lambda k: count_tokens(serialise(kept[k])), reverse=True)

    for key in sizes:
        if count_tokens(serialise(kept)) <= budget_tokens:
            break
        if len(kept) == 1:
            break
        kept.pop(key)
        dropped.append(key)

    if dropped:
        kept["_omitted_fields"] = dropped
        kept["_note"] = (
            "Fields omitted to fit the result budget. Fetch the linked resource for the full artefact."
        )
    return kept
