from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

ENGINE_VERSION = "1.0.0"


def _canonical(value: Any) -> Any:
    """Reduces a value to JSON primitives with a platform-stable float encoding.

    Floats become strings because JSON has no canonical float form: `json.dumps`
    would render 1.0 and 1 identically on some paths, and repr digits are the only
    shortest-roundtrip form guaranteed stable across CPython builds. NaN and the
    infinities get explicit tokens rather than the non-standard JSON literals, and
    -0.0 folds into 0.0 so that a sign that no arithmetic here can observe cannot
    change a hash.
    """
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "f:nan"
        if math.isinf(value):
            return "f:inf" if value > 0 else "f:-inf"
        return f"f:{value + 0.0!r}"
    if isinstance(value, str):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, Sequence | set | frozenset):
        items = sorted(value, key=repr) if isinstance(value, set | frozenset) else value
        return [_canonical(v) for v in items]
    raise TypeError(f"cannot canonicalise {type(value).__name__} for hashing")


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(
    spec: Any,
    dataset_id: str,
    costs: Any,
    engine_version: str = ENGINE_VERSION,
    **extra: Any,
) -> str:
    """Content address for a run. Stable across processes, platforms and key order."""
    payload = {
        "spec": spec,
        "dataset_id": dataset_id,
        "costs": costs,
        "engine_version": engine_version,
        "extra": extra,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
