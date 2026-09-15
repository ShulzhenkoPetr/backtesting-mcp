from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from btmcp.eval.schema import DIALECTS, SchemaDialect

LADDER_FILE = Path("btmcp/eval/ladder.yaml")


@dataclass(frozen=True)
class Quota:
    """A free provider's allowance. Spend is measured against this instead of money."""

    daily_tokens: int | None = None
    daily_requests: int | None = None
    per_minute_requests: int | None = None


@dataclass(frozen=True)
class Rung:
    name: str
    provider: str
    model: str
    rank: int
    base_url: str | None = None
    api_key_env: str | None = None
    dialect: str = "anthropic"
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    quota: Quota | None = None
    effort: str | None = None
    max_tokens: int = 8000
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("name", "provider", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"rung {field_name!r} must be a non-empty string, got {value!r} "
                    "(YAML parses an unquoted `null` as nothing)"
                )
        if self.dialect not in DIALECTS:
            raise ValueError(f"unknown dialect {self.dialect!r}; known: {', '.join(sorted(DIALECTS))}")

    @property
    def free(self) -> bool:
        return self.input_per_mtok is None and self.output_per_mtok is None

    @property
    def schema_dialect(self) -> SchemaDialect:
        return DIALECTS[self.dialect]

    def credential(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def available(self) -> bool:
        """A rung with no configured credential cannot be swept; report rather than fail."""
        if self.provider == "null":
            return True
        if self.api_key_env is None:
            return True
        return bool(self.credential())


class Ladder:
    """An ordered, open set of capability rungs.

    Rungs live in YAML rather than in code so a new provider or model can be added to
    the capability sweep without touching the runner. `rank` orders them by expected
    capability, which is what X5 varies; names are free-form labels.
    """

    def __init__(self, rungs: list[Rung]) -> None:
        self._rungs = {r.name: r for r in rungs}
        if len(self._rungs) != len(rungs):
            duplicates = [r.name for r in rungs if list(r.name for r in rungs).count(r.name) > 1]
            raise ValueError(f"duplicate rung name(s): {sorted(set(duplicates))}")

    def __len__(self) -> int:
        return len(self._rungs)

    def __contains__(self, name: str) -> bool:
        return name in self._rungs

    def __iter__(self):
        return iter(self.ordered())

    def get(self, name: str) -> Rung:
        if name in self._rungs:
            return self._rungs[name]
        raise KeyError(f"unknown rung '{name}'; known rungs: {', '.join(sorted(self._rungs))}")

    def ordered(self) -> list[Rung]:
        return sorted(self._rungs.values(), key=lambda r: (r.rank, r.name))

    def available(self) -> list[Rung]:
        return [r for r in self.ordered() if r.available()]

    def names(self) -> list[str]:
        return [r.name for r in self.ordered()]


def _rung(data: dict[str, Any]) -> Rung:
    quota_data = data.pop("quota", None)
    quota = Quota(**quota_data) if quota_data else None
    return Rung(quota=quota, **data)


def load_ladder(path: Path | None = None) -> Ladder:
    source = path or LADDER_FILE
    data = yaml.safe_load(source.read_text())
    return Ladder([_rung(dict(entry)) for entry in data["rungs"]])


def resolve(spec: str, ladder: Ladder | None = None) -> Rung:
    """Accepts a rung name, or a raw `provider:model` for a model not on the ladder."""
    rungs = ladder or load_ladder()
    if spec in rungs:
        return rungs.get(spec)
    if ":" in spec:
        provider, _, model = spec.partition(":")
        template = next((r for r in rungs.ordered() if r.provider == provider), None)
        if template is None:
            raise KeyError(f"no rung configured for provider '{provider}'")
        return replace(template, name=spec, model=model)
    raise KeyError(f"unknown model '{spec}'; use a rung name ({', '.join(rungs.names())}) or provider:model")
