from __future__ import annotations

from typing import Annotated, Any, Literal, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from btmcp.core.errors import InvalidSpec

Primitive = Literal["buy_and_hold", "sma_cross", "threshold", "forecast_signal"]
PRIMITIVES: tuple[str, ...] = get_args(Primitive)


class EmptyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SmaCrossParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast: int = Field(ge=2, le=200, description="Fast moving-average window in bars.")
    slow: int = Field(ge=3, le=400, description="Slow moving-average window in bars.")

    @model_validator(mode="after")
    def _fast_below_slow(self) -> SmaCrossParams:
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be strictly less than slow ({self.slow})")
        return self


class ThresholdParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback: int = Field(ge=2, le=504, description="Bars over which the trailing return is measured.")
    entry_pct: float = Field(description="Go long when the trailing return exceeds this, in percent.")
    exit_pct: float = Field(description="Go flat when the trailing return falls below this, in percent.")

    @model_validator(mode="after")
    def _entry_above_exit(self) -> ThresholdParams:
        if self.entry_pct < self.exit_pct:
            raise ValueError(f"entry_pct ({self.entry_pct}) must be >= exit_pct ({self.exit_pct})")
        return self


class ForecastParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecaster_id: str = Field(description="Registered forecaster, e.g. 'seasonal_naive' or 'drift'.")
    horizon: int = Field(ge=1, le=63, description="Forecast horizon in bars.")
    quantile: float = Field(gt=0, lt=1, description="Predictive quantile the signal reads.")
    threshold: float = Field(
        description="Go long when the forecast return at that quantile exceeds this, in percent."
    )


Params = Union[EmptyParams, SmaCrossParams, ThresholdParams, ForecastParams]  # noqa: UP007

PARAMS_FOR: dict[str, type[BaseModel]] = {
    "buy_and_hold": EmptyParams,
    "sma_cross": SmaCrossParams,
    "threshold": ThresholdParams,
    "forecast_signal": ForecastParams,
}


class StrategySpec(BaseModel):
    """Declarative strategy description. `extra="forbid"` throughout, deliberately.

    A hallucinated field is a failure worth surfacing: silently dropping it would let
    a model believe it configured something it did not, and the eval could not tell
    the difference between a spec that was understood and one that was ignored.
    """

    model_config = ConfigDict(extra="forbid")

    primitive: Primitive = Field(description="Which strategy primitive to run.")
    params: Params = Field(description="Parameters for the chosen primitive. Use {} for buy_and_hold.")
    warmup_bars: int = Field(
        default=0, ge=0, le=504, description="Bars of history required before the first decision."
    )
    rebalance: Literal["daily", "weekly", "monthly"] = "daily"
    sizing: Literal["full", "fixed_fraction"] = "full"
    fraction: float | None = Field(
        default=None, gt=0, le=1, description="Required when sizing is fixed_fraction."
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_params_to_the_primitive(cls, data: Any) -> Any:
        """Selects the params model from `primitive` so errors name one type, not four."""
        if not isinstance(data, dict):
            return data
        primitive = data.get("primitive")
        if primitive not in PARAMS_FOR:
            return data
        params = data.get("params")
        if params is None:
            params = {}
        if isinstance(params, dict):
            data = {**data, "params": PARAMS_FOR[primitive].model_validate(params)}
        return data

    @model_validator(mode="after")
    def _params_match_primitive(self) -> StrategySpec:
        expected = PARAMS_FOR[self.primitive]
        if not isinstance(self.params, expected):
            raise ValueError(
                f"primitive '{self.primitive}' requires {expected.__name__}, got {type(self.params).__name__}"
            )
        if self.sizing == "fixed_fraction" and self.fraction is None:
            raise ValueError("sizing 'fixed_fraction' requires 'fraction' to be set")
        if self.sizing == "full" and self.fraction is not None:
            raise ValueError("'fraction' is only meaningful when sizing is 'fixed_fraction'")
        return self

    @property
    def min_bars(self) -> int:
        """Bars of history the primitive itself needs, before warmup_bars is applied."""
        if isinstance(self.params, SmaCrossParams):
            return self.params.slow
        if isinstance(self.params, ThresholdParams):
            return self.params.lookback + 1
        if isinstance(self.params, ForecastParams):
            return 2 * self.params.horizon + 2
        return 1


def parse_spec(payload: dict[str, Any]) -> StrategySpec:
    """Validates a spec, converting pydantic errors into actionable typed errors.

    The error text is part of the measured surface: E5 scores whether a model can
    self-correct from it, so it names the offending key and lists the valid options
    rather than echoing a pydantic traceback. Params are validated against the
    primitive's own model first, so a wrong key is reported against the four fields
    that primitive accepts instead of against a four-way union.
    """
    primitive = payload.get("primitive")
    if primitive is None:
        raise InvalidSpec(
            cause="strategy spec is missing 'primitive'",
            suggested_action=f"set primitive to one of: {', '.join(PRIMITIVES)}",
            offending_key="primitive",
            valid_options=list(PRIMITIVES),
        )
    if primitive not in PARAMS_FOR:
        raise InvalidSpec(
            cause=f"unknown primitive '{primitive}'",
            suggested_action=f"use one of: {', '.join(PRIMITIVES)}",
            offending_key="primitive",
            valid_options=list(PRIMITIVES),
        )

    params_model = PARAMS_FOR[primitive]
    raw_params = payload.get("params") or {}
    if isinstance(raw_params, dict):
        try:
            params_model.model_validate(raw_params)
        except ValidationError as exc:
            raise _spec_error(exc, primitive, scope="params") from None

    try:
        return StrategySpec.model_validate(payload)
    except ValidationError as exc:
        raise _spec_error(exc, primitive, scope="spec") from None


def _spec_error(exc: ValidationError, primitive: str, scope: str) -> InvalidSpec:
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first["loc"])
    key = f"params.{loc}" if scope == "params" and loc else (loc or "<root>")
    return InvalidSpec(
        cause=f"invalid strategy spec at '{key}': {first['msg']}",
        suggested_action=_suggest(first, primitive, scope),
        offending_key=key,
        valid_options=_valid_options(primitive, scope),
        error_count=len(exc.errors()),
    )


def _valid_options(primitive: str, scope: str) -> list[str]:
    if scope == "params":
        return list(PARAMS_FOR[primitive].model_fields)
    return list(StrategySpec.model_fields)


def _suggest(error: Any, primitive: str, scope: str) -> str:
    fields = ", ".join(_valid_options(primitive, scope)) or "no parameters"
    where = f"primitive '{primitive}'" if scope == "params" else "StrategySpec"
    if error["type"] == "extra_forbidden":
        return f"remove that key; {where} accepts exactly: {fields}"
    if error["type"] == "missing":
        return f"{where} requires: {fields}"
    if error["type"].startswith("value_error"):
        return f"{error['msg'].removeprefix('Value error, ')}; {where} accepts: {fields}"
    if error["type"] in {"greater_than", "less_than", "greater_than_equal", "less_than_equal"}:
        return f"{error['msg']}; call validate_strategy to dry-run a spec before running it"
    return f"correct '{'.'.join(str(p) for p in error['loc'])}'; {where} accepts: {fields}"


AnnotatedSpec = Annotated[StrategySpec, Field(description="A declarative strategy specification.")]
