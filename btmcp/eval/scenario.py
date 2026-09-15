from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "happy_path",
    "multi_step",
    "ambiguity",
    "error_recovery",
    "refusal",
    "injection",
    "token_stress",
]
Split = Literal["dev", "holdout"]


class ToolCalled(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_called"] = "tool_called"
    tool: str
    min_times: int = 1


class NoToolCalled(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["no_tool_called"] = "no_tool_called"
    tool: str


class NoSuccessfulCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["no_successful_call"] = "no_successful_call"
    tool: str


class ArgMatches(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["arg_matches"] = "arg_matches"
    tool: str
    path: str
    equals: Any


class ArgNeverMatches(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["arg_never_matches"] = "arg_never_matches"
    tool: str
    path: str
    equals: Any


class ConfigHashReached(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["config_hash_reached"] = "config_hash_reached"
    value: str | None = None
    spec: dict[str, Any] | None = None


class ErrorCodeSeen(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error_code_seen"] = "error_code_seen"
    code: str


class RecoveredAfterError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["recovered_after_error"] = "recovered_after_error"
    tool: str
    within: int = 3


class MaxCalls(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["max_calls"] = "max_calls"
    value: int


class ResultTokensUnder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["result_tokens_under"] = "result_tokens_under"
    value: int


class FinalAnswerMatches(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["final_answer_matches"] = "final_answer_matches"
    pattern: str


class AskedForClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["asked_for_clarification"] = "asked_for_clarification"


Assertion = Annotated[
    ToolCalled
    | NoToolCalled
    | NoSuccessfulCall
    | ArgMatches
    | ArgNeverMatches
    | ConfigHashReached
    | ErrorCodeSeen
    | RecoveredAfterError
    | MaxCalls
    | ResultTokensUnder
    | FinalAnswerMatches
    | AskedForClarification,
    Field(discriminator="type"),
]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    split: Split
    prompt: str
    tier: Literal["read", "simulate", "act"] = "simulate"
    max_turns: int = Field(default=10, ge=1, le=30)
    assertions: list[Assertion] = Field(min_length=1)
    rubric: str | None = None
    allowed_tools: list[str] | None = None
    payload_id: str | None = None
    notes: str = ""


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(path.read_text()))


def load_scenarios(
    root: Path = Path("btmcp/eval/scenarios"),
    split: Split | None = None,
    ids: list[str] | None = None,
) -> list[Scenario]:
    scenarios = [load_scenario(p) for p in sorted(root.glob("*.yaml"))]
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            raise ValueError(f"duplicate scenario id: {scenario.id}")
        seen.add(scenario.id)
    if split:
        scenarios = [s for s in scenarios if s.split == split]
    if ids:
        wanted = set(ids)
        scenarios = [s for s in scenarios if s.id in wanted]
    return scenarios
