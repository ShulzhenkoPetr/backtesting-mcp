from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import ElicitResult

from btmcp.eval.assertions import CallRecord, Evidence, evaluate_all
from btmcp.eval.events import Event, ToolOutcome, ToolResults, ToolSpec, Turn, UserPrompt
from btmcp.eval.models import Model
from btmcp.eval.scenario import Scenario
from btmcp.server.app import AppConfig
from btmcp.server.guardrails.fencing import FencingConfig

SYSTEM_PROMPT = (
    "You are a quantitative research assistant with access to a point-in-time "
    "backtesting server. Use its tools to answer the user's request. "
    "Every data tool requires an `as_of` cutoff and cannot return a bar dated after it. "
    "State transaction costs explicitly; there are no defaults. "
    "If the request is ambiguous in a way that changes the result, ask the user rather "
    "than guessing. When you have the answer, state it plainly in your final message."
)


@dataclass
class TurnLog:
    index: int
    text: str
    tool_calls: list[dict[str, Any]]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


@dataclass
class RunOutcome:
    scenario_id: str
    category: str
    split: str
    surface: str
    model: str
    seed: int
    fencing: bool
    task_success: bool
    assertions: list[dict[str, Any]]
    calls_to_completion: float
    total_tokens: int
    cache_read_tokens: int
    wrong_tool_calls: int
    total_calls: int
    hit_max_turns: bool
    error_codes: list[str]
    final_answer: str
    elicitations: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    rubric_score: int | None = None
    elapsed_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def tool_specs(tools: Any) -> list[ToolSpec]:
    return [ToolSpec(t.name, t.description or "", t.input_schema) for t in tools]


async def run_scenario(
    scenario: Scenario,
    surface_build: Any,
    model: Model,
    seed: int = 0,
    fencing: bool = True,
    runs_root: Path | None = None,
    surface_name: str = "a",
) -> RunOutcome:
    """Drives one agent loop against one surface and grades it.

    Assertions are evaluated against the server's own audit log joined to the
    arguments the model actually sent, not against the transcript: the log is written
    server-side and cannot be talked around by a model that merely claims to have
    called something.
    """
    config = AppConfig(
        surface=surface_name,
        tier=scenario.tier,
        runs_root=runs_root or Path("runs"),
        fencing=FencingConfig(enabled=fencing),
    )
    mcp, services = surface_build(config)

    elicitations: list[str] = []

    async def on_elicit(context: Any, params: Any) -> ElicitResult:
        """Records what the server asked for and declines.

        Declining keeps runs comparable: an auto-accepting harness would supply the
        very parameters the ambiguity scenarios are testing whether the model invents.
        """
        elicitations.append(getattr(params, "message", ""))
        return ElicitResult(action="decline")

    started = time.perf_counter()
    history: list[Event] = [UserPrompt(scenario.prompt)]
    turns: list[TurnLog] = []
    call_args: list[tuple[str, dict[str, Any], Any]] = []
    final_answer = ""
    hit_max_turns = True

    # Legacy mode deliberately: elicitation needs a back-channel for server-initiated
    # requests, which the stateless 2026-07-28 envelope does not provide. Surface B's
    # design turns on eliciting, so measuring it on a transport that cannot elicit
    # would score the protocol rather than the surface.
    async with Client(mcp, mode="legacy", elicitation_callback=on_elicit) as client:
        specs = tool_specs((await client.list_tools()).tools)

        for index in range(scenario.max_turns):
            turn: Turn = await model.respond(SYSTEM_PROMPT, specs, history)
            turns.append(
                TurnLog(
                    index=index,
                    text=turn.text,
                    tool_calls=[{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                    stop_reason=turn.stop_reason,
                    input_tokens=turn.input_tokens,
                    output_tokens=turn.output_tokens,
                    cache_read_tokens=turn.cache_read_tokens,
                )
            )

            if not turn.tool_calls:
                final_answer = turn.text
                hit_max_turns = False
                break

            history.append(turn)
            outcomes: list[ToolOutcome] = []
            for call in turn.tool_calls:
                payload, is_error = await _invoke(client, call.name, call.arguments)
                call_args.append((call.name, call.arguments, payload))
                outcomes.append(ToolOutcome(call.id, call.name, payload, is_error))
            history.append(ToolResults(outcomes))

    audit = list(services.audit.records)
    calls = _join(audit, call_args)
    evidence = Evidence(
        calls=calls,
        final_answer=final_answer,
        turns_used=len(turns),
        hit_max_turns=hit_max_turns,
        surface=surface_name,
        elicitations=elicitations,
    )
    outcomes = evaluate_all(scenario, evidence)
    allowed = set(scenario.allowed_tools) if scenario.allowed_tools else None
    wrong = sum(1 for c in calls if allowed is not None and c.tool not in allowed)

    return RunOutcome(
        scenario_id=scenario.id,
        category=scenario.category,
        split=scenario.split,
        surface=surface_name,
        model=model.id,
        seed=seed,
        fencing=fencing,
        task_success=all(o.passed for o in outcomes),
        assertions=[asdict(o) for o in outcomes],
        calls_to_completion=float("inf") if hit_max_turns else len(calls),
        total_tokens=sum(t.input_tokens + t.output_tokens for t in turns),
        cache_read_tokens=sum(t.cache_read_tokens for t in turns),
        wrong_tool_calls=wrong,
        total_calls=len(calls),
        hit_max_turns=hit_max_turns,
        error_codes=[c.error_code for c in calls if c.error_code],
        final_answer=final_answer,
        elicitations=elicitations,
        turns=[asdict(t) for t in turns],
        audit=audit,
        elapsed_s=time.perf_counter() - started,
    )


async def _invoke(client: Client, name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
    try:
        result = await client.call_tool(name, arguments)
    except MCPError as exc:
        return exc.error.data or {"code": "MCP_ERROR", "cause": exc.error.message}, True
    if result.is_error:
        return {"code": "INVALID_ARGUMENTS", "cause": str(result.content)}, True
    return result.structured_content, False


def _join(audit: list[dict[str, Any]], call_args: list[tuple[str, dict[str, Any], Any]]) -> list[CallRecord]:
    """Pairs audit rows with the arguments and payloads the runner observed.

    The audit log stores only an argument hash, so the runner supplies the arguments;
    the two are aligned positionally because the server processes one call at a time.
    """
    records: list[CallRecord] = []
    for index, row in enumerate(audit):
        name, arguments, payload = call_args[index] if index < len(call_args) else (row["tool"], {}, None)
        records.append(
            CallRecord(
                index=index,
                tool=row["tool"],
                arguments=arguments if name == row["tool"] else {},
                outcome=row["outcome"],
                error_code=row.get("error_code"),
                result_tokens=row.get("result_tokens", 0),
                result=payload,
            )
        )
    return records


def write_outcome(outcome: RunOutcome, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{outcome.scenario_id}__{outcome.surface}__seed{outcome.seed}.json"
    path.write_text(outcome.to_json())
    return path


def outcome_exists(scenario_id: str, surface: str, seed: int, out_dir: Path) -> bool:
    return (out_dir / f"{scenario_id}__{surface}__seed{seed}.json").exists()


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()
