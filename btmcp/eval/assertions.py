from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from btmcp.eval.scenario import Assertion, Scenario
from btmcp.eval.surfaces import resolve

CLARIFICATION_MARKERS = ("?",)
CLARIFY_PHRASES = (
    "could you",
    "can you clarify",
    "which ",
    "do you want",
    "should i",
    "would you like",
    "please confirm",
    "let me know",
    "what period",
    "i need to know",
    "before i run",
    "to proceed",
)


@dataclass
class CallRecord:
    """One tool call as the audit log saw it, joined to the arguments the model sent."""

    index: int
    tool: str
    arguments: dict[str, Any]
    outcome: str
    error_code: str | None
    result_tokens: int
    result: Any = None


@dataclass
class Evidence:
    calls: list[CallRecord]
    final_answer: str
    turns_used: int
    hit_max_turns: bool
    surface: str = "a"
    elicitations: list[str] = field(default_factory=list)

    def by_tool(self, capability: str) -> list[CallRecord]:
        binding = resolve(self.surface, capability)
        return [c for c in self.calls if binding.matches(c.tool, c.arguments)]


@dataclass
class AssertionOutcome:
    type: str
    passed: bool
    detail: str


def _dig(payload: Any, path: str) -> Any:
    """Resolves a dotted path, treating integer segments as list indices."""
    current = payload
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                return None
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _numeric_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        return abs(float(actual) - float(expected)) < 1e-9
    return actual == expected


def evaluate(assertion: Assertion, evidence: Evidence, scenario: Scenario) -> AssertionOutcome:
    kind = assertion.type

    if kind == "tool_called":
        hits = evidence.by_tool(assertion.tool)
        ok = len(hits) >= assertion.min_times
        return AssertionOutcome(kind, ok, f"{assertion.tool} called {len(hits)}x, need {assertion.min_times}")

    if kind == "no_tool_called":
        hits = evidence.by_tool(assertion.tool)
        return AssertionOutcome(kind, not hits, f"{assertion.tool} called {len(hits)}x, need 0")

    if kind == "arg_matches":
        hits = evidence.by_tool(assertion.tool)
        if not hits:
            return AssertionOutcome(kind, False, f"{assertion.tool} was never called")
        seen = [_dig(c.arguments, assertion.path) for c in hits]
        ok = any(_numeric_equal(v, assertion.equals) for v in seen)
        return AssertionOutcome(kind, ok, f"{assertion.path}={seen!r}, expected {assertion.equals!r}")

    if kind == "no_successful_call":
        hits = [c for c in evidence.by_tool(assertion.tool) if c.outcome == "ok"]
        return AssertionOutcome(kind, not hits, f"{assertion.tool} completed {len(hits)}x, need 0")

    if kind == "arg_never_matches":
        hits = evidence.by_tool(assertion.tool)
        seen = [_dig(c.arguments, assertion.path) for c in hits]
        ok = not any(_numeric_equal(v, assertion.equals) for v in seen)
        return AssertionOutcome(kind, ok, f"{assertion.path}={seen!r}, must never be {assertion.equals!r}")

    if kind == "config_hash_reached":
        expected = assertion.value
        reached = [
            c.result.get("run_id") for c in evidence.by_tool("run_backtest") if isinstance(c.result, dict)
        ]
        reached += [c.arguments.get("run_id") for c in evidence.by_tool("get_run")]
        reached = [r for r in reached if r]
        if expected is None:
            return AssertionOutcome(kind, bool(reached), f"runs reached: {[r[:12] for r in reached]}")
        ok = expected in reached
        return AssertionOutcome(kind, ok, f"expected {expected[:12]}, reached {[r[:12] for r in reached]}")

    if kind == "error_code_seen":
        codes = [c.error_code for c in evidence.calls if c.error_code]
        return AssertionOutcome(kind, assertion.code in codes, f"error codes seen: {codes}")

    if kind == "recovered_after_error":
        first_error = next((c.index for c in evidence.calls if c.outcome != "ok"), None)
        if first_error is None:
            return AssertionOutcome(kind, False, "no error occurred, so no recovery to measure")
        window = [
            c
            for c in evidence.calls
            if first_error < c.index <= first_error + assertion.within
            and c.tool == assertion.tool
            and c.outcome == "ok"
        ]
        return AssertionOutcome(
            kind, bool(window), f"{assertion.tool} succeeded within {assertion.within} calls: {bool(window)}"
        )

    if kind == "max_calls":
        ok = len(evidence.calls) <= assertion.value
        return AssertionOutcome(kind, ok, f"{len(evidence.calls)} calls, limit {assertion.value}")

    if kind == "result_tokens_under":
        worst = max((c.result_tokens for c in evidence.calls), default=0)
        return AssertionOutcome(
            kind, worst <= assertion.value, f"largest result {worst} tokens, limit {assertion.value}"
        )

    if kind == "final_answer_matches":
        ok = re.search(assertion.pattern, evidence.final_answer, re.IGNORECASE | re.DOTALL) is not None
        return AssertionOutcome(
            kind, ok, f"pattern {assertion.pattern!r} against {len(evidence.final_answer)} chars"
        )

    if kind == "asked_for_clarification":
        # Either route counts: asking the user in prose, or triggering the server's
        # own elicitation. Surface B elicits by design, so requiring prose would
        # score a deliberate design difference as a failure.
        text = evidence.final_answer.lower()
        asked = "?" in text and any(phrase in text for phrase in CLARIFY_PHRASES)
        elicited = bool(evidence.elicitations)
        return AssertionOutcome(
            kind, asked or elicited, f"asked in prose: {asked}, server elicited: {elicited}"
        )

    raise ValueError(f"unhandled assertion type: {kind}")


def evaluate_all(scenario: Scenario, evidence: Evidence) -> list[AssertionOutcome]:
    return [evaluate(a, evidence, scenario) for a in scenario.assertions]
