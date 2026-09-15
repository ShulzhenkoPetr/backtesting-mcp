from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None
    """Opaque provider keys captured from this call and merged back onto it when the
    transcript is re-rendered. Gemini 3 returns a thought signature on the first
    function call of each step and rejects the following request with a 400 unless it
    comes back verbatim, so the value has to survive the round trip through the
    neutral history. Providers that send nothing leave this None and their rendered
    payload is unchanged."""


@dataclass
class ToolOutcome:
    call_id: str
    name: str
    payload: Any
    is_error: bool


@dataclass
class UserPrompt:
    text: str


@dataclass
class ToolResults:
    results: list[ToolOutcome]


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    native: Any = None
    """Provider-native content for this turn, echoed back verbatim when the provider
    needs it (Anthropic requires thinking blocks returned unchanged). Adapters that
    did not produce it fall back to rendering the neutral fields."""


Event = UserPrompt | Turn | ToolResults
