from __future__ import annotations

import json
from typing import Any

from btmcp.eval.events import Event, ToolSpec, Turn, UserPrompt
from btmcp.eval.schema import SchemaDialect, sanitize

MAX_RESULT_CHARS = 20_000


def _payload(outcome_payload: Any) -> str:
    return json.dumps(outcome_payload, default=str)[:MAX_RESULT_CHARS]


def anthropic_tools(tools: list[ToolSpec], dialect: SchemaDialect) -> list[dict[str, Any]]:
    rendered = [
        {"name": t.name, "description": t.description, "input_schema": sanitize(t.input_schema, dialect)}
        for t in tools
    ]
    if rendered:
        # Tool schemas are byte-identical on every turn of every run, so the breakpoint
        # sits after the last one and the whole block is served from cache thereafter.
        rendered[-1]["cache_control"] = {"type": "ephemeral"}
    return rendered


def anthropic_messages(history: list[Event]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for event in history:
        if isinstance(event, UserPrompt):
            messages.append({"role": "user", "content": event.text})
        elif isinstance(event, Turn):
            messages.append({"role": "assistant", "content": event.native or _assistant_blocks(event)})
        else:
            # One user message carrying every result: splitting them across messages
            # teaches the model to stop issuing parallel calls.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.call_id,
                            "content": _payload(r.payload),
                            "is_error": r.is_error,
                        }
                        for r in event.results
                    ],
                }
            )
    return messages


def _assistant_blocks(turn: Turn) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if turn.text:
        blocks.append({"type": "text", "text": turn.text})
    blocks += [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in turn.tool_calls
    ]
    return blocks or [{"type": "text", "text": "(no content)"}]


def openai_tools(tools: list[ToolSpec], dialect: SchemaDialect) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": sanitize(t.input_schema, dialect),
                **({"strict": True} if dialect.require_all_properties else {}),
            },
        }
        for t in tools
    ]


def openai_messages(system: str, history: list[Event]) -> list[dict[str, Any]]:
    """Renders the neutral history into chat-completions form.

    The shapes genuinely differ: the system prompt is a message rather than a
    top-level field, tool arguments are a JSON string rather than an object, and each
    tool result is its own `role: "tool"` message rather than blocks inside one user
    turn.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for event in history:
        if isinstance(event, UserPrompt):
            messages.append({"role": "user", "content": event.text})
        elif isinstance(event, Turn):
            message: dict[str, Any] = {"role": "assistant", "content": event.text or None}
            if event.tool_calls:
                # `c.extra` carries provider keys that came back on the call and have
                # to go out again unchanged - Gemini 3's thought signature is one, and
                # omitting it is a 400 on the next request, not a silent degradation.
                message["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments, default=str)},
                        **(c.extra or {}),
                    }
                    for c in event.tool_calls
                ]
            messages.append(message)
        else:
            messages += [
                {"role": "tool", "tool_call_id": r.call_id, "name": r.name, "content": _payload(r.payload)}
                for r in event.results
            ]
    return messages
