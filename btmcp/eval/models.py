from __future__ import annotations

import json
from typing import Any, Protocol

from btmcp.eval.events import Event, ToolCall, ToolSpec, Turn
from btmcp.eval.ladder import Ladder, Rung, resolve
from btmcp.eval.rendering import anthropic_messages, anthropic_tools, openai_messages, openai_tools
from btmcp.eval.schema import OPENAI_STRICT, sanitize


class Model(Protocol):
    id: str

    async def respond(self, system: str, tools: list[ToolSpec], history: list[Event]) -> Turn: ...

    async def complete_json(
        self, system: str, prompt: str, schema: dict[str, Any], name: str
    ) -> dict[str, Any]: ...


def parse_arguments(value: Any) -> dict[str, Any]:
    """Tool inputs arrive with provider-specific JSON escaping; never string-match them."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value or "{}")
    return dict(value)


class AnthropicModel:
    def __init__(self, rung: Rung) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=rung.credential())
        self.rung = rung
        self.id = rung.name

    async def respond(self, system: str, tools: list[ToolSpec], history: list[Event]) -> Turn:
        response = await self._client.messages.create(
            model=self.rung.model,
            max_tokens=self.rung.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=anthropic_tools(tools, self.rung.schema_dialect),
            messages=anthropic_messages(history),
            output_config={"effort": self.rung.effort} if self.rung.effort else {},
            thinking={"type": "adaptive"},
        )

        text: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, parse_arguments(block.input)))

        return Turn(
            text="\n".join(text).strip(),
            tool_calls=calls,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            native=response.content,
        )

    async def complete_json(
        self, system: str, prompt: str, schema: dict[str, Any], name: str
    ) -> dict[str, Any]:
        response = await self._client.messages.create(
            model=self.rung.model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": sanitize(schema, self.rung.schema_dialect)}
            },
        )
        return json.loads(next(b.text for b in response.content if b.type == "text"))


class OpenAICompatModel:
    """Any provider speaking the OpenAI chat-completions API.

    Providers diverge on what schemas they accept, so the dialect comes from the rung
    rather than being fixed here: Cerebras strict mode rejects validation keywords, and
    several servers never resolve `$ref`.
    """

    def __init__(self, rung: Rung) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                f"rung '{rung.name}' needs the OpenAI-compatible client. "
                "Install it with: uv sync --extra openai"
            ) from exc

        self._client = openai.AsyncOpenAI(
            base_url=rung.base_url,
            api_key=rung.credential() or "not-required",
        )
        self.rung = rung
        self.id = rung.name

    async def respond(self, system: str, tools: list[ToolSpec], history: list[Event]) -> Turn:
        response = await self._client.chat.completions.create(
            model=self.rung.model,
            max_tokens=self.rung.max_tokens,
            messages=openai_messages(system, history),
            tools=openai_tools(tools, self.rung.schema_dialect) or None,
        )
        choice = response.choices[0]
        message = choice.message

        calls = [
            ToolCall(
                c.id,
                c.function.name,
                parse_arguments(c.function.arguments),
                extra=provider_extra(c),
            )
            for c in (message.tool_calls or [])
        ]
        usage = response.usage
        return Turn(
            text=(message.content or "").strip(),
            tool_calls=calls,
            stop_reason=choice.finish_reason or "end_turn",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            cache_read_tokens=_cached_tokens(usage),
        )

    async def complete_json(
        self, system: str, prompt: str, schema: dict[str, Any], name: str
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self.rung.model,
            max_tokens=1000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": sanitize(schema, OPENAI_STRICT),
                },
            },
        )
        return json.loads(response.choices[0].message.content or "{}")


PASSTHROUGH_TOOL_CALL_KEYS = ("extra_content",)
"""Keys on a returned tool call that must be echoed back on the next request.

Only `extra_content` today, which is how Gemini 3 carries its thought signature over
the OpenAI-compatible endpoint. Kept as a table because this is a class of provider
behaviour, not a one-off: the alternative is a Google branch inside the adapter.
"""


def provider_extra(call: Any) -> dict[str, Any] | None:
    """Lifts the passthrough keys off one tool call, or None if it carries none.

    These are undeclared fields on the OpenAI SDK's model, so they arrive either as an
    attribute or inside `model_extra` depending on SDK version - and not at all for
    every provider that is not Gemini. Read defensively and stay silent when absent,
    so no other provider's payload changes shape.
    """
    extra = getattr(call, "model_extra", None) or {}
    out: dict[str, Any] = {}
    for key in PASSTHROUGH_TOOL_CALL_KEYS:
        value = getattr(call, key, None)
        if value is None and isinstance(extra, dict):
            value = extra.get(key)
        if value is None and isinstance(call, dict):
            value = call.get(key)
        if value is None:
            continue
        out[key] = value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value
    return out or None


def _cached_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return int(getattr(details, "cached_tokens", 0) or 0) if details else 0


class NullModel:
    """Answers without calling anything, so the sweep path is testable at zero cost."""

    id = "null"

    def __init__(self, rung: Rung | None = None) -> None:
        self.rung = rung

    async def respond(self, system: str, tools: list[ToolSpec], history: list[Event]) -> Turn:
        return Turn(text="No tools were called.", input_tokens=len(system) // 4, output_tokens=5)

    async def complete_json(
        self, system: str, prompt: str, schema: dict[str, Any], name: str
    ) -> dict[str, Any]:
        return {}


class ScriptedModel:
    """Replays fixed turns and records the rendered history it was handed."""

    def __init__(self, turns: list[Turn], model_id: str = "scripted") -> None:
        self.id = model_id
        self._turns = list(turns)
        self.seen: list[list[Event]] = []

    async def respond(self, system: str, tools: list[ToolSpec], history: list[Event]) -> Turn:
        self.seen.append(list(history))
        return self._turns.pop(0) if self._turns else Turn(text="(script exhausted)")

    async def complete_json(
        self, system: str, prompt: str, schema: dict[str, Any], name: str
    ) -> dict[str, Any]:
        return {}


BUILDERS = {
    "anthropic": AnthropicModel,
    "openai_compat": OpenAICompatModel,
    "null": NullModel,
}


def build_model(spec: str, ladder: Ladder | None = None) -> Model:
    rung = resolve(spec, ladder)
    builder = BUILDERS.get(rung.provider)
    if builder is None:
        raise KeyError(f"unknown provider '{rung.provider}'; known: {', '.join(sorted(BUILDERS))}")
    return builder(rung)
