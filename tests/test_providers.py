from __future__ import annotations

import json
from pathlib import Path

import pytest

from btmcp.eval.events import ToolCall, ToolOutcome, ToolResults, ToolSpec, Turn, UserPrompt
from btmcp.eval.ladder import Ladder, Quota, Rung, load_ladder, resolve
from btmcp.eval.models import BUILDERS, build_model, parse_arguments
from btmcp.eval.rendering import anthropic_messages, anthropic_tools, openai_messages, openai_tools
from btmcp.eval.schema import ANTHROPIC, DIALECTS, OPENAI, OPENAI_STRICT, sanitize

NESTED = {
    "type": "object",
    "$defs": {
        "CostModel": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fee_bps": {"type": "number", "minimum": 0, "maximum": 1000, "title": "Fee Bps"},
                "note": {"type": "string", "pattern": "^[a-z]+$", "minLength": 1, "maxLength": 40},
            },
            "required": ["fee_bps"],
        }
    },
    "properties": {
        "costs": {"$ref": "#/$defs/CostModel"},
        "start": {"type": "string", "format": "date"},
        "fields": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
    },
    "required": ["costs", "start"],
}

RECURSIVE = {
    "type": "object",
    "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    "properties": {"root": {"$ref": "#/$defs/Node"}},
}


# ---- schema sanitizer ----------------------------------------------------


def test_anthropic_dialect_leaves_schemas_alone() -> None:
    assert sanitize(NESTED, ANTHROPIC) == NESTED


def test_refs_are_inlined_and_defs_dropped() -> None:
    out = sanitize(NESTED, OPENAI)
    blob = json.dumps(out)
    assert "$ref" not in blob and "$defs" not in blob
    assert out["properties"]["costs"]["properties"]["fee_bps"]["type"] == "number"


def test_strict_mode_strips_keywords_cerebras_rejects() -> None:
    """Cerebras strict mode rejects pattern/minLength/maxLength; Pydantic emits all three."""
    blob = json.dumps(sanitize(NESTED, OPENAI_STRICT))
    for keyword in ("pattern", "minLength", "maxLength", "format", "minimum", "maximum", "title"):
        assert keyword not in blob, f"{keyword} survived strict sanitization"


def test_strict_mode_keeps_the_schema_usable() -> None:
    out = sanitize(NESTED, OPENAI_STRICT)
    costs = out["properties"]["costs"]
    assert costs["type"] == "object"
    assert set(costs["properties"]) == {"fee_bps", "note"}
    assert costs["properties"]["note"]["type"] == ["string", "null"], "optional stays expressible"


def test_strict_mode_requires_every_property_and_nullables_the_optional_ones() -> None:
    out = sanitize(NESTED, OPENAI_STRICT)
    assert set(out["required"]) == {"costs", "start", "fields"}
    assert out["additionalProperties"] is False
    # `fields` was optional, so it stays expressible as null.
    assert {"type": "null"} in out["properties"]["fields"]["anyOf"]
    # `note` was optional inside the nested object too.
    assert out["properties"]["costs"]["properties"]["note"]["type"] == ["string", "null"]


def test_recursive_refs_do_not_expand_forever() -> None:
    out = sanitize(RECURSIVE, OPENAI)
    assert out["properties"]["root"]["properties"]["child"] == {"type": "object"}


def test_a_dangling_ref_degrades_rather_than_raising() -> None:
    out = sanitize({"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}}, OPENAI)
    assert out["properties"]["x"] == {"type": "object"}


def test_sanitize_does_not_mutate_its_input() -> None:
    before = json.dumps(NESTED, sort_keys=True)
    sanitize(NESTED, OPENAI_STRICT)
    assert json.dumps(NESTED, sort_keys=True) == before


def test_real_tool_schemas_survive_every_dialect() -> None:
    import asyncio

    from mcp import Client

    from btmcp.server.app import AppConfig
    from btmcp.server.surface_a import build

    async def collect():
        mcp, _ = build(AppConfig())
        async with Client(mcp) as c:
            return [(t.name, t.input_schema) for t in (await c.list_tools()).tools]

    for name, schema in asyncio.run(collect()):
        for dialect in DIALECTS.values():
            out = sanitize(schema, dialect)
            assert out.get("type") == "object", f"{name} lost its shape under {dialect.name}"
            if dialect.inline_refs:
                assert "$ref" not in json.dumps(out), f"{name} kept a $ref under {dialect.name}"


# ---- rendering -----------------------------------------------------------


TOOLS = [ToolSpec("get_bars", "Read bars.", NESTED)]
HISTORY = [
    UserPrompt("Backtest SYN-04."),
    Turn(
        text="Calling now.",
        tool_calls=[ToolCall("c1", "get_bars", {"symbol": "SYN-04"})],
        stop_reason="tool_use",
    ),
    ToolResults([ToolOutcome("c1", "get_bars", {"n_rows": 5}, False)]),
]


def test_anthropic_puts_every_tool_result_in_one_user_message() -> None:
    history = [
        *HISTORY[:2],
        ToolResults(
            [ToolOutcome("c1", "get_bars", {"n": 1}, False), ToolOutcome("c2", "get_news", {"n": 2}, False)]
        ),
    ]
    messages = anthropic_messages(history)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert len(messages[-1]["content"]) == 2
    assert all(b["type"] == "tool_result" for b in messages[-1]["content"])


def test_openai_puts_each_tool_result_in_its_own_message() -> None:
    history = [
        *HISTORY[:2],
        ToolResults(
            [ToolOutcome("c1", "get_bars", {"n": 1}, False), ToolOutcome("c2", "get_news", {"n": 2}, False)]
        ),
    ]
    messages = openai_messages("SYS", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
    assert messages[-1]["tool_call_id"] == "c2"


def test_openai_encodes_tool_arguments_as_a_json_string() -> None:
    messages = openai_messages("SYS", HISTORY)
    assistant = next(m for m in messages if m["role"] == "assistant")
    call = assistant["tool_calls"][0]
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"symbol": "SYN-04"}


def test_anthropic_echoes_native_content_when_present() -> None:
    """Thinking blocks must be returned unchanged, so a native turn is not re-synthesised."""
    native = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "hi"}]
    messages = anthropic_messages([UserPrompt("q"), Turn(text="hi", native=native)])
    assert messages[-1]["content"] is native


def test_anthropic_synthesises_content_when_a_turn_has_none() -> None:
    messages = anthropic_messages([UserPrompt("q"), Turn(text="hi")])
    assert messages[-1]["content"] == [{"type": "text", "text": "hi"}]


def test_the_system_prompt_is_a_message_only_for_openai() -> None:
    assert openai_messages("SYS", HISTORY)[0] == {"role": "system", "content": "SYS"}
    assert all(m["role"] != "system" for m in anthropic_messages(HISTORY))


def test_anthropic_marks_a_cache_breakpoint_on_the_last_tool() -> None:
    rendered = anthropic_tools(TOOLS, ANTHROPIC)
    assert rendered[-1]["cache_control"] == {"type": "ephemeral"}


def test_openai_tools_declare_strict_only_for_strict_dialects() -> None:
    assert "strict" not in openai_tools(TOOLS, OPENAI)[0]["function"]
    assert openai_tools(TOOLS, OPENAI_STRICT)[0]["function"]["strict"] is True


def test_openai_tool_schemas_are_sanitized() -> None:
    params = openai_tools(TOOLS, OPENAI_STRICT)[0]["function"]["parameters"]
    assert "$ref" not in json.dumps(params) and "pattern" not in json.dumps(params)


def test_renderers_need_no_provider_sdk() -> None:
    """Rendering is pure, so both wire formats are testable without either SDK installed."""
    import ast

    for module in ("btmcp/eval/rendering.py", "btmcp/eval/schema.py", "btmcp/eval/events.py"):
        tree = ast.parse(Path(module).read_text())
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not names & {"anthropic", "openai"}, f"{module} imports a provider SDK"


def test_tool_arguments_are_parsed_not_string_matched() -> None:
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments("") == {}


# ---- ladder --------------------------------------------------------------


def test_the_shipped_ladder_is_ordered_and_open() -> None:
    ladder = load_ladder()
    ranks = [r.rank for r in ladder]
    assert ranks == sorted(ranks)
    assert {r.provider for r in ladder} >= {"anthropic", "openai_compat", "null"}
    assert all(r.provider in BUILDERS for r in ladder)


def test_every_rung_dialect_exists() -> None:
    for rung in load_ladder():
        assert rung.dialect in DIALECTS


def test_free_and_paid_rungs_are_distinguishable() -> None:
    """Cerebras is deliberately not the free example any more.

    Its free tier became $5 of expiring trial credit, so the rung carries prices and
    reports against the euro ceiling. A rung that is genuinely free is the one that
    should stand for `free` here.
    """
    ladder = load_ladder()
    assert ladder.get("cheap").free is False
    assert ladder.get("cerebras-oss-120b").free is False
    assert ladder.get("groq-llama-70b").free is True


def test_cerebras_rungs_use_the_cerebras_dialect() -> None:
    """Cerebras validates tool schemas whether or not strict mode is asked for.

    It needs the keyword stripping and the empty-branch drop, but not strict mode's
    required-everything rewrite - that would make `params: {}` unrepresentable, and
    `buy_and_hold` has no other spelling.
    """
    ladder = load_ladder()
    assert ladder.get("cerebras-oss-120b").dialect == "cerebras"
    assert ladder.get("cerebras-qwen-27b").dialect == "cerebras"


def test_empty_object_union_branches_are_dropped_only_where_needed() -> None:
    """`EmptyParams` serialises to a property-less object, which Cerebras refuses."""
    from btmcp.eval.schema import CEREBRAS, OPENAI_STRICT, is_empty_object, sanitize

    schema = {
        "type": "object",
        "properties": {
            "params": {
                "anyOf": [
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    {"type": "object", "properties": {"fast": {"type": "integer"}}},
                ]
            }
        },
    }
    kept = sanitize(schema, OPENAI_STRICT)["properties"]["params"]["anyOf"]
    dropped = sanitize(schema, CEREBRAS)["properties"]["params"]["anyOf"]
    assert any(is_empty_object(b) for b in kept), "other dialects keep the branch"
    assert not any(is_empty_object(b) for b in dropped)
    assert any(b.get("properties") for b in dropped), "the real branch has to survive"


def test_an_open_object_parameter_loses_its_type_but_keeps_its_description() -> None:
    """`run_backtest.spec` is `dict[str, Any]` on purpose; Cerebras refuses that shape.

    Untyping keeps the description - which is where the real instruction lives, since
    it points at the strategy-primitives resource - without inlining StrategySpec and
    changing the tool surface X0 measured.
    """
    from btmcp.eval.schema import CEREBRAS, OPENAI, sanitize

    schema = {
        "type": "object",
        "properties": {
            "spec": {"type": "object", "additionalProperties": True, "description": "A strategy spec."}
        },
        "required": ["spec"],
    }
    spec = sanitize(schema, CEREBRAS)["properties"]["spec"]
    assert "type" not in spec and "additionalProperties" not in spec
    assert spec["description"] == "A strategy spec."
    assert sanitize(schema, CEREBRAS)["required"] == ["spec"], "still a required parameter"
    assert sanitize(schema, OPENAI)["properties"]["spec"]["type"] == "object", "only cerebras untypes"


def test_the_root_parameter_object_is_never_untyped() -> None:
    """A no-argument tool still has to present `parameters` as an object."""
    from btmcp.eval.schema import CEREBRAS, sanitize

    assert sanitize({"type": "object", "properties": {}}, CEREBRAS) == {
        "type": "object",
        "properties": {},
    }


def test_a_union_of_only_empty_objects_is_left_alone() -> None:
    """Dropping every branch would silently delete the parameter; visibly wrong is better."""
    from btmcp.eval.schema import CEREBRAS, sanitize

    schema = {"anyOf": [{"type": "object", "properties": {}}, {"type": "object"}]}
    assert len(sanitize(schema, CEREBRAS)["anyOf"]) == 2


def test_a_provider_model_spec_resolves_without_a_rung() -> None:
    rung = resolve("openai_compat:qwen2.5-32b")
    assert rung.provider == "openai_compat" and rung.model == "qwen2.5-32b"


def test_unknown_rung_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        resolve("nope")


def test_duplicate_rung_names_are_rejected() -> None:
    rung = Rung(name="x", provider="null", model="m", rank=1)
    with pytest.raises(ValueError, match="duplicate"):
        Ladder([rung, rung])


def test_a_yaml_null_name_is_caught() -> None:
    """`name: null` unquoted in YAML parses as None, which would silently break lookups."""
    with pytest.raises(ValueError, match="non-empty string"):
        Rung(name=None, provider="null", model="m", rank=0)  # type: ignore[arg-type]


def test_an_unknown_dialect_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown dialect"):
        Rung(name="x", provider="null", model="m", rank=0, dialect="nope")


def test_availability_reflects_configured_credentials(monkeypatch) -> None:
    rung = Rung(name="x", provider="openai_compat", model="m", rank=1, api_key_env="BTMCP_TEST_KEY")
    monkeypatch.delenv("BTMCP_TEST_KEY", raising=False)
    assert not rung.available()
    monkeypatch.setenv("BTMCP_TEST_KEY", "secret")
    assert rung.available() and rung.credential() == "secret"


def test_a_local_rung_needs_no_credential() -> None:
    local = load_ladder().get("local-8b")
    assert local.api_key_env is None and local.available()
    assert local.base_url and "localhost" in local.base_url


def test_quota_carries_what_a_free_provider_limits() -> None:
    quota = load_ladder().get("groq-llama-70b").quota
    assert isinstance(quota, Quota)
    assert quota.daily_tokens and quota.per_minute_requests


def test_build_model_returns_the_right_adapter() -> None:
    from btmcp.eval.models import NullModel

    assert isinstance(build_model("null"), NullModel)


def test_a_missing_provider_sdk_names_the_extra_to_install() -> None:
    """The adapter is optional; failing to have it should say how to fix that."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no module named openai")
        return real_import(name, *args, **kwargs)

    models = importlib.import_module("btmcp.eval.models")
    builtins.__import__ = blocked
    try:
        with pytest.raises(ImportError, match="uv sync --extra openai"):
            models.OpenAICompatModel(load_ladder().get("local-8b"))
    finally:
        builtins.__import__ = real_import
