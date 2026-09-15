from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

STRUCTURAL = ("properties", "items", "prefixItems", "additionalProperties")
COMBINATORS = ("anyOf", "oneOf", "allOf")


@dataclass(frozen=True)
class SchemaDialect:
    """What a provider will accept in a tool or response schema.

    Providers diverge more than the JSON Schema spec suggests: several OpenAI-compatible
    servers never resolve `$ref`, and strict modes (Cerebras among them) reject
    validation keywords they cannot enforce. Describing the target rather than
    hard-coding one output keeps the differences in one table.
    """

    name: str
    inline_refs: bool = False
    strip: frozenset[str] = field(default_factory=frozenset)
    force_additional_properties_false: bool = False
    require_all_properties: bool = False
    drop_empty_object_branches: bool = False
    """Removes property-less object members from an `anyOf`/`oneOf`.

    A pydantic model with no fields - `EmptyParams`, the `buy_and_hold` branch of
    `StrategySpec.params` - serialises to `{"type": "object", "properties": {}}`, and
    Cerebras rejects the whole request with "Object fields require at least one of:
    'properties' or 'anyOf'". The branch is dropped only when a sibling survives, so
    the union never collapses to nothing."""

    untype_open_objects: bool = False
    """Drops `type: object` from a deliberately open object, keeping its description.

    `run_backtest`'s `spec` is `dict[str, Any]` on purpose: the tool advertises a
    short opaque schema and points at the strategy-primitives resource for the real
    shape, which is what keeps its result inside the X0 token budget. Serialised that
    is `{"type": "object", "additionalProperties": true}` with no properties, which
    Cerebras refuses.

    Removing the type leaves an unconstrained value carrying the same description, so
    the model is told exactly what it was told before minus one word. Inlining the
    full StrategySpec instead would be a real change to the tool surface - bigger
    schemas, different token counts, and precisely the variable X3 exists to test."""


ANTHROPIC = SchemaDialect(name="anthropic")

OPENAI = SchemaDialect(
    name="openai",
    inline_refs=True,
    strip=frozenset({"title", "$comment"}),
)

OPENAI_STRICT = SchemaDialect(
    name="openai_strict",
    inline_refs=True,
    strip=frozenset(
        {
            "title",
            "$comment",
            "pattern",
            "minLength",
            "maxLength",
            "format",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minItems",
            "maxItems",
            "default",
            "examples",
        }
    ),
    force_additional_properties_false=True,
    require_all_properties=True,
)

CEREBRAS = SchemaDialect(
    name="cerebras",
    inline_refs=True,
    strip=OPENAI_STRICT.strip,
    drop_empty_object_branches=True,
    untype_open_objects=True,
)
"""Cerebras without `strict: true`.

Cerebras validates tool schemas whether or not strict mode is requested, and it
refuses a property-less object outright - so the keyword stripping and `$ref`
inlining are still needed, and empty union branches still have to go. What is
*not* wanted here is strict mode's rewrite: forcing every property into `required`
would make `params: {}` unrepresentable once the empty branch is dropped, which is
exactly how `buy_and_hold` is expressed. Advisory validation lets the model send
`{}`, which the server still accepts.
"""

DIALECTS: dict[str, SchemaDialect] = {d.name: d for d in (ANTHROPIC, OPENAI, OPENAI_STRICT, CEREBRAS)}


def sanitize(schema: dict[str, Any], dialect: SchemaDialect) -> dict[str, Any]:
    working = copy.deepcopy(schema)
    defs = {**working.get("$defs", {}), **working.get("definitions", {})}

    if dialect.inline_refs:
        working = _inline(working, defs, ())
        working.pop("$defs", None)
        working.pop("definitions", None)

    # The root carries the tool's parameter object and must stay typed even when it
    # declares nothing: a provider expects `parameters` to be an object.
    return _apply(working, dialect, root=True)


def _inline(node: Any, defs: dict[str, Any], stack: tuple[str, ...]) -> Any:
    """Replaces `$ref` with its definition.

    A definition that refers to itself would expand forever, so a ref already on the
    stack collapses to an unconstrained object: providers that cannot follow `$ref`
    also cannot express recursion, and a permissive node is better than a crash.
    """
    if isinstance(node, list):
        return [_inline(item, defs, stack) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        if name in stack:
            return {"type": "object"}
        target = defs.get(name)
        if target is None:
            return {k: v for k, v in node.items() if k != "$ref"} or {"type": "object"}
        merged = {**_inline(copy.deepcopy(target), defs, (*stack, name))}
        for key, value in node.items():
            if key != "$ref":
                merged[key] = _inline(value, defs, stack)
        return merged

    return {key: _inline(value, defs, stack) for key, value in node.items()}


def _apply(node: Any, dialect: SchemaDialect, root: bool = False) -> Any:
    if isinstance(node, list):
        return [_apply(item, dialect) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in dialect.strip:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _apply(sub, dialect) for name, sub in value.items()}
        elif key in COMBINATORS and isinstance(value, list):
            out[key] = _branches(value, dialect)
        elif key in STRUCTURAL or key in COMBINATORS:
            out[key] = _apply(value, dialect)
        else:
            out[key] = _apply(value, dialect) if isinstance(value, dict | list) else value

    if _is_object(out):
        if dialect.force_additional_properties_false:
            out["additionalProperties"] = False
        if dialect.require_all_properties:
            names = list(out.get("properties", {}))
            optional = [n for n in names if n not in set(out.get("required", []))]
            out["required"] = names
            for name in optional:
                out["properties"][name] = _nullable(out["properties"][name])

    if dialect.untype_open_objects and not root and is_empty_object(out):
        return {k: v for k, v in out.items() if k not in ("type", "additionalProperties", "required")}
    return out


def _branches(members: list[Any], dialect: SchemaDialect) -> list[Any]:
    """Applies the dialect to each union member, dropping the ones it cannot express.

    Never returns an empty list: if every member is a property-less object there is
    nothing better to send than the original, and a visibly wrong schema beats a
    silently missing parameter.
    """
    if dialect.drop_empty_object_branches:
        # Filtered before the dialect is applied, because untype_open_objects would
        # otherwise erase the very `type: object` this test looks for.
        kept = [member for member in members if not is_empty_object(member)]
        members = kept or members
    return [_apply(member, dialect) for member in members]


def is_empty_object(node: Any) -> bool:
    """An object that declares no properties and no alternatives."""
    return (
        isinstance(node, dict)
        and _is_object(node)
        and not node.get("properties")
        and not node.get("anyOf")
        and not node.get("oneOf")
    )


def _is_object(node: dict[str, Any]) -> bool:
    return node.get("type") == "object" or "properties" in node


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    """Strict mode requires every property in `required`, so optional ones become nullable."""
    if "anyOf" in node:
        variants = node["anyOf"]
        if not any(v.get("type") == "null" for v in variants if isinstance(v, dict)):
            return {**node, "anyOf": [*variants, {"type": "null"}]}
        return node
    declared = node.get("type")
    if declared is None:
        return node
    if isinstance(declared, list):
        return node if "null" in declared else {**node, "type": [*declared, "null"]}
    return {**node, "type": [declared, "null"]}
