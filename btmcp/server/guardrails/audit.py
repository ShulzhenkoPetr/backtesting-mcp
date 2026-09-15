from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.shared.exceptions import MCPError

from btmcp.server.shaping import count_tokens, serialise

FIELDS = (
    "ts",
    "tool",
    "surface",
    "tier",
    "arg_hash",
    "dataset_id",
    "latency_ms",
    "result_bytes",
    "result_tokens",
    "outcome",
    "error_code",
)


def _measurable(result: Any) -> Any:
    """Extracts the payload a client actually puts in a context window.

    The 2026-07-28 result envelope carries the same data twice: `structuredContent`
    and a pretty-printed duplicate in `content`. No client feeds both to a model, so
    counting the envelope would roughly double every figure. Structured content is
    the canonical copy and is what every measurement in this repo uses.
    """
    if isinstance(result, dict):
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        if result.get("content") is not None:
            return result["content"]
    return result


def arg_hash(arguments: Any) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()[:16]


class AuditLog:
    """JSONL, one line per tool call. Field names are part of the eval contract.

    The eval asserts against this rather than against transcripts, so it is written
    server-side and does not depend on the client honouring MCP logging (which the
    2026-07-28 revision deprecates anyway).
    """

    def __init__(self, path: Path | str | None, surface: str, tier: str, dataset_id: str) -> None:
        self.path = Path(path) if path else None
        self.surface = surface
        self.tier = tier
        self.dataset_id = dataset_id
        self.records: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.path:
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def record(
        self,
        tool: str,
        arguments: Any,
        latency_ms: float,
        outcome: str,
        result: Any = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        text = serialise(_measurable(result)) if result is not None else ""
        record = {
            "ts": dt.datetime.now(dt.UTC).isoformat(),
            "tool": tool,
            "surface": self.surface,
            "tier": self.tier,
            "arg_hash": arg_hash(arguments),
            "dataset_id": self.dataset_id,
            "latency_ms": round(latency_ms, 3),
            "result_bytes": len(text.encode()),
            "result_tokens": count_tokens(text) if text else 0,
            "outcome": outcome,
            "error_code": error_code,
        }
        self.write(record)
        return record


class AuditMiddleware:
    def __init__(self, log: AuditLog) -> None:
        self.log = log

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        if ctx.method != "tools/call":
            return await call_next(ctx)

        params = ctx.params if isinstance(ctx.params, dict) else {}
        tool = str(params.get("name", "<unknown>"))
        arguments = params.get("arguments", {})
        started = time.perf_counter()
        try:
            result = await call_next(ctx)
        except MCPError as exc:
            code = (exc.error.data or {}).get("code") if hasattr(exc, "error") else None
            self.log.record(
                tool,
                arguments,
                (time.perf_counter() - started) * 1000.0,
                outcome="error",
                error_code=str(code) if code else "MCP_ERROR",
            )
            raise
        except Exception:
            self.log.record(
                tool,
                arguments,
                (time.perf_counter() - started) * 1000.0,
                outcome="error",
                error_code="INTERNAL_ERROR",
            )
            raise

        self.log.record(
            tool, arguments, (time.perf_counter() - started) * 1000.0, outcome="ok", result=result
        )
        return result
