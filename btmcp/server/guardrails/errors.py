from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from btmcp.core.errors import BtmcpError, InvalidSpec

INVALID_PARAMS = -32602
TOOL_ERROR = -32003


def to_mcp_error(exc: Exception) -> MCPError:
    """Maps a domain exception onto the wire. No stack trace crosses the boundary."""
    if isinstance(exc, BtmcpError):
        return MCPError(code=TOOL_ERROR, message=exc.cause, data=exc.to_payload())
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        key = ".".join(str(p) for p in first["loc"]) or "<root>"
        spec_error = InvalidSpec(
            cause=f"invalid arguments at '{key}': {first['msg']}",
            suggested_action="correct the named argument and retry; check the tool's input schema",
            offending_key=key,
        )
        return MCPError(code=INVALID_PARAMS, message=spec_error.cause, data=spec_error.to_payload())
    return MCPError(
        code=TOOL_ERROR,
        message="internal error",
        data={
            "code": "INTERNAL_ERROR",
            "cause": "the server failed to complete the call",
            "suggested_action": "retry once; if it persists the request is not serviceable as written",
            "retryable": True,
        },
    )


def typed_errors[T](fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return await fn(*args, **kwargs)
        except MCPError:
            raise
        except Exception as exc:
            raise to_mcp_error(exc) from None

    return wrapper
