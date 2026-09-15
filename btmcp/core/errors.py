from __future__ import annotations

from typing import Any


class BtmcpError(Exception):
    """Base for every failure that is allowed to cross the tool boundary.

    Carries the fields the wire envelope needs (E4-T4) so no handler has to
    reconstruct them, and so no stack trace is ever the thing a caller sees.
    """

    code: str = "INTERNAL_ERROR"
    retryable: bool = False

    def __init__(self, cause: str, suggested_action: str, **context: Any) -> None:
        super().__init__(cause)
        self.cause = cause
        self.suggested_action = suggested_action
        self.context = context

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "cause": self.cause,
            "suggested_action": self.suggested_action,
            "retryable": self.retryable,
        }
        if self.context:
            payload["context"] = self.context
        return payload


class UnknownSymbol(BtmcpError):
    code = "UNKNOWN_SYMBOL"


class InsufficientHistory(BtmcpError):
    code = "INSUFFICIENT_HISTORY"


class InvalidSpec(BtmcpError):
    code = "INVALID_SPEC"


class MissingCostModel(BtmcpError):
    code = "MISSING_COST_MODEL"


class TierDenied(BtmcpError):
    code = "TIER_DENIED"


class LookaheadError(BtmcpError):
    """Raised when a caller asks for data at or beyond a timestamp it cannot see.

    Reaching this exception means a defence in depth fired. The primary guarantee
    is structural: `BarStore` has no code path that returns a bar after `as_of`,
    so the ordinary way to request the future is to receive an empty frame, not
    an error.
    """

    code = "LOOKAHEAD_REFUSED"


class BudgetExceeded(BtmcpError):
    code = "BUDGET_EXCEEDED"


class RunNotFound(BtmcpError):
    code = "RUN_NOT_FOUND"
