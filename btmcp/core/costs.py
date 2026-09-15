from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

TRADING_DAYS = 252


class CostModel(BaseModel):
    """Transaction costs. `fee_bps` and `slippage_bps` deliberately have no defaults.

    A frictionless backtest is a legitimate thing to run and an illegitimate thing to
    run by accident. Requiring `{fee_bps: 0, slippage_bps: 0}` makes the zero-cost
    assumption an explicit statement in the run's config hash rather than a silent
    default that nobody chose.
    """

    model_config = ConfigDict(extra="forbid")

    fee_bps: float = Field(
        ge=0, le=1000, description="Commission per side, in basis points of traded notional."
    )
    slippage_bps: float = Field(ge=0, le=1000, description="Execution slippage per side, in basis points.")
    borrow_bps_annual: float = Field(
        default=0.0, ge=0, le=10000, description="Annual borrow cost on held notional."
    )

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.fee_bps + self.slippage_bps)

    def trade_cost(self, traded_notional: float) -> float:
        """Cost of moving `traded_notional`, charged per side on the absolute amount."""
        return abs(traded_notional) * (self.fee_bps + self.slippage_bps) / 10_000.0

    def carry_cost(self, held_notional: float, bars: int = 1) -> float:
        if self.borrow_bps_annual == 0.0:
            return 0.0
        return abs(held_notional) * (self.borrow_bps_annual / 10_000.0) * (bars / TRADING_DAYS)


FRICTIONLESS = CostModel(fee_bps=0.0, slippage_bps=0.0)
"""Explicit zero-cost model, for tests and for callers who mean it."""
