from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """Produces predictive quantiles for a univariate series.

    Deliberately narrow: `context` is the history the strategy is allowed to see at
    its decision timestamp, and nothing in this signature can reach past it. A
    forecaster is therefore incapable of introducing lookahead even if it is wrong.
    """

    id: str

    def forecast(self, context: np.ndarray, horizon: int, quantiles: list[float]) -> np.ndarray:
        """Returns an array of shape (len(quantiles), horizon) of predicted levels."""
        ...
