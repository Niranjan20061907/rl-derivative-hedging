"""Shared, self-financing portfolio accounting for one short European call."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Number = float | np.ndarray

CONTRACT_VERSION = "funded-call-v2-observation7"


@dataclass
class Portfolio:
    cash: Number
    position: Number = 0.0
    error: Number = 0.0


def execute_step(
    state: Portfolio,
    action: Number,
    spot: Number,
    next_spot: Number,
    option_value: Number,
    dt: float,
    r: float,
    cost_rate: float,
    terminal: bool = False,
) -> dict[str, Number]:
    """Trade now, accrue cash, then mark/settle. Costs use actual traded shares."""
    batch = isinstance(action, np.ndarray)
    if not (np.isfinite(action).all() if batch else math.isfinite(action)):
        raise ValueError("Action must be finite.")
    if batch:
        target = np.clip(state.position + np.clip(action, -1.0, 1.0), 0.0, 1.0)
    else:
        target = min(1.0, max(0.0, state.position + min(1.0, max(-1.0, action))))
    executed = target - state.position
    cost = cost_rate * abs(executed) * spot
    cash = (state.cash - executed * spot - cost) * math.exp(r * dt)
    liquidation = cost_rate * target * next_spot if terminal else 0.0
    portfolio_value = cash + target * next_spot - liquidation
    error = portfolio_value - option_value
    pnl = error - state.error
    if terminal:
        cash = error  # Sell stock, pay trading costs, and settle the liability.
        position = np.zeros_like(target) if batch else 0.0
    else:
        position = target
    state.cash, state.position, state.error = cash, position, error
    return {
        "cash_balance": cash,
        "hedge_position": position,
        "portfolio_value": portfolio_value,
        "hedging_error": error,
        "pnl": pnl,
        "transaction_cost": cost + liquidation,
        "executed_action": executed,
        "turnover": abs(executed) + (target if terminal else 0.0),
    }


def observation(
    spot: Number, K: float, remaining: float, T: float, sigma: float, features, state: Portfolio, scale: float
) -> np.ndarray:
    return np.asarray(
        [math.log(spot / K), features[1], features[2], remaining / T, sigma, state.position, state.cash / scale],
        dtype=np.float32,
    )
