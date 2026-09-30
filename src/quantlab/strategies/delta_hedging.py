"""Daily and periodic delta baselines, using the executed position."""

from __future__ import annotations

import numpy as np

from quantlab.strategies.base import Strategy


class DeltaHedgingStrategy(Strategy):
    name = "delta"

    def __init__(self, K: float, r: float, sigma: float, T: float, steps: int, interval: int = 1):
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("interval must be a positive integer.")
        self.K, self.r, self.sigma, self.T, self.steps = K, r, sigma, T, steps
        self.interval = interval
        self.name = "delta" if interval == 1 else f"delta_{interval}"

    def action(self, observation: np.ndarray, info: dict) -> float:
        if info["t"] % self.interval:
            return 0.0
        return float(observation[1] - info["hedge_position"])


def delta_hedge(prices: np.ndarray, K: float, r: float, sigma: float, T: float, cost_rate: float = 0.0):
    from quantlab.backtesting.engine import BacktestEngine

    result = BacktestEngine(K, r, sigma, T, cost_rate).run(
        prices, DeltaHedgingStrategy(K, r, sigma, T, len(prices) - 1)
    )
    return {
        "option_values": result.option_values,
        "deltas": result.hedge_positions,
        "portfolio": result.portfolio_pnl,
        "hedging_error": result.hedging_error,
        "transaction_costs": result.transaction_costs,
    }
