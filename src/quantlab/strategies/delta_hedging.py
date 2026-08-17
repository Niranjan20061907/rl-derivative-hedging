"""Black-Scholes delta hedging baseline."""

from __future__ import annotations

import numpy as np

from quantlab.pricing.black_scholes import bs_delta
from quantlab.strategies.base import Strategy


class DeltaHedgingStrategy(Strategy):
    """Discrete delta hedging strategy using Black-Scholes delta."""

    name = "delta"

    def __init__(self, K: float, r: float, sigma: float, T: float, steps: int) -> None:
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.hedge_position = 0.0

    def reset(self) -> None:
        self.hedge_position = 0.0

    def action(self, observation: np.ndarray, info: dict) -> float:
        del observation
        remaining_T = self.T * (1 - info["t"] / self.steps)
        target_delta = bs_delta(info["price"], self.K, remaining_T, self.r, self.sigma)
        hedge_change = target_delta - self.hedge_position
        self.hedge_position = target_delta
        return float(hedge_change)


def delta_hedge(prices: np.ndarray, K: float, r: float, sigma: float, T: float, cost_rate: float = 0.0):
    """Compatibility helper returning the original baseline-style fields plus costs."""
    from quantlab.backtesting.engine import BacktestEngine

    strategy = DeltaHedgingStrategy(K=K, r=r, sigma=sigma, T=T, steps=len(prices) - 1)
    result = BacktestEngine(K=K, r=r, sigma=sigma, T=T, cost_rate=cost_rate).run(prices, strategy)
    return {
        "option_values": result.option_values,
        "deltas": result.hedge_positions,
        "portfolio": result.portfolio_pnl,
        "hedging_error": result.hedging_error,
        "transaction_costs": result.transaction_costs,
    }
