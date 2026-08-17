"""Minimal reusable backtesting engine."""

from __future__ import annotations

import numpy as np

from quantlab.pricing.black_scholes import bs_call_price, bs_delta, bs_gamma
from quantlab.strategies.base import Strategy, StrategyResult


class BacktestEngine:
    """Run a strategy over a fixed price path using the hedging accounting."""

    def __init__(self, K: float, r: float, sigma: float, T: float, cost_rate: float = 0.001) -> None:
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.cost_rate = cost_rate

    def run(self, prices: np.ndarray, strategy: Strategy) -> StrategyResult:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1 or len(prices) < 2:
            raise ValueError("prices must be a one-dimensional path with at least two values.")
        if np.any(prices <= 0):
            raise ValueError("prices must be positive.")

        n_steps = len(prices) - 1
        strategy.reset()
        option_values = np.empty(n_steps + 1, dtype=np.float64)
        hedge_positions = np.zeros(n_steps + 1, dtype=np.float64)
        portfolio_pnl = np.zeros(n_steps + 1, dtype=np.float64)
        hedging_error = np.zeros(n_steps + 1, dtype=np.float64)
        transaction_costs = np.zeros(n_steps + 1, dtype=np.float64)
        actions = np.zeros(n_steps, dtype=np.float64)

        option_values[0] = bs_call_price(prices[0], self.K, self.T, self.r, self.sigma)
        current_option = option_values[0]
        hedge_position = 0.0

        for t in range(n_steps):
            remaining_T = self.T * (1 - t / n_steps)
            obs = self._observation(prices[t], remaining_T, hedge_position)
            info = {
                "t": t,
                "price": float(prices[t]),
                "option_value": float(current_option),
                "hedge_position": hedge_position,
            }
            action = float(np.clip(strategy.action(obs, info), -1.0, 1.0))
            actions[t] = action
            cost = self.cost_rate * abs(action) * prices[t]
            hedge_position += action

            next_remaining_T = self.T * (1 - (t + 1) / n_steps)
            next_option = bs_call_price(prices[t + 1], self.K, next_remaining_T, self.r, self.sigma)
            step_pnl = next_option - current_option - hedge_position * (prices[t + 1] - prices[t]) - cost

            option_values[t + 1] = next_option
            hedge_positions[t + 1] = hedge_position
            transaction_costs[t + 1] = cost
            portfolio_pnl[t + 1] = step_pnl
            hedging_error[t + 1] = hedging_error[t] + step_pnl
            current_option = next_option

        return StrategyResult(
            name=strategy.name,
            prices=prices,
            option_values=option_values,
            hedge_positions=hedge_positions,
            portfolio_pnl=portfolio_pnl,
            hedging_error=hedging_error,
            transaction_costs=transaction_costs,
            actions=actions,
        )

    def _observation(self, S: float, remaining_T: float, hedge_position: float) -> np.ndarray:
        return np.array(
            [
                np.log(S / self.K),
                bs_delta(S, self.K, remaining_T, self.r, self.sigma),
                bs_gamma(S, self.K, remaining_T, self.r, self.sigma),
                remaining_T / self.T,
                self.sigma,
                hedge_position,
            ],
            dtype=np.float32,
        )
