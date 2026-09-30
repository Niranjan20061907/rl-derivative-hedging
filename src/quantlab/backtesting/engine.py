"""Reference and optimized backtesting over identical self-financing accounting."""

from __future__ import annotations

import numpy as np

from quantlab.backtesting.accounting import Portfolio, execute_step, observation
from quantlab.pricing.black_scholes import call_features, scalar_call_features
from quantlab.strategies.base import Strategy, StrategyResult


class BacktestEngine:
    def __init__(
        self, K: float, r: float, sigma: float, T: float, cost_rate: float = 0.001, optimized: bool = True
    ) -> None:
        if not np.isfinite([K, r, sigma, T, cost_rate]).all():
            raise ValueError("Engine parameters must be finite.")
        if K <= 0 or T <= 0 or sigma < 0 or cost_rate < 0:
            raise ValueError("Invalid engine parameters.")
        self.K, self.r, self.sigma, self.T = K, r, sigma, T
        self.cost_rate, self.optimized = cost_rate, optimized

    def run(self, prices: np.ndarray, strategy: Strategy) -> StrategyResult:
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1 or len(prices) < 2 or not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("prices must be a finite positive 1D path with at least two values.")
        n = len(prices) - 1
        times = self.T * (1 - np.arange(n + 1) / n)
        features = call_features(prices, self.K, times, self.r, self.sigma) if self.optimized else None
        initial = scalar_call_features(prices[0], self.K, self.T, self.r, self.sigma)
        state = Portfolio(cash=initial[0])
        scale = max(initial[0], 1e-8)
        values = {
            key: np.zeros(n + 1)
            for key in (
                "hedge_positions",
                "portfolio_pnl",
                "hedging_error",
                "transaction_costs",
                "cash_balances",
                "portfolio_values",
                "turnover",
            )
        }
        option_values = np.empty(n + 1)
        option_values[0] = initial[0]
        values["cash_balances"][0] = values["portfolio_values"][0] = initial[0]
        actions = np.zeros(n)
        strategy.reset()
        for t in range(n):
            current = (
                features[t]
                if features is not None
                else scalar_call_features(prices[t], self.K, times[t], self.r, self.sigma)
            )
            obs = observation(prices[t], self.K, times[t], self.T, self.sigma, current, state, scale)
            info = {
                "t": t,
                "price": float(prices[t]),
                "option_value": float(current[0]),
                "hedge_position": state.position,
                "cash_balance": state.cash,
            }
            raw = np.asarray(strategy.action(obs, info))
            if raw.shape != ():
                raise ValueError("Strategy actions must be scalar.")
            next_value = (
                float(features[t + 1, 0])
                if features is not None
                else scalar_call_features(prices[t + 1], self.K, times[t + 1], self.r, self.sigma)[0]
            )
            outcome = execute_step(
                state,
                float(raw),
                prices[t],
                prices[t + 1],
                next_value,
                self.T / n,
                self.r,
                self.cost_rate,
                terminal=t == n - 1,
            )
            actions[t] = outcome["executed_action"]
            option_values[t + 1] = next_value
            for array, key in (
                ("hedge_positions", "hedge_position"),
                ("portfolio_pnl", "pnl"),
                ("hedging_error", "hedging_error"),
                ("transaction_costs", "transaction_cost"),
                ("cash_balances", "cash_balance"),
                ("portfolio_values", "portfolio_value"),
                ("turnover", "turnover"),
            ):
                values[array][t + 1] = outcome[key]
        return StrategyResult(
            name=strategy.name,
            prices=prices,
            option_values=option_values,
            actions=actions,
            terminal_error=state.error,
            **values,
        )
