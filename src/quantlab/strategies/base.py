"""Base strategy contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StrategyResult:
    """Structured output from a strategy backtest."""

    name: str
    prices: np.ndarray
    option_values: np.ndarray
    hedge_positions: np.ndarray
    portfolio_pnl: np.ndarray
    hedging_error: np.ndarray
    transaction_costs: np.ndarray
    actions: np.ndarray


class Strategy:
    """Minimal strategy interface used by the backtesting engine."""

    name = "strategy"

    def reset(self) -> None:
        """Reset any stateful strategy internals before a new run."""

    def action(self, observation: np.ndarray, info: dict) -> float:
        """Return a hedge-position change for the current timestep."""
        raise NotImplementedError
