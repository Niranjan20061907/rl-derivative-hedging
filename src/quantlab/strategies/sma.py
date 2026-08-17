"""Simple moving-average strategy for future backtesting examples."""

from __future__ import annotations

from collections import deque

import numpy as np

from quantlab.strategies.base import Strategy


class SMAStrategy(Strategy):
    """Deterministic long/flat signal based on short and long moving averages."""

    name = "sma"

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("SMA windows must be positive.")
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")
        self.short_window = short_window
        self.long_window = long_window
        self.prices: deque[float] = deque(maxlen=long_window)
        self.position = 0.0

    def reset(self) -> None:
        self.prices.clear()
        self.position = 0.0

    def action(self, observation: np.ndarray, info: dict) -> float:
        del observation
        self.prices.append(float(info["price"]))
        if len(self.prices) < self.long_window:
            target = 0.0
        else:
            values = np.asarray(self.prices, dtype=np.float64)
            target = 1.0 if values[-self.short_window :].mean() > values.mean() else 0.0
        change = target - self.position
        self.position = target
        return float(change)
