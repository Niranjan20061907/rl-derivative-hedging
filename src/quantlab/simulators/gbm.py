"""Geometric Brownian motion simulator."""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one GBM path with ``steps + 1`` prices."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if rng is not None and seed is not None:
        raise ValueError("Pass either seed or rng, not both.")

    local_rng = rng if rng is not None else np.random.default_rng(seed)
    dt = T / steps
    prices = np.empty(steps + 1, dtype=np.float64)
    prices[0] = S0

    shocks = local_rng.normal(size=steps)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    prices[1:] = S0 * np.exp(np.cumsum(increments))
    return prices
