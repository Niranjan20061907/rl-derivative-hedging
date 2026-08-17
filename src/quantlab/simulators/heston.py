"""Heston stochastic-volatility simulator."""

from __future__ import annotations

import numpy as np


def simulate_heston(
    S0: float,
    v0: float,
    rho: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    steps: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one Heston price and variance path using Euler discretization."""
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if v0 < 0 or theta < 0:
        raise ValueError("v0 and theta cannot be negative.")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")
    if kappa < 0:
        raise ValueError("kappa cannot be negative.")
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
    variances = np.empty(steps + 1, dtype=np.float64)
    prices[0] = S0
    variances[0] = v0

    for t in range(1, steps + 1):
        z1 = local_rng.normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * local_rng.normal()
        v_prev = variances[t - 1]
        dv = kappa * (theta - v_prev) * dt + sigma * np.sqrt(max(v_prev, 0.0)) * np.sqrt(dt) * z2
        variances[t] = max(v_prev + dv, 0.0)
        dS = prices[t - 1] * np.sqrt(max(v_prev, 0.0)) * np.sqrt(dt) * z1
        prices[t] = max(prices[t - 1] + dS, 1e-12)

    return prices, variances
