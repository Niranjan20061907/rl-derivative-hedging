"""Seeded single-path and batch geometric Brownian motion."""

from __future__ import annotations

import numpy as np


def _validate(S0, mu, sigma, T, steps):
    if not np.isfinite([S0, mu, sigma, T]).all() or S0 <= 0 or sigma < 0 or T <= 0:
        raise ValueError("Invalid GBM parameters.")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer.")


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    _validate(S0, mu, sigma, T, steps)
    if rng is not None and seed is not None:
        raise ValueError("Pass seed or rng, not both.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    return _from_shocks(S0, mu, sigma, T, steps, rng.normal(size=steps))


def _from_shocks(S0, mu, sigma, T, steps, shocks):
    dt = T / steps
    paths = np.empty((*shocks.shape[:-1], steps + 1))
    paths[..., 0] = S0
    paths[..., 1:] = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks, axis=-1))
    return paths


def simulate_gbm_batch(S0: float, mu: float, sigma: float, T: float, steps: int, seeds) -> np.ndarray:
    """One independent seed per row; chunk size does not affect any path."""
    _validate(S0, mu, sigma, T, steps)
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds cannot be empty.")
    shocks = np.stack([np.random.default_rng(seed).normal(size=steps) for seed in seeds])
    return _from_shocks(S0, mu, sigma, T, steps, shocks)
