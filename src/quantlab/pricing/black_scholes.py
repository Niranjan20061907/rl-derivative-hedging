"""European-call prices and Greeks; scalar reference and vectorized implementation."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr


def call_features(S, K: float, T, r: float, sigma: float) -> np.ndarray:
    """Return broadcast price/delta/gamma arrays, with features on the last axis."""
    spots, times = np.broadcast_arrays(np.asarray(S, dtype=float), np.asarray(T, dtype=float))
    if not (np.isfinite(spots).all() and np.isfinite(times).all() and np.isfinite([K, r, sigma]).all()):
        raise ValueError("Pricing inputs must be finite.")
    if np.any(spots <= 0) or K <= 0 or np.any(times < 0) or sigma < 0:
        raise ValueError("Require positive S/K and nonnegative T/sigma.")
    strike = K * np.exp(-r * times)
    price = np.maximum(spots - strike, 0.0)
    delta = (spots > strike).astype(float)
    gamma = np.zeros_like(spots)
    active = (times > 0) & (sigma > 0)
    root = sigma * np.sqrt(np.where(active, times, 1.0))
    root = np.where(active, root, 1.0)
    d1 = (np.log(spots / K) + (r + 0.5 * sigma**2) * times) / root
    price = np.where(active, spots * ndtr(d1) - strike * ndtr(d1 - root), price)
    delta = np.where(active, ndtr(d1), delta)
    gamma = np.where(active, np.exp(-0.5 * d1**2) / (math.sqrt(2 * math.pi) * spots * root), gamma)
    return np.stack([price, delta, gamma], axis=-1)


def scalar_call_features(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float, float]:
    """Independent scalar formula for the reference engine."""
    if not all(math.isfinite(x) for x in (S, K, T, r, sigma)):
        raise ValueError("Pricing inputs must be finite.")
    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Require positive S/K and nonnegative T/sigma.")
    strike = K * math.exp(-r * T)
    if T == 0 or sigma == 0:
        return max(S - strike, 0.0), float(S > strike), 0.0
    root = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / root
    cdf1 = 0.5 * math.erfc(-d1 / math.sqrt(2))
    cdf2 = 0.5 * math.erfc(-(d1 - root) / math.sqrt(2))
    return S * cdf1 - strike * cdf2, cdf1, math.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * S * root)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return scalar_call_features(S, K, T, r, sigma)[0]


def bs_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return scalar_call_features(S, K, T, r, sigma)[1]


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return scalar_call_features(S, K, T, r, sigma)[2]
