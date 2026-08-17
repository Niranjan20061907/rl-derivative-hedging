"""Black-Scholes pricing and Greeks for European call options."""

from __future__ import annotations

import math

from scipy.stats import norm


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("S must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T cannot be negative.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes European call option price."""
    _validate_inputs(S, K, T, sigma)
    if T == 0:
        return max(S - K, 0.0)
    if sigma == 0:
        return max(S - K * math.exp(-r * T), 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes delta for a European call option."""
    _validate_inputs(S, K, T, sigma)
    if T == 0:
        return 1.0 if S > K else 0.0
    if sigma == 0:
        return 1.0 if S > K * math.exp(-r * T) else 0.0

    return float(norm.cdf(_d1(S, K, T, r, sigma)))


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return the Black-Scholes gamma for a European call option."""
    _validate_inputs(S, K, T, sigma)
    if T == 0 or sigma == 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))
