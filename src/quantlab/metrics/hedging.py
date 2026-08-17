"""Pure metric functions for hedging experiments."""

from __future__ import annotations

import numpy as np


def hedging_metrics(hedging_error: np.ndarray, pnl: np.ndarray) -> dict[str, float]:
    """Return core hedging metrics from error and P&L arrays."""
    errors = np.asarray(hedging_error, dtype=np.float64)
    pnl_values = np.asarray(pnl, dtype=np.float64)
    if errors.size == 0 or pnl_values.size == 0:
        raise ValueError("Metric inputs cannot be empty.")
    abs_error = np.abs(errors)
    return {
        "mean_absolute_hedging_error": float(np.mean(abs_error)),
        "p95_absolute_hedging_error": float(np.percentile(abs_error, 95)),
        "pnl_mean": float(np.mean(pnl_values)),
        "pnl_std": float(np.std(pnl_values)),
        "final_hedging_error": float(errors[-1]),
    }
