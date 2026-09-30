"""Trajectory diagnostics and cross-path terminal risk, with paired uncertainty."""

from __future__ import annotations

import numpy as np


def hedging_metrics(hedging_error: np.ndarray, pnl: np.ndarray) -> dict[str, float]:
    errors, pnl = np.asarray(hedging_error, dtype=float), np.asarray(pnl, dtype=float)
    if errors.ndim != 1 or pnl.ndim != 1 or not errors.size or not pnl.size:
        raise ValueError("Metrics require nonempty 1D arrays.")
    if not np.isfinite(errors).all() or not np.isfinite(pnl).all():
        raise ValueError("Metric inputs must be finite.")
    return {
        "mean_absolute_hedging_error": float(np.abs(errors).mean()),
        "p95_absolute_hedging_error": float(np.percentile(np.abs(errors), 95)),
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std()),
        "final_hedging_error": float(errors[-1]),
    }


def terminal_metrics(errors, costs, turnover) -> dict[str, float]:
    errors, costs, turnover = [np.asarray(x, dtype=float) for x in (errors, costs, turnover)]
    if (
        errors.ndim != 1
        or not errors.size
        or costs.shape != errors.shape
        or turnover.shape != errors.shape
        or not all(np.isfinite(x).all() for x in (errors, costs, turnover))
    ):
        raise ValueError("Terminal metrics require matching finite nonempty 1D arrays.")
    # Positive shortfall only; select the largest ceil(5% * n) losses without interpolating a threshold.
    losses = np.maximum(-errors, 0)
    count = max(1, int(np.ceil(0.05 * len(errors))))
    return {
        "terminal_mae": float(np.abs(errors).mean()),
        "terminal_rmse": float(np.sqrt(np.mean(errors**2))),
        "terminal_bias": float(errors.mean()),
        "terminal_p95_absolute_error": float(np.percentile(abs(errors), 95)),
        "worst_5pct_loss_mean": float(np.sort(losses)[-count:].mean()),
        "mean_transaction_cost": float(costs.mean()),
        "mean_turnover": float(turnover.mean()),
    }


def paired_bootstrap(candidate, baseline, seed: int = 77, samples: int = 2000) -> dict:
    """Path-paired candidate-minus-baseline MAE/RMSE differences; negative is better."""
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.ndim != 1 or not a.size or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("Bootstrap requires matching finite nonempty paths.")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be positive.")
    rng = np.random.default_rng(seed)
    mae, rmse = np.empty(samples), np.empty(samples)
    # Bounded memory, independent of the number of resamples.
    for start in range(0, samples, 100):
        end = min(samples, start + 100)
        idx = rng.integers(len(a), size=(end - start, len(a)))
        x, y = a[idx], b[idx]
        mae[start:end] = np.mean(abs(x) - abs(y), axis=1)
        rmse[start:end] = np.sqrt(np.mean(x * x, axis=1)) - np.sqrt(np.mean(y * y, axis=1))
    return {
        "mae_difference": float(np.mean(abs(a) - abs(b))),
        "mae_difference_ci95": np.percentile(mae, [2.5, 97.5]).tolist(),
        "rmse_difference": float(np.sqrt(np.mean(a * a)) - np.sqrt(np.mean(b * b))),
        "rmse_difference_ci95": np.percentile(rmse, [2.5, 97.5]).tolist(),
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "unit": "paired price path",
    }
