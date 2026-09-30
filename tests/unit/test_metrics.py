import numpy as np
import pytest

from quantlab.metrics.hedging import hedging_metrics


def test_hedging_metrics_hand_calculated():
    result = hedging_metrics(np.array([0.0, -2.0, 4.0]), np.array([1.0, -1.0, 2.0]))
    assert result["mean_absolute_hedging_error"] == pytest.approx(2.0)
    assert result["pnl_mean"] == pytest.approx(2.0 / 3.0)
    assert result["pnl_std"] == pytest.approx(np.std([1.0, -1.0, 2.0]))
    assert result["final_hedging_error"] == pytest.approx(4.0)


def test_terminal_tail_and_signed_bias():
    from quantlab.metrics.hedging import terminal_metrics

    result = terminal_metrics([-4, 4], [1, 3], [2, 4])
    assert result["terminal_bias"] == 0
    assert result["terminal_mae"] == result["terminal_rmse"] == 4
    assert result["worst_5pct_loss_mean"] == 4
    assert result["mean_transaction_cost"] == 2


def test_paired_bootstrap_identical_paths():
    from quantlab.metrics.hedging import paired_bootstrap

    result = paired_bootstrap([1, -2, 3], [1, -2, 3], samples=30)
    assert result["mae_difference_ci95"] == [0, 0]
    assert result["rmse_difference_ci95"] == [0, 0]
