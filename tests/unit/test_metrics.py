import numpy as np
import pytest

from quantlab.metrics.hedging import hedging_metrics


def test_hedging_metrics_hand_calculated():
    result = hedging_metrics(np.array([0.0, -2.0, 4.0]), np.array([1.0, -1.0, 2.0]))
    assert result["mean_absolute_hedging_error"] == pytest.approx(2.0)
    assert result["pnl_mean"] == pytest.approx(2.0 / 3.0)
    assert result["pnl_std"] == pytest.approx(np.std([1.0, -1.0, 2.0]))
    assert result["final_hedging_error"] == pytest.approx(4.0)
