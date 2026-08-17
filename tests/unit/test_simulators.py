import numpy as np
import pytest

from quantlab.simulators.gbm import simulate_gbm
from quantlab.simulators.heston import simulate_heston


def test_gbm_shape_positive_and_reproducible():
    first = simulate_gbm(100, 0.05, 0.2, 1.0, 8, seed=7)
    second = simulate_gbm(100, 0.05, 0.2, 1.0, 8, seed=7)
    assert first.shape == (9,)
    assert np.all(first > 0)
    np.testing.assert_allclose(first, second)


def test_heston_shape_positive_and_reproducible():
    prices, variances = simulate_heston(100, 0.04, -0.7, 2.0, 0.04, 0.5, 1.0, 8, seed=7)
    prices_2, variances_2 = simulate_heston(100, 0.04, -0.7, 2.0, 0.04, 0.5, 1.0, 8, seed=7)
    assert prices.shape == (9,)
    assert variances.shape == (9,)
    assert np.all(prices > 0)
    assert np.all(variances >= 0)
    np.testing.assert_allclose(prices, prices_2)
    np.testing.assert_allclose(variances, variances_2)


def test_gbm_validates_parameters():
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, 0.2, 1.0, 0)
