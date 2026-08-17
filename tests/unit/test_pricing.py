import pytest

from quantlab.pricing.black_scholes import bs_call_price, bs_delta, bs_gamma


def test_black_scholes_known_values():
    assert bs_call_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.4506, rel=1e-4)
    assert bs_delta(100, 100, 1.0, 0.05, 0.2) == pytest.approx(0.6368, rel=1e-4)
    assert bs_gamma(100, 100, 1.0, 0.05, 0.2) == pytest.approx(0.01876, rel=1e-3)


def test_black_scholes_validates_inputs():
    with pytest.raises(ValueError):
        bs_call_price(0, 100, 1.0, 0.05, 0.2)
