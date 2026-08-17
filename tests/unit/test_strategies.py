import numpy as np

from quantlab.strategies.delta_hedging import DeltaHedgingStrategy
from quantlab.strategies.sma import SMAStrategy


def test_delta_strategy_produces_valid_hedge_change():
    strategy = DeltaHedgingStrategy(K=100, r=0.05, sigma=0.2, T=1.0, steps=4)
    action = strategy.action(np.zeros(6), {"t": 0, "price": 100.0})
    assert 0.0 <= action <= 1.0


def test_sma_strategy_is_deterministic_for_fixed_prices():
    prices = [1, 2, 3, 4, 5]
    first = SMAStrategy(short_window=2, long_window=3)
    second = SMAStrategy(short_window=2, long_window=3)
    first_actions = [first.action(np.zeros(6), {"price": p}) for p in prices]
    second_actions = [second.action(np.zeros(6), {"price": p}) for p in prices]
    assert first_actions == second_actions
