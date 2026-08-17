import numpy as np

from quantlab.backtesting.engine import BacktestEngine
from quantlab.strategies.delta_hedging import DeltaHedgingStrategy


def test_backtest_engine_runs_small_deterministic_path():
    prices = np.array([100.0, 101.0, 99.0])
    strategy = DeltaHedgingStrategy(K=100, r=0.05, sigma=0.2, T=1.0, steps=2)
    result = BacktestEngine(K=100, r=0.05, sigma=0.2, T=1.0).run(prices, strategy)
    assert result.prices.shape == (3,)
    assert result.hedge_positions.shape == (3,)
    assert result.actions.shape == (2,)
    assert np.isfinite(result.hedging_error).all()
