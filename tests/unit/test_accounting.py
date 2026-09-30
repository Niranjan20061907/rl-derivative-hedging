from dataclasses import replace

import numpy as np
import pytest

from quantlab.backtesting.accounting import Portfolio, execute_step
from quantlab.backtesting.engine import BacktestEngine
from quantlab.environments.hedging_env import HedgingEnv, HedgingEnvParams
from quantlab.pricing.black_scholes import call_features, scalar_call_features
from quantlab.simulators.gbm import simulate_gbm, simulate_gbm_batch
from quantlab.strategies.delta_hedging import DeltaHedgingStrategy


def test_hand_calculated_trade_interest_and_settlement():
    state = Portfolio(cash=10)
    first = execute_step(state, 0.5, 100, 110, 15, 1, np.log(1.1), 0.01)
    assert first["cash_balance"] == pytest.approx(-44.55)
    assert first["portfolio_value"] == pytest.approx(10.45)
    assert first["hedging_error"] == pytest.approx(-4.55)
    final = execute_step(state, 0, 110, 120, 20, 1, 0, 0.01, terminal=True)
    assert final["transaction_cost"] == pytest.approx(0.6)
    assert state.position == 0
    assert state.cash == pytest.approx(-5.15)
    assert first["pnl"] + final["pnl"] == pytest.approx(state.error)


def test_deterministic_replication_with_financing():
    prices = simulate_gbm(100, 0.05, 0, 1, 252, seed=42)
    result = BacktestEngine(100, 0.05, 0, 1, 0).run(prices, DeltaHedgingStrategy(100, 0.05, 0, 1, 252))
    assert result.terminal_error == pytest.approx(0, abs=1e-10)
    assert result.cash_balances[-1] == pytest.approx(result.terminal_error)
    assert result.portfolio_values[-1] == pytest.approx(result.option_values[-1], abs=1e-10)
    assert result.hedge_positions[-1] == 0


@pytest.mark.parametrize("sigma", [0, 0.1, 0.2, 0.4])
@pytest.mark.parametrize("cost", [0, 0.001])
def test_reference_optimized_environment_parity_and_reward(sigma, cost):
    p = HedgingEnvParams(steps=20, sigma=sigma, cost_rate=cost)
    env = HedgingEnv(p)
    obs, info = env.reset(seed=123)
    strategy = DeltaHedgingStrategy(p.K, p.r, p.sigma, p.T, p.steps)
    errors, rewards, costs = [0], [], [0]
    for _ in range(p.steps):
        obs, reward, _, _, info = env.step([strategy.action(obs, info)])
        errors.append(info["hedging_error"])
        rewards.append(reward)
        costs.append(info["transaction_cost"])
    results = [
        BacktestEngine(p.K, p.r, p.sigma, p.T, cost, optimized=flag).run(env.prices, strategy) for flag in (False, True)
    ]
    for result in results:
        np.testing.assert_allclose(result.hedging_error, errors, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.transaction_costs, costs, atol=1e-10, rtol=1e-10)
    assert sum(rewards) == pytest.approx(-((info["hedging_error"] / env.scale) ** 2), abs=1e-10)


def test_bounded_positions_and_actual_costs():
    env = HedgingEnv(HedgingEnvParams(steps=4))
    env.reset(seed=1)
    _, _, _, _, first = env.step([10])
    assert first["hedge_position"] == 1
    _, _, _, _, second = env.step([1])
    assert second["executed_action"] == second["transaction_cost"] == 0
    env.step([-10])
    _, _, done, _, last = env.step([-10])
    assert done and last["hedge_position"] == 0
    assert last["transaction_cost"] == 0
    with pytest.raises(RuntimeError):
        env.step([0])


@pytest.mark.parametrize("action", [[], [0, 1], [[0]], [np.nan], [np.inf], 0])
def test_malformed_action_rejected(action):
    env = HedgingEnv(steps=2)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize("change", [{"T": 0}, {"sigma": np.nan}, {"cost_rate": -1}, {"steps": 1.5}])
def test_environment_validation(change):
    with pytest.raises(ValueError):
        replace(HedgingEnvParams(), **change)


def test_batch_paths_are_chunk_independent():
    paths = simulate_gbm_batch(100, 0.05, 0.2, 1, 12, range(10, 17))
    for seed, path in zip(range(10, 17), paths, strict=True):
        np.testing.assert_array_equal(path, simulate_gbm(100, 0.05, 0.2, 1, 12, seed=seed))
    chunks = np.concatenate(
        [
            simulate_gbm_batch(100, 0.05, 0.2, 1, 12, range(10, 13)),
            simulate_gbm_batch(100, 0.05, 0.2, 1, 12, range(13, 17)),
        ]
    )
    np.testing.assert_array_equal(paths, chunks)


def test_scalar_vector_prices_at_boundaries():
    spots = np.array([1.0, 99.0, 100.0, 101.0, 10000.0])
    for sigma in (0, 0.2):
        for time in (0, 1e-8, 1):
            vector = call_features(spots, 100, time, -0.01, sigma)
            scalar = [scalar_call_features(s, 100, time, -0.01, sigma) for s in spots]
            np.testing.assert_allclose(vector, scalar, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("prices", [[100, np.nan], [100, np.inf], [100, -1]])
def test_bad_paths_rejected(prices):
    with pytest.raises(ValueError):
        BacktestEngine(100, 0.05, 0.2, 1).run(prices, DeltaHedgingStrategy(100, 0.05, 0.2, 1, 1))


def test_cash_funding_no_hedge_and_terminal_costs():
    from quantlab.rl.evaluate import NoHedgeStrategy

    prices = np.array([100.0, 110.0])
    result = BacktestEngine(100, 0.05, 0.2, 1, 0.001).run(prices, NoHedgeStrategy())
    premium = scalar_call_features(100, 100, 1, 0.05, 0.2)[0]
    assert result.terminal_error == pytest.approx(premium * np.exp(0.05) - 10)
    assert result.transaction_costs.sum() == result.turnover.sum() == 0


def test_strategy_cannot_observe_future_path():
    from quantlab.strategies.base import Strategy

    class CurrentOnly(Strategy):
        def action(self, observation, info):
            assert set(info) == {"t", "price", "option_value", "hedge_position", "cash_balance"}
            assert observation.shape == (7,)
            return 0.0

    BacktestEngine(100, 0.05, 0.2, 1).run([100, 101, 102], CurrentOnly())
