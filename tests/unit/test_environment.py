import numpy as np

from quantlab.environments.hedging_env import HedgingEnv, HedgingEnvParams


def test_reset_observation_space_and_seed_reproducibility():
    env = HedgingEnv(HedgingEnvParams(steps=4))
    obs_1, info_1 = env.reset(seed=42)
    prices_1 = env.prices.copy()
    obs_2, info_2 = env.reset(seed=42)
    np.testing.assert_allclose(prices_1, env.prices)
    np.testing.assert_allclose(obs_1, obs_2)
    assert env.observation_space.contains(obs_1)
    assert info_1["t"] == info_2["t"] == 0


def test_step_return_structure_and_termination():
    env = HedgingEnv(HedgingEnvParams(steps=2))
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["t"] == 1

    _, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["t"] == 2


def test_action_is_clipped_to_action_space():
    env = HedgingEnv(HedgingEnvParams(steps=1))
    env.reset(seed=1)
    env.step(np.array([10.0], dtype=np.float32))
    assert env.hedge_position == 1.0
