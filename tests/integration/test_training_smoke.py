from dataclasses import replace

import pytest

from quantlab.environments.hedging_env import HedgingEnvParams
from quantlab.rl.config import ExperimentConfig, PPOConfig
from quantlab.rl.train import train


def test_training_smoke(tmp_path):
    pytest.importorskip("stable_baselines3")
    config = ExperimentConfig(
        seed=123,
        experiment_name="smoke",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        environment=HedgingEnvParams(steps=8),
        ppo=replace(PPOConfig(), total_timesteps=8, n_steps=8, batch_size=4, verbose=0),
    )
    metadata = train(config)
    assert metadata["total_timesteps"] == 8
    assert (tmp_path / "checkpoints" / "smoke_seed123.zip").exists()
