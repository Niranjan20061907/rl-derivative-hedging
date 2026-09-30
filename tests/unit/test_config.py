from dataclasses import replace

import pytest

from quantlab.rl.config import EvaluationConfig, ExperimentConfig, PPOConfig, load_config


@pytest.mark.parametrize(
    "kwargs",
    [{"learning_rate": float("nan")}, {"gamma": 0.99}, {"n_steps": 2.5}, {"total_timesteps": 0}, {"batch_size": True}],
)
def test_invalid_ppo_configuration(kwargs):
    with pytest.raises(ValueError):
        replace(PPOConfig(), **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"episodes": 0}, {"chunk_size": 1.5}, {"seed_start": -1}, {"bootstrap_samples": float("inf")}]
)
def test_invalid_evaluation_configuration(kwargs):
    with pytest.raises(ValueError):
        replace(EvaluationConfig(), **kwargs)


def test_invalid_experiment_and_yaml(tmp_path):
    with pytest.raises(ValueError):
        ExperimentConfig(seed=-1)
    path = tmp_path / "bad.yaml"
    path.write_text("- invalid\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
