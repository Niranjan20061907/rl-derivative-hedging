import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quantlab.environments.hedging_env import HedgingEnvParams
from quantlab.rl.artifacts import load_policy
from quantlab.rl.config import EvaluationConfig, ExperimentConfig, PPOConfig
from quantlab.rl.evaluate import evaluate, policy_outcomes, run_env_episode
from quantlab.rl.train import train
from quantlab.simulators.gbm import simulate_gbm_batch
from quantlab.strategies.ppo_hedging import PPOHedgingStrategy


def test_training_evaluation_bundle_and_batch_parity(tmp_path):
    pytest.importorskip("stable_baselines3")
    config = ExperimentConfig(
        seed=123,
        experiment_name="smoke",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        environment=HedgingEnvParams(steps=8),
        ppo=replace(PPOConfig(), total_timesteps=16, n_steps=8, batch_size=4, verbose=0, validation_frequency=8),
        evaluation=EvaluationConfig(episodes=3, chunk_size=2, bootstrap_samples=20),
    )
    metadata = train(config)
    assert metadata["total_timesteps"] == metadata["actual_timesteps"] == 16
    model_path = Path(metadata["model_path"])
    assert model_path.exists()
    model, normalizer, manifest = load_policy(model_path, config.environment)
    assert normalizer.training is False
    assert normalizer.norm_reward is False
    assert manifest["validation_seed_start"] == 100000
    paths = simulate_gbm_batch(100, 0.05, 0.2, 1, 8, [200000, 200001])
    original_statistics = normalizer.obs_rms.mean.copy()
    batch_errors, _, _, _ = policy_outcomes(paths, config.environment, model, normalizer)
    from quantlab.environments.hedging_env import HedgingEnv

    for i, seed in enumerate([200000, 200001]):
        error, _, _ = run_env_episode(
            HedgingEnv(config.environment), PPOHedgingStrategy(model, normalizer=normalizer), seed
        )
        assert batch_errors[i] == pytest.approx(error[-1], abs=1e-5)
    np.testing.assert_array_equal(original_statistics, normalizer.obs_rms.mean)
    before = normalizer.obs_rms.mean.copy()
    result = evaluate(config, model_path, tmp_path / "evaluation", matrix=False)
    assert "ppo" in result["scenarios"]["sigma0.2_cost0.001"]["metrics"]
    np.testing.assert_array_equal(before, normalizer.obs_rms.mean)
    normalizer.close()
    manifest_path = model_path.with_suffix(".json")
    bad = json.loads(manifest_path.read_text())
    bad["contract_version"] = "legacy"
    manifest_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Incompatible"):
        load_policy(model_path, config.environment)


def test_legacy_checkpoint_rejected_before_loading(tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        load_policy(tmp_path / "old.zip", HedgingEnvParams())


def test_overlapping_test_seeds_rejected(tmp_path):
    config = ExperimentConfig(evaluation=EvaluationConfig(episodes=2, seed_start=100000))
    with pytest.raises(ValueError, match="overlaps"):
        evaluate(config, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_bundle_tampering_rejected(tmp_path):
    from quantlab.backtesting.accounting import CONTRACT_VERSION

    model = tmp_path / "model.zip"
    model.write_bytes(b"changed model")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "normalization_file": "model.pkl",
                "model_sha256": "wrong hash",
            }
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_policy(model, HedgingEnvParams())


def test_seeded_training_is_repeatable(tmp_path):
    config = ExperimentConfig(
        seed=321,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        environment=HedgingEnvParams(steps=4),
        ppo=replace(PPOConfig(), total_timesteps=8, n_steps=8, batch_size=4),
    )
    first, second = train(config), train(config)
    assert first["model_path"] != second["model_path"]
    assert first["validation"] == second["validation"]
    a, na, _ = load_policy(first["model_path"], config.environment)
    b, nb, _ = load_policy(second["model_path"], config.environment)
    np.testing.assert_array_equal(na.obs_rms.mean, nb.obs_rms.mean)
    np.testing.assert_array_equal(na.obs_rms.var, nb.obs_rms.var)
    for key, value in a.policy.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), b.policy.state_dict()[key].numpy())
    na.close()
    nb.close()
