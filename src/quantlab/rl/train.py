"""CPU PPO training with bounded logging and validation-selected, versioned bundles."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quantlab.backtesting.accounting import CONTRACT_VERSION
from quantlab.environments.hedging_env import HedgingEnv
from quantlab.rl.artifacts import provenance, run_id, sha256
from quantlab.rl.config import ExperimentConfig, load_config, save_config


class RewardLoggingCallback:
    """Factory kept for compatibility; rows are flushed in batches of at most 256."""

    def __init__(self, csv_path: Path, config=None, checkpoint=None, normalizer=None):
        from stable_baselines3.common.callbacks import BaseCallback

        class Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.best_rmse = float("inf")
                self.history = []
                self.next_validation = config.ppo.validation_frequency if config else None

            def _on_training_start(self):
                self.stream = csv_path.open("w", newline="")
                self.writer = csv.DictWriter(self.stream, fieldnames=["timesteps", "reward"])
                self.writer.writeheader()

            def flush(self):
                self.writer.writerows(self.rows)
                self.stream.flush()
                self.rows.clear()

            def validate(self):
                from quantlab.rl.evaluate import validation_rmse

                score = validation_rmse(self.model, normalizer, config.environment, range(100000, 100200))
                self.history.append({"timesteps": self.num_timesteps, "terminal_rmse": score})
                if score < self.best_rmse:
                    self.best_rmse = score
                    self.model.save(checkpoint)
                    normalizer.save(str(checkpoint.with_suffix(".pkl")))
                (csv_path.parent / "validation.json").write_text(json.dumps(self.history, indent=2))
                print(f"seed={config.seed} step={self.num_timesteps} validation_rmse={score:.4f}", flush=True)

            def _on_step(self):
                self.rows.append({"timesteps": self.num_timesteps, "reward": float(np.mean(self.locals["rewards"]))})
                if len(self.rows) >= 256:
                    self.flush()
                if config and self.num_timesteps >= self.next_validation:
                    self.validate()
                    self.next_validation += config.ppo.validation_frequency
                return True

            def _on_training_end(self):
                self.flush()
                self.stream.close()
                if config:
                    self.validate()

        self.callback = Callback()


def set_global_seeds(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_model(config, env):
    from stable_baselines3 import PPO

    p = config.ppo
    return PPO(
        p.policy,
        env,
        learning_rate=p.learning_rate,
        batch_size=p.batch_size,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        n_steps=p.n_steps,
        verbose=p.verbose,
        seed=config.seed,
        device="cpu",
    )


def train(config: ExperimentConfig) -> dict:
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if config.environment.use_heston:
        raise ValueError("Validated v2 training uses GBM; Heston is deferred.")
    if config.seed in range(100000, 100200) or config.seed in range(200000, 201000):
        raise ValueError("Training seed overlaps held-out ranges.")
    set_global_seeds(config.seed)
    identifier = run_id(config.seed)
    run_dir = Path(config.results_dir) / f"run_{identifier}"
    checkpoint_dir = Path(config.checkpoint_dir) / f"{config.experiment_name}_{identifier}"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model_path = checkpoint_dir / "best.zip"
    env = HedgingEnv(config.environment)
    check_env(env, warn=True)
    wrapped = DummyVecEnv([lambda: env])
    wrapped.seed(config.seed)
    normalizer = VecNormalize(wrapped, norm_obs=True, norm_reward=False)
    model = build_model(config, normalizer)
    logger = RewardLoggingCallback(run_dir / "training_log.csv", config, model_path, normalizer).callback
    save_config(config, run_dir / "config.yaml")
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=config.ppo.total_timesteps, callback=logger)
    finally:
        if hasattr(logger, "stream") and not logger.stream.closed:
            logger.flush()
            logger.stream.close()
    elapsed = time.perf_counter() - start
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "model_sha256": sha256(model_path),
        "normalization_file": "best.pkl",
        "normalization_sha256": sha256(model_path.with_suffix(".pkl")),
        "seed": config.seed,
        "environment": asdict(config.environment),
        "best_validation_rmse": logger.best_rmse,
        "validation_seed_start": 100000,
        "validation_episodes": 200,
        "run_id": identifier,
    }
    model_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    metadata = {
        "run_id": identifier,
        "seed": config.seed,
        "total_timesteps": config.ppo.total_timesteps,
        "actual_timesteps": model.num_timesteps,
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "training_time_seconds": elapsed,
        "provenance": provenance(),
        "config": config.to_dict(),
        "validation": logger.history,
        "manifest": manifest,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    normalizer.close()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        from dataclasses import replace

        config = replace(config, seed=args.seed)
    print(json.dumps(train(config), indent=2))


if __name__ == "__main__":
    main()
