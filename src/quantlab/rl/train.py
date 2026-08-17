"""PPO training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantlab import __version__
from quantlab.environments.hedging_env import HedgingEnv
from quantlab.rl.config import ExperimentConfig, load_config, save_config


class RewardLoggingCallback:
    """Small SB3 callback factory that writes per-step rewards to CSV."""

    def __init__(self, csv_path: Path):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, output_path: Path):
                super().__init__()
                self.output_path = output_path
                self.rows: list[dict[str, float]] = []

            def _on_step(self) -> bool:
                rewards = self.locals.get("rewards")
                if rewards is not None:
                    self.rows.append({"timesteps": self.num_timesteps, "reward": float(np.mean(rewards))})
                return True

            def _on_training_end(self) -> None:
                with self.output_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["timesteps", "reward"])
                    writer.writeheader()
                    writer.writerows(self.rows)

        self.callback = _Callback(csv_path)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def build_model(config: ExperimentConfig, env: HedgingEnv):
    from stable_baselines3 import PPO

    ppo = config.ppo
    return PPO(
        policy=ppo.policy,
        env=env,
        learning_rate=ppo.learning_rate,
        batch_size=ppo.batch_size,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        n_steps=ppo.n_steps,
        verbose=ppo.verbose,
        seed=config.seed,
    )


def train(config: ExperimentConfig) -> dict:
    from stable_baselines3.common.env_checker import check_env

    set_global_seeds(config.seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config.results_dir) / f"run_{timestamp}_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = HedgingEnv(config.environment)
    check_env(env, warn=True)
    env.reset(seed=config.seed)

    model = build_model(config, env)
    training_log = run_dir / "training_log.csv"
    callback = RewardLoggingCallback(training_log).callback

    start = time.perf_counter()
    model.learn(total_timesteps=config.ppo.total_timesteps, callback=callback)
    training_time_seconds = time.perf_counter() - start
    actual_timesteps = int(model.num_timesteps)

    model_path = checkpoint_dir / f"{config.experiment_name}_seed{config.seed}.zip"
    model.save(model_path)
    save_config(config, run_dir / "config.yaml")
    metadata = {
        "timestamp_utc": timestamp,
        "seed": config.seed,
        "total_timesteps": config.ppo.total_timesteps,
        "actual_timesteps": actual_timesteps,
        "environment": asdict(config.environment),
        "ppo": asdict(config.ppo),
        "model_path": str(model_path),
        "training_time_seconds": training_time_seconds,
        "software": {
            "quantlab_version": __version__,
            "python": platform.python_version(),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({"training_time_seconds": training_time_seconds}, indent=2))
    return metadata | {"run_dir": str(run_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO hedging policy.")
    parser.add_argument("--config", default="configs/rl.yaml", help="Path to YAML experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = train(load_config(args.config))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
