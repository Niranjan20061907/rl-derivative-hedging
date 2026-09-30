"""Configuration loading for QuantLab RL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quantlab.environments.hedging_env import HedgingEnvParams


@dataclass(frozen=True)
class PPOConfig:
    policy: str = "MlpPolicy"
    total_timesteps: int = 100_000
    learning_rate: float = 0.0003
    batch_size: int = 64
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 256
    verbose: int = 0
    validation_frequency: int = 25_000

    def __post_init__(self):
        counts = [self.total_timesteps, self.n_steps, self.batch_size, self.validation_frequency]
        if any(not isinstance(x, int) or isinstance(x, bool) for x in counts):
            raise ValueError("PPO step counts must be integers.")
        if not np.isfinite([self.learning_rate, self.gamma, self.gae_lambda, self.clip_range]).all():
            raise ValueError("PPO parameters must be finite.")
        if (
            self.total_timesteps <= 0
            or self.n_steps < 2
            or self.batch_size < 2
            or self.validation_frequency <= 0
            or self.learning_rate <= 0
            or self.gamma != 1.0
            or not 0 < self.gae_lambda <= 1
            or not 0 < self.clip_range < 1
        ):
            raise ValueError("Invalid PPO parameters; telescoping reward requires gamma=1.")


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 1000
    deterministic: bool = True
    include_no_hedge: bool = True
    seed_start: int = 200_000
    chunk_size: int = 250
    bootstrap_samples: int = 2000

    def __post_init__(self):
        counts = [self.episodes, self.chunk_size, self.bootstrap_samples, self.seed_start]
        if any(not isinstance(x, int) or isinstance(x, bool) for x in counts):
            raise ValueError("Evaluation counts and seeds must be integers.")
        if min(self.episodes, self.chunk_size, self.bootstrap_samples) <= 0 or self.seed_start < 0:
            raise ValueError("Invalid evaluation parameters.")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 11
    experiment_name: str = "ppo_hedging"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    environment: HedgingEnvParams = field(default_factory=HedgingEnvParams)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not self.experiment_name or not self.checkpoint_dir or not self.results_dir:
            raise ValueError("Run names and directories cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    env = HedgingEnvParams(**data.get("environment", {}))
    ppo = PPOConfig(**data.get("ppo", {}))
    evaluation = EvaluationConfig(**data.get("evaluation", {}))
    top_level = {k: v for k, v in data.items() if k not in {"environment", "ppo", "evaluation"}}
    return ExperimentConfig(environment=env, ppo=ppo, evaluation=evaluation, **top_level)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save an experiment config to YAML."""
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
