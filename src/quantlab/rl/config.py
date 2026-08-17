"""Configuration loading for QuantLab RL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantlab.environments.hedging_env import HedgingEnvParams


@dataclass(frozen=True)
class PPOConfig:
    policy: str = "MlpPolicy"
    total_timesteps: int = 10_000
    learning_rate: float = 0.0003
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 256
    verbose: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 10
    deterministic: bool = True
    include_no_hedge: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    experiment_name: str = "ppo_hedging"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    environment: HedgingEnvParams = field(default_factory=HedgingEnvParams)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    env = HedgingEnvParams(**data.get("environment", {}))
    ppo = PPOConfig(**data.get("ppo", {}))
    evaluation = EvaluationConfig(**data.get("evaluation", {}))
    top_level = {k: v for k, v in data.items() if k not in {"environment", "ppo", "evaluation"}}
    return ExperimentConfig(environment=env, ppo=ppo, evaluation=evaluation, **top_level)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save an experiment config to YAML."""
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
