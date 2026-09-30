"""PPO hedging strategy wrapper."""

from __future__ import annotations

import numpy as np

from quantlab.strategies.base import Strategy


class PPOHedgingStrategy(Strategy):
    """Strategy adapter for a Stable-Baselines3 PPO model."""

    name = "ppo"

    def __init__(self, model, deterministic: bool = True, normalizer=None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.deterministic = deterministic

    def action(self, observation: np.ndarray, info: dict) -> float:
        del info
        if self.normalizer is not None:
            observation = self.normalizer.normalize_obs(observation)
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return float(np.asarray(action).reshape(-1)[0])
