"""PPO hedging strategy wrapper."""

from __future__ import annotations

import numpy as np

from quantlab.strategies.base import Strategy


class PPOHedgingStrategy(Strategy):
    """Strategy adapter for a Stable-Baselines3 PPO model."""

    name = "ppo"

    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def action(self, observation: np.ndarray, info: dict) -> float:
        del info
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return float(np.asarray(action).reshape(-1)[0])
