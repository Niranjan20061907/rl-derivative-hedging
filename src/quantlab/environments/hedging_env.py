"""Gymnasium environment for discrete option hedging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quantlab.pricing.black_scholes import bs_call_price, bs_delta, bs_gamma
from quantlab.simulators.gbm import simulate_gbm
from quantlab.simulators.heston import simulate_heston


@dataclass(frozen=True)
class HedgingEnvParams:
    """Parameters controlling the hedging environment."""

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    steps: int = 252
    cost_rate: float = 0.001
    use_heston: bool = False
    lambda_risk: float = 0.1
    heston_rho: float = -0.7
    heston_kappa: float = 2.0
    heston_theta: float | None = None
    heston_vol_of_vol: float = 0.5


class HedgingEnv(gym.Env):
    """RL environment for hedging a European call option.

    Observation: ``[log_moneyness, delta, gamma, remaining_time_scaled, sigma, hedge_position]``.
    Action: change in hedge position, clipped by the action space to ``[-1, 1]``.
    Reward: current prototype reward ``pnl - lambda_risk * pnl**2``.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: HedgingEnvParams | None = None, **kwargs: Any) -> None:
        super().__init__()
        if params is None:
            params = HedgingEnvParams(**kwargs)
        elif kwargs:
            raise ValueError("Pass either params or keyword parameters, not both.")
        self.params = params

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.prices: np.ndarray
        self.vars: np.ndarray | None = None
        self.t = 0
        self.hedge_position = 0.0
        self.option_value = 0.0
        self.last_pnl = 0.0
        self.last_cost = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        if self.params.use_heston:
            theta = self.params.heston_theta if self.params.heston_theta is not None else self.params.sigma**2
            self.prices, self.vars = simulate_heston(
                S0=self.params.S0,
                v0=self.params.sigma**2,
                rho=self.params.heston_rho,
                kappa=self.params.heston_kappa,
                theta=theta,
                sigma=self.params.heston_vol_of_vol,
                T=self.params.T,
                steps=self.params.steps,
                rng=self.np_random,
            )
        else:
            self.prices = simulate_gbm(
                S0=self.params.S0,
                mu=self.params.r,
                sigma=self.params.sigma,
                T=self.params.T,
                steps=self.params.steps,
                rng=self.np_random,
            )
            self.vars = None

        self.t = 0
        self.hedge_position = 0.0
        self.last_pnl = 0.0
        self.last_cost = 0.0
        self.option_value = bs_call_price(
            self.prices[self.t],
            self.params.K,
            self.params.T,
            self.params.r,
            self.params.sigma,
        )

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        clipped_action = np.clip(action_array, self.action_space.low, self.action_space.high)
        hedge_change = float(clipped_action.reshape(-1)[0])

        S = float(self.prices[self.t])
        cost = self.params.cost_rate * abs(hedge_change) * S
        self.hedge_position += hedge_change

        self.t += 1
        terminated = self.t == self.params.steps
        truncated = False

        new_S = float(self.prices[self.t])
        remaining_T = self.params.T * (1 - self.t / self.params.steps)
        new_option_value = bs_call_price(new_S, self.params.K, remaining_T, self.params.r, self.params.sigma)

        pnl = new_option_value - self.option_value - self.hedge_position * (new_S - S) - cost
        reward = pnl - self.params.lambda_risk * (pnl**2)

        self.option_value = new_option_value
        self.last_pnl = float(pnl)
        self.last_cost = float(cost)

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        S = float(self.prices[self.t])
        remaining_T = self.params.T * (1 - self.t / self.params.steps)
        delta = bs_delta(S, self.params.K, remaining_T, self.params.r, self.params.sigma)
        gamma = bs_gamma(S, self.params.K, remaining_T, self.params.r, self.params.sigma)
        log_moneyness = np.log(S / self.params.K)
        time_scaled = remaining_T / self.params.T
        return np.array(
            [log_moneyness, delta, gamma, time_scaled, self.params.sigma, self.hedge_position],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "price": float(self.prices[self.t]),
            "option_value": float(self.option_value),
            "hedge_position": float(self.hedge_position),
            "pnl": float(self.last_pnl),
            "transaction_cost": float(self.last_cost),
        }
