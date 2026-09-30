"""Versioned Gymnasium environment backed by the shared portfolio ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quantlab.backtesting.accounting import CONTRACT_VERSION, Portfolio, execute_step, observation
from quantlab.pricing.black_scholes import call_features
from quantlab.simulators.gbm import simulate_gbm
from quantlab.simulators.heston import simulate_heston


@dataclass(frozen=True)
class HedgingEnvParams:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    steps: int = 252
    cost_rate: float = 0.001
    use_heston: bool = False
    lambda_risk: float = 0.1  # Legacy config field; v2 reward does not use it.
    heston_rho: float = -0.7
    heston_kappa: float = 2.0
    heston_theta: float | None = None
    heston_vol_of_vol: float = 0.5

    def __post_init__(self):
        numeric = [
            self.S0,
            self.K,
            self.r,
            self.sigma,
            self.T,
            self.cost_rate,
            self.lambda_risk,
            self.heston_rho,
            self.heston_kappa,
            self.heston_vol_of_vol,
        ]
        if self.heston_theta is not None:
            numeric.append(self.heston_theta)
        if not np.isfinite(numeric).all():
            raise ValueError("Environment parameters must be finite.")
        if (
            self.S0 <= 0
            or self.K <= 0
            or self.T <= 0
            or self.sigma < 0
            or self.cost_rate < 0
            or self.lambda_risk < 0
            or not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or self.steps <= 0
            or abs(self.heston_rho) > 1
            or self.heston_kappa < 0
            or self.heston_vol_of_vol < 0
            or (self.heston_theta is not None and self.heston_theta < 0)
        ):
            raise ValueError("Invalid environment parameters.")


class HedgingEnv(gym.Env):
    metadata = {"render_modes": []}
    contract_version = CONTRACT_VERSION

    def __init__(self, params: HedgingEnvParams | None = None, **kwargs: Any):
        super().__init__()
        if params is not None and kwargs:
            raise ValueError("Pass params or keyword parameters, not both.")
        self.params = params or HedgingEnvParams(**kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self._ready = False
        self.vars = None

    @property
    def hedge_position(self):
        return self.state.position

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p = self.params
        if options and "prices" in options:
            path = np.asarray(options["prices"], dtype=float)
            if path.shape != (p.steps + 1,) or not np.isfinite(path).all() or np.any(path <= 0):
                raise ValueError("Injected path must have steps+1 finite positive prices.")
            if not np.isclose(path[0], p.S0):
                raise ValueError("Injected path must start at S0.")
            self.prices, self.vars = path.copy(), None
        elif p.use_heston:
            self.prices, self.vars = simulate_heston(
                S0=p.S0,
                v0=p.sigma**2,
                rho=p.heston_rho,
                kappa=p.heston_kappa,
                theta=p.sigma**2 if p.heston_theta is None else p.heston_theta,
                sigma=p.heston_vol_of_vol,
                T=p.T,
                steps=p.steps,
                rng=self.np_random,
            )
        else:
            self.prices = simulate_gbm(p.S0, p.r, p.sigma, p.T, p.steps, rng=self.np_random)
            self.vars = None
        self.times = p.T * (1 - np.arange(p.steps + 1) / p.steps)
        # Heston remains a documented fixed-volatility Black-Scholes valuation stress test.
        self.features = call_features(self.prices, p.K, self.times, p.r, p.sigma)
        self.t = 0
        self.scale = max(float(self.features[0, 0]), 1e-8)
        self.state = Portfolio(cash=float(self.features[0, 0]))
        self.last = {
            "pnl": 0.0,
            "transaction_cost": 0.0,
            "executed_action": 0.0,
            "turnover": 0.0,
            "portfolio_value": self.state.cash,
        }
        self._ready = True
        return self._get_obs(), self._get_info()

    def step(self, action):
        if not self._ready or self.t >= self.params.steps:
            raise RuntimeError("Call reset before stepping a new episode.")
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Action must have shape (1,) and be finite.")
        p = self.params
        previous_error = self.state.error / self.scale
        self.last = execute_step(
            self.state,
            float(action[0]),
            self.prices[self.t],
            self.prices[self.t + 1],
            float(self.features[self.t + 1, 0]),
            p.T / p.steps,
            p.r,
            p.cost_rate,
            terminal=self.t + 1 == p.steps,
        )
        self.t += 1
        reward = previous_error**2 - (self.state.error / self.scale) ** 2
        return self._get_obs(), float(reward), self.t == p.steps, False, self._get_info()

    def _get_obs(self):
        p = self.params
        return observation(
            self.prices[self.t], p.K, self.times[self.t], p.T, p.sigma, self.features[self.t], self.state, self.scale
        )

    def _get_info(self):
        return self.last | {
            "t": self.t,
            "price": float(self.prices[self.t]),
            "option_value": float(self.features[self.t, 0]),
            "hedge_position": self.state.position,
            "cash_balance": self.state.cash,
            "hedging_error": self.state.error,
            "contract_version": CONTRACT_VERSION,
        }
