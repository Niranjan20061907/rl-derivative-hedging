import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.heston import simulate_heston
from simulator.gbm import simulate_gbm
from pricing.black_scholes import bs_call_price, bs_delta, bs_gamma


class HedgingEnv(gym.Env):
    """
    RL environment for option hedging.
    """

    def __init__(
        self,
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        steps=252,
        cost_rate=0.001,
        use_heston=False,
    ):
        super().__init__()

        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.cost_rate = cost_rate
        self.use_heston = use_heston

        # Action: change in hedge position
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [S, delta, gamma, T, sigma, hedge_position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.use_heston:
            self.prices, self.vars = simulate_heston(
                S0=self.S0,
                v0=self.sigma**2,
                rho=-0.7,
                kappa=2.0,
                theta=self.sigma**2,
                sigma=0.5,
                T=self.T,
                steps=self.steps,
            )
        else:
            self.prices = simulate_gbm(
                S0=self.S0, mu=self.r, sigma=self.sigma, T=self.T, steps=self.steps
            )

        self.t = 0
        self.hedge_position = 0.0

        self.option_value = bs_call_price(
            self.prices[self.t], self.K, self.T, self.r, self.sigma
        )

        return self._get_obs(), {}

    # def _get_obs(self):
    #     S = self.prices[self.t]
    #     remaining_T = self.T * (1 - self.t / self.steps)

    #     delta = bs_delta(S, self.K, remaining_T, self.r, self.sigma)
    #     gamma = bs_gamma(S, self.K, remaining_T, self.r, self.sigma)

    #     return np.array(
    #         [S, delta, gamma, remaining_T, self.sigma, self.hedge_position],
    #         dtype=np.float32,
    #     )

    def _get_obs(self):
        S = self.prices[self.t]
        remaining_T = self.T * (1 - self.t / self.steps)

        delta = bs_delta(S, self.K, remaining_T, self.r, self.sigma)
        gamma = bs_gamma(S, self.K, remaining_T, self.r, self.sigma)

        log_moneyness = np.log(S / self.K)
        time_scaled = remaining_T / self.T

        return np.array(
            [
                log_moneyness,
                delta,
                gamma,
                time_scaled,
                self.sigma,
                self.hedge_position,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        action = float(action[0])

        S = self.prices[self.t]

        # Transaction cost
        cost = self.cost_rate * abs(action) * S

        # Update hedge
        self.hedge_position += action

        # Move forward in time
        self.t += 1
        done = self.t == self.steps

        new_S = self.prices[self.t]
        remaining_T = self.T * (1 - self.t / self.steps)

        new_option_value = bs_call_price(new_S, self.K, remaining_T, self.r, self.sigma)

        # Hedged portfolio P&L
        pnl = (
            new_option_value
            - self.option_value
            - self.hedge_position * (new_S - S)
            - cost
        )

        self.option_value = new_option_value

        # Reward: penalize variance + cost
        reward = -(pnl**2)

        return self._get_obs(), reward, done, False, {}
