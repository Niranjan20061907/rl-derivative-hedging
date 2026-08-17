import numpy as np


def simulate_gbm(S0, mu, sigma, T, steps, seed=None):
    """
    Simulate a single GBM price path.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    prices = np.zeros(steps + 1)
    prices[0] = S0

    for t in range(1, steps + 1):
        z = np.random.normal()
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return prices
