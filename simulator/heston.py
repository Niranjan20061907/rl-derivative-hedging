import numpy as np


def simulate_heston(
    S0,
    v0,
    rho,
    kappa,
    theta,
    sigma,
    T,
    steps,
    seed=None,
):
    """_summary_

    Args:
        S0 (): Initial Stock Price
        v0 (): Initial Variance (Volatility²)
        --> Volatility = √variance

        rho (): Correlation Between Price & Volatility

        kappa (): Mean Reversion Speed of Volatility
        How fast volatility pulls back to its long-term average.
                •	Large kappa → volatility snaps back quickly
                •	Small kappa → volatility stays high/low for long

        theta (): Long-Run Average Variance

        sigma (): Volatility of Volatility (Vol-of-Vol)
        T (): Total Time Horizon
        steps (): Number of Time Steps
        seed (): Random Seed (Reproducibility).
                •   Fixes randomness so results can be repeated.

    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    prices = np.zeros(steps + 1)
    vars = np.zeros(steps + 1)

    prices[0] = S0
    vars[0] = v0

    for t in range(1, steps + 1):
        z1 = np.random.normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal()

        v_prev = vars[t - 1]

        dv = (
            kappa * (theta - v_prev) * dt
            + sigma * np.sqrt(max(v_prev, 0)) * np.sqrt(dt) * z2
        )

        vars[t] = max(v_prev + dv, 0)

        dS = prices[t - 1] * np.sqrt(v_prev) * np.sqrt(dt) * z1
        prices[t] = prices[t - 1] + dS

    return prices, vars
