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
