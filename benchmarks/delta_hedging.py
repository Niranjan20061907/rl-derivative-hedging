import numpy as np
from pricing.black_scholes import bs_call_price, bs_delta


def delta_hedge(prices, K, r, sigma, T):
    """
    Perform discrete-time delta hedging for a short call option.
    """
    n_steps = len(prices) - 1
    dt = T / n_steps

    option_values = np.zeros(n_steps + 1)
    deltas = np.zeros(n_steps + 1)
    portfolio = np.zeros(n_steps + 1)

    # Initial values
    option_values[0] = bs_call_price(prices[0], K, T, r, sigma)
    deltas[0] = bs_delta(prices[0], K, T, r, sigma)
    portfolio[0] = option_values[0] - deltas[0] * prices[0]

    for t in range(1, n_steps + 1):
        remaining_T = T - t * dt

        option_values[t] = bs_call_price(prices[t], K, remaining_T, r, sigma)

        deltas[t] = bs_delta(prices[t], K, remaining_T, r, sigma)

        portfolio[t] = option_values[t] - deltas[t] * prices[t]

    hedging_error = portfolio - portfolio[0]

    return {
        "option_values": option_values,
        "deltas": deltas,
        "portfolio": portfolio,
        "hedging_error": hedging_error,
    }
