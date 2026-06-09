import numpy as np


def simulate_gbm(S0, mu, sigma, dt, n_steps, n_paths=1):
    """
    Simulate stock price paths using Geometric Brownian Motion.

    S0      : initial stock price
    mu      : drift (annualized, e.g. 0.05 = 5%)
    sigma   : volatility (annualized, e.g. 0.20 = 20%)
    dt      : time step in years (e.g. 1/252 = one trading day)
    n_steps : number of time steps
    n_paths : number of independent paths to simulate

    Returns: array of shape (n_steps + 1, n_paths)
    """
    prices = np.zeros((n_steps + 1, n_paths))
    prices[0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return prices


def simulate_order_flow(n_steps, lambda_buy=3, lambda_sell=3):
    """
    Simulate order arrivals using Poisson processes.
    
    lambda_buy  : average buy orders per time step
    lambda_sell : average sell orders per time step
    
    Returns: array of shape (n_steps,)
        Positive = net buy orders hitting MM (MM sells)
        Negative = net sell orders hitting MM (MM buys)
    """
    buys  = np.random.poisson(lambda_buy,  n_steps)
    sells = np.random.poisson(lambda_sell, n_steps)
    return buys - sells  # net order flow per step