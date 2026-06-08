import numpy as np
from scipy.stats import norm

def call_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European call option.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years (e.g. 0.5 for 6 months)
    r: risk-free interest rate
    sigma: volatility of the underlying asset
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



def put_price(S, K, T, r, sigma):
    """Black-Scholes price for a European put option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


"""Greeks"""
def delta(S, K, T, r, sigma, option_type = "call"):
    """
    Delta of a European call or put option.
    
    Delta: Sensitivity of the option's price to changes in the underlying asset's price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)


def gamma(S, K, T, r, sigma):
    """Gamma: rate of change of delta. Same for calls and puts."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type='call'):
    """Theta: daily time decay (divided by 365 to get per-day value)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def vega(S, K, T, r, sigma):
    """Vega: sensitivity to 1% change in volatility."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100
