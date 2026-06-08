from src.black_scholes import call_price, put_price, delta

# Known values you can verify against any online B-S calculator
# S=100, K=100, T=1 year, r=5%, sigma=20%
# Call should be ~10.45, Put ~5.57

print(call_price(100, 100, 1, 0.05, 0.20))   # expect ~10.45
print(put_price(100, 100, 1, 0.05, 0.20))    # expect ~5.57
print(delta(100, 100, 1, 0.05, 0.20))        # expect ~0.637

# Sanity check: put-call parity
# call - put should equal S - K*e^(-rT)
import numpy as np
c = call_price(100, 100, 1, 0.05, 0.20)
p = put_price(100, 100, 1, 0.05, 0.20)
print(f"Put-call parity check: {c - p:.4f} vs {100 - 100*np.exp(-0.05):.4f}")
# These two numbers must match


from src.black_scholes import gamma, theta, vega
# S=100, K=100, T=1, r=0.05, sigma=0.20
print(gamma(100, 100, 1, 0.05, 0.20))   # expect ~0.0188
print(theta(100, 100, 1, 0.05, 0.20))   # expect ~-0.0152 (negative, option loses value daily)
print(vega(100, 100, 1, 0.05, 0.20))    # expect ~0.3753