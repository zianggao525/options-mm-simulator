import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import simulate_gbm, simulate_order_flow

# --- GBM sanity checks ---
np.random.seed(42)
S0, mu, sigma, dt, n_steps = 100, 0.05, 0.20, 1/252, 252

paths = simulate_gbm(S0, mu, sigma, dt, n_steps, n_paths=1000)

final_prices = paths[-1]
expected_mean = S0 * np.exp(mu * n_steps * dt)
actual_mean   = final_prices.mean()
diff          = abs(actual_mean - expected_mean)

status = "PASS" if diff < 2.0 else "FAIL"
print(f"{status}: GBM final price mean")
print(f"       expected={expected_mean:.4f}, got={actual_mean:.4f}, diff={diff:.4f}")

always_positive = (paths > 0).all()
status = "PASS" if always_positive else "FAIL"
print(f"{status}: GBM prices always positive")

# --- Order flow sanity checks (Poisson model) ---
lambda_buy, lambda_sell = 3, 3
orders = simulate_order_flow(10000, lambda_buy=lambda_buy, lambda_sell=lambda_sell)

mean_flow     = orders.mean()
std_flow      = orders.std()
expected_mean = lambda_buy - lambda_sell          # should be 0
expected_std  = np.sqrt(lambda_buy + lambda_sell) # variance of difference of two Poissons

status = "PASS" if abs(mean_flow - expected_mean) < 0.1 else "FAIL"
print(f"{status}: Net order flow mean ~{expected_mean}")
print(f"       expected={expected_mean:.3f}, got={mean_flow:.3f}")

status = "PASS" if abs(std_flow - expected_std) < 0.1 else "FAIL"
print(f"{status}: Net order flow std ~{expected_std:.3f}")
print(f"       expected={expected_std:.3f}, got={std_flow:.3f}")

buy_steps  = (orders > 0).mean()
sell_steps = (orders < 0).mean()
zero_steps = (orders == 0).mean()
balance    = abs(buy_steps - sell_steps)

status = "PASS" if balance < 0.02 else "FAIL"
print(f"{status}: Buy/sell steps roughly balanced")
print(f"       buy={buy_steps:.3f}, sell={sell_steps:.3f}, imbalance={balance:.3f}")

status = "PASS" if zero_steps > 0 else "FAIL"
print(f"{status}: Some zero net-flow steps exist")
print(f"       zero_steps={zero_steps:.3f}")

# --- Plot 10 sample paths ---
sample_paths = simulate_gbm(S0, mu, sigma, dt, n_steps, n_paths=10)
plt.figure(figsize=(10, 5))
plt.plot(sample_paths)
plt.title("GBM Simulated Stock Paths (10 paths, 1 year)")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price")
plt.axhline(S0, color='black', linestyle='--', linewidth=0.8, label='S0=100')
plt.legend()
plt.tight_layout()
plt.savefig("tests/gbm_paths.png", dpi=150)
print("\nPASS: Plot saved to tests/gbm_paths.png")