import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.black_scholes import call_price, put_price, delta, gamma, theta, vega

TOLERANCE = 0.01  # acceptable difference from expected value


def check(label, actual, expected, tol=TOLERANCE):
    diff = abs(actual - expected)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"{status}: {label}")
    print(f"       expected={expected:.4f}, got={actual:.4f}, diff={diff:.4f}")
    return status == "PASS"


def check_parity(label, lhs, rhs, tol=TOLERANCE):
    diff = abs(lhs - rhs)
    status = "PASS" if diff <= tol else "FAIL"
    print(f"{status}: {label}")
    print(f"       lhs={lhs:.4f}, rhs={rhs:.4f}, diff={diff:.4f}")
    return status == "PASS"


# --- Test parameters ---
# S=100, K=100, T=1yr, r=5%, sigma=20%
# Reference: https://www.tradingblock.com/calculators/option-greeks-calculator
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20

print("=" * 50)
print("Black-Scholes Pricer Tests")
print("=" * 50)

results = []

# Pricing
results.append(check("Call price (ATM)",  call_price(S, K, T, r, sigma), 10.45))
results.append(check("Put price (ATM)",   put_price(S, K, T, r, sigma),   5.57))

# Put-call parity: call - put = S - K*e^(-rT)
c = call_price(S, K, T, r, sigma)
p = put_price(S, K, T, r, sigma)
results.append(check_parity("Put-call parity", c - p, S - K * np.exp(-r * T)))

# Greeks
results.append(check("Delta (call)",  delta(S, K, T, r, sigma, option_type='call'),  0.637))
results.append(check("Delta (put)",   delta(S, K, T, r, sigma, option_type='put'),  -0.363))
results.append(check("Gamma",         gamma(S, K, T, r, sigma),                      0.0188))
results.append(check("Theta (call)",  theta(S, K, T, r, sigma, option_type='call'), -0.018))
results.append(check("Vega",          vega(S, K, T, r, sigma),                       0.3753))

# Delta bounds: call delta in (0,1), put delta in (-1,0)
call_d = delta(S, K, T, r, sigma, option_type='call')
put_d  = delta(S, K, T, r, sigma, option_type='put')
results.append(check_parity("Call delta in (0,1)",  float(0 < call_d < 1), 1.0, tol=0))
results.append(check_parity("Put delta in (-1,0)",  float(-1 < put_d < 0), 1.0, tol=0))

# Theta should be negative (options lose value over time)
results.append(check_parity("Theta is negative",    float(theta(S, K, T, r, sigma) < 0), 1.0, tol=0))

# Gamma should be positive
results.append(check_parity("Gamma is positive",    float(gamma(S, K, T, r, sigma) > 0), 1.0, tol=0))

print("=" * 50)
passed = sum(results)
total  = len(results)
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("All tests passed.")
else:
    print(f"{total - passed} test(s) failed — review output above.")
print("=" * 50)