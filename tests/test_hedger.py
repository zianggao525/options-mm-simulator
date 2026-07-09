import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_maker import MarketMaker
from src.hedger import rebalance, net_delta

S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    if detail:
        print(f"       {detail}")
    return condition

print("=" * 50)
print("Hedger Tests")
print("=" * 50)

results = []

# Flat inventory needs no hedge
mm_flat = MarketMaker(S, K, T, r, sigma)
trade_flat = rebalance(mm_flat, S, T)
results.append(check("Flat inventory requires no hedge trade",
    trade_flat == 0 and mm_flat.hedge_pos == 0,
    f"trade={trade_flat}, hedge_pos={mm_flat.hedge_pos}"))

# Long inventory (positive delta) should be hedged with a short stock position
mm_long = MarketMaker(S, K, T, r, sigma)
mm_long.inventory = 10
rebalance(mm_long, S, T)
results.append(check("Long inventory hedged with short stock position",
    mm_long.hedge_pos < 0,
    f"inventory={mm_long.inventory}, hedge_pos={mm_long.hedge_pos:.4f}"))

# Short inventory (negative delta) should be hedged with a long stock position
mm_short = MarketMaker(S, K, T, r, sigma)
mm_short.inventory = -10
rebalance(mm_short, S, T)
results.append(check("Short inventory hedged with long stock position",
    mm_short.hedge_pos > 0,
    f"inventory={mm_short.inventory}, hedge_pos={mm_short.hedge_pos:.4f}"))

# Net delta should be ~0 immediately after rebalancing
results.append(check("Net delta is ~0 after rebalance (long case)",
    abs(net_delta(mm_long, S, T)) < 1e-8,
    f"net_delta={net_delta(mm_long, S, T):.8f}"))

results.append(check("Net delta is ~0 after rebalance (short case)",
    abs(net_delta(mm_short, S, T)) < 1e-8,
    f"net_delta={net_delta(mm_short, S, T):.8f}"))

# Re-rebalancing after inventory changes should only trade the incremental delta
mm_step = MarketMaker(S, K, T, r, sigma)
mm_step.inventory = 5
first_trade = rebalance(mm_step, S, T)
mm_step.inventory = 8
second_trade = rebalance(mm_step, S, T)
results.append(check("Second rebalance only trades incremental delta",
    second_trade != first_trade and abs(net_delta(mm_step, S, T)) < 1e-8,
    f"first_trade={first_trade:.4f}, second_trade={second_trade:.4f}"))

# Buying shares to hedge should cost cash
mm_cash = MarketMaker(S, K, T, r, sigma)
mm_cash.inventory = -10  # short inventory -> hedge buys stock -> cash decreases
cash_before = mm_cash.cash
rebalance(mm_cash, S, T)
results.append(check("Buying hedge shares decreases cash",
    mm_cash.cash < cash_before,
    f"cash_before={cash_before:.4f}, cash_after={mm_cash.cash:.4f}"))

print("=" * 50)
passed = sum(results)
total  = len(results)
print(f"Results: {passed}/{total} passed")
print("All tests passed." if passed == total else f"{total-passed} test(s) failed.")
print("=" * 50)
