import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.market_maker import MarketMaker

S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    if detail:
        print(f"       {detail}")
    return condition

print("=" * 50)
print("Market Maker Tests")
print("=" * 50)

results = []
mm = MarketMaker(S, K, T, r, sigma)

# Basic quote structure
bid, ask = mm.get_quotes(S, T)
results.append(check("Ask > Bid",
    ask > bid,
    f"bid={bid:.4f}, ask={ask:.4f}, spread={ask-bid:.4f}"))

results.append(check("Spread is positive",
    (ask - bid) > 0,
    f"spread={ask-bid:.4f}"))

results.append(check("Quotes are positive",
    bid > 0 and ask > 0,
    f"bid={bid:.4f}, ask={ask:.4f}"))

# Inventory skew: if MM is long, reservation price should drop (want to sell)
mm_long = MarketMaker(S, K, T, r, sigma)
mm_long.inventory = 10
r_long = mm_long.reservation_price(S, T)

mm_short = MarketMaker(S, K, T, r, sigma)
mm_short.inventory = -10
r_short = mm_short.reservation_price(S, T)

mm_flat = MarketMaker(S, K, T, r, sigma)
r_flat = mm_flat.reservation_price(S, T)

results.append(check("Long inventory → lower reservation price",
    r_long < r_flat,
    f"long={r_long:.4f}, flat={r_flat:.4f}, short={r_short:.4f}"))

results.append(check("Short inventory → higher reservation price",
    r_short > r_flat,
    f"short={r_short:.4f}, flat={r_flat:.4f}"))

# Spread narrows as time runs out (Avellaneda-Stoikov: less time left means less
# accumulated inventory risk, so the time-decay term shrinks toward the constant
# order-flow term)
spread_early = mm.optimal_spread(T_remaining=0.25)
spread_late  = mm.optimal_spread(T_remaining=0.01)
results.append(check("Spread narrows near expiry",
    spread_late < spread_early,
    f"early={spread_early:.4f}, near_expiry={spread_late:.4f}"))

# Order flow processing
mm2 = MarketMaker(S, K, T, r, sigma)
mm2.process_order_flow(net_flow=5, S=S, T_remaining=T)
results.append(check("Selling to buyers decreases inventory",
    mm2.inventory == -5,
    f"inventory={mm2.inventory}"))

mm3 = MarketMaker(S, K, T, r, sigma)
mm3.process_order_flow(net_flow=-5, S=S, T_remaining=T)
results.append(check("Buying from sellers increases inventory",
    mm3.inventory == 5,
    f"inventory={mm3.inventory}"))

print("=" * 50)
passed = sum(results)
total  = len(results)
print(f"Results: {passed}/{total} passed")
print("All tests passed." if passed == total else f"{total-passed} test(s) failed.")
print("=" * 50)