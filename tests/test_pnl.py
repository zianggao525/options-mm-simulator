import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_maker import MarketMaker
from src.pnl import mark_to_market, decompose_step

S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    if detail:
        print(f"       {detail}")
    return condition

print("=" * 50)
print("PnL Tests")
print("=" * 50)

results = []

# A flat, untraded book has zero portfolio value
mm_flat = MarketMaker(S, K, T, r, sigma)
mtm_flat = mark_to_market(mm_flat, S, T)
results.append(check("Flat book has zero mark-to-market value",
    abs(mtm_flat) < 1e-8,
    f"mtm={mtm_flat:.8f}"))

# Selling into a buy order should immediately capture half the spread as P&L
mm_trade = MarketMaker(S, K, T, r, sigma)
bid, ask = mm_trade.get_quotes(S, T)
mm_trade.process_order_flow(net_flow=5, S=S, T_remaining=T)
mtm_trade = mark_to_market(mm_trade, S, T)
expected_edge = 5 * (ask - bid) / 2
results.append(check("Trade immediately captures spread edge as P&L",
    abs(mtm_trade - expected_edge) < 1e-8,
    f"mtm={mtm_trade:.4f}, expected={expected_edge:.4f}"))

# Theta P&L: long inventory loses value to time decay, short gains
mm_long = MarketMaker(S, K, T, r, sigma)
mm_long.inventory = 10
theta_pnl_long, _ = decompose_step(mm_long, S_prev=S, S_new=S, T_remaining=T, dt=1/365)
results.append(check("Long inventory has negative theta P&L",
    theta_pnl_long < 0,
    f"theta_pnl={theta_pnl_long:.6f}"))

mm_short = MarketMaker(S, K, T, r, sigma)
mm_short.inventory = -10
theta_pnl_short, _ = decompose_step(mm_short, S_prev=S, S_new=S, T_remaining=T, dt=1/365)
results.append(check("Short inventory has positive theta P&L",
    theta_pnl_short > 0,
    f"theta_pnl={theta_pnl_short:.6f}"))

# Gamma P&L: long inventory benefits from a move in either direction
_, gamma_pnl_up = decompose_step(mm_long, S_prev=S, S_new=S + 5, T_remaining=T, dt=1/365)
_, gamma_pnl_down = decompose_step(mm_long, S_prev=S, S_new=S - 5, T_remaining=T, dt=1/365)
results.append(check("Long inventory has positive gamma P&L on an up move",
    gamma_pnl_up > 0,
    f"gamma_pnl={gamma_pnl_up:.4f}"))
results.append(check("Long inventory has positive gamma P&L on a down move",
    gamma_pnl_down > 0,
    f"gamma_pnl={gamma_pnl_down:.4f}"))

# Short inventory has the opposite sign: big moves hurt a short gamma position
_, gamma_pnl_short = decompose_step(mm_short, S_prev=S, S_new=S + 5, T_remaining=T, dt=1/365)
results.append(check("Short inventory has negative gamma P&L on a move",
    gamma_pnl_short < 0,
    f"gamma_pnl={gamma_pnl_short:.4f}"))

# No move, no time elapsed -> no theta or gamma P&L
mm_zero = MarketMaker(S, K, T, r, sigma)
mm_zero.inventory = 10
theta_zero, gamma_zero = decompose_step(mm_zero, S_prev=S, S_new=S, T_remaining=T, dt=0)
results.append(check("Zero elapsed time means zero theta and gamma P&L",
    theta_zero == 0 and gamma_zero == 0,
    f"theta_pnl={theta_zero}, gamma_pnl={gamma_zero}"))

print("=" * 50)
passed = sum(results)
total  = len(results)
print(f"Results: {passed}/{total} passed")
print("All tests passed." if passed == total else f"{total-passed} test(s) failed.")
print("=" * 50)
