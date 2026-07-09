from src.black_scholes import delta as bs_delta


def rebalance(mm, S, T_remaining):
    """
    Delta-hedge the market maker's current option inventory by trading
    shares of the underlying so net portfolio delta is (near) zero.

    Long option inventory carries positive delta, so mm needs a short
    stock position (and vice versa) to offset it.

    Returns the number of shares traded this step (+ = bought, - = sold).
    """
    option_delta = bs_delta(S, mm.K, T_remaining, mm.r, mm.sigma)
    target_hedge = -mm.inventory * option_delta

    trade = target_hedge - mm.hedge_pos
    mm.cash -= trade * S
    mm.hedge_pos += trade
    return trade


def net_delta(mm, S, T_remaining):
    """
    Portfolio delta after hedging: option delta from inventory plus the
    hedge share position. Should be ~0 immediately after rebalance().
    """
    option_delta = bs_delta(S, mm.K, T_remaining, mm.r, mm.sigma)
    return mm.inventory * option_delta + mm.hedge_pos
