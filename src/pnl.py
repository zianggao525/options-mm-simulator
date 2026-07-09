from src.black_scholes import call_price, gamma as bs_gamma, theta as bs_theta


def mark_to_market(mm, S, T_remaining):
    """
    Total portfolio value: cash plus the mark-to-market value of the
    option inventory and the hedge share position.

    For a freshly flat book this is 0. Right after a trade (before any
    hedging or stock move), this captures the spread edge earned on
    that trade.
    """
    option_value = mm.inventory * call_price(S, mm.K, T_remaining, mm.r, mm.sigma)
    hedge_value  = mm.hedge_pos * S
    return mm.cash + option_value + hedge_value


def decompose_step(mm, S_prev, S_new, T_remaining, dt):
    """
    Theoretical theta and gamma P&L for one time step, evaluated at the
    start-of-step Greeks.

    This is the classic delta-hedged option P&L decomposition: once delta
    risk is hedged away, what's left is time decay (theta) and a gamma
    term driven by the realized move in the stock. Gamma P&L is always
    the same sign as inventory regardless of which way the stock moves,
    since it scales with the squared move.

    Returns (theta_pnl, gamma_pnl).
    """
    g  = bs_gamma(S_prev, mm.K, T_remaining, mm.r, mm.sigma)
    th = bs_theta(S_prev, mm.K, T_remaining, mm.r, mm.sigma)  # per day

    theta_pnl = mm.inventory * th * (dt * 365)
    gamma_pnl = 0.5 * mm.inventory * g * (S_new - S_prev) ** 2
    return theta_pnl, gamma_pnl
