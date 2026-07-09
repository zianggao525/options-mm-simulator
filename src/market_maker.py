import numpy as np
from src.black_scholes import call_price, delta as bs_delta

# --- Model Parameters ---
GAMMA = 0.1    # risk aversion: higher = wider spreads, more aggressive inventory skew
KAPPA = 1.5    # order arrival sensitivity: higher = traders more sensitive to spread width


class MarketMaker:
    def __init__(self, S, K, T, r, sigma, gamma=GAMMA, kappa=KAPPA):
        """
        S     : initial stock price
        K     : option strike
        T     : time to expiry in years
        r     : risk-free rate
        sigma : volatility
        """
        self.K     = K
        self.r     = r
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = kappa

        self.inventory  = 0      # net options position (+ = long, - = short)
        self.cash       = 0.0    # cumulative cash from trading
        self.hedge_pos  = 0.0    # current share position from delta hedging

    def reservation_price(self, S, T_remaining):
        """
        Avellaneda-Stoikov reservation price.
        Adjusts the mid price down if long inventory, up if short.
        """
        mid = call_price(S, self.K, T_remaining, self.r, self.sigma)
        adjustment = self.inventory * self.gamma * self.sigma**2 * T_remaining
        return mid - adjustment

    def optimal_spread(self, T_remaining):
        """
        Avellaneda-Stoikov optimal bid-ask spread.
        Widens as time remaining decreases and as risk aversion increases.
        """
        spread = (self.gamma * self.sigma**2 * T_remaining
                  + (2 / self.gamma) * np.log(1 + self.gamma / self.kappa))
        return max(spread, 0.01)  # floor at 1 cent

    def get_quotes(self, S, T_remaining):
        """
        Returns (bid, ask) quotes for the option.
        """
        r_price = self.reservation_price(S, T_remaining)
        spread  = self.optimal_spread(T_remaining)
        bid = r_price - spread / 2
        ask = r_price + spread / 2
        return bid, ask

    def process_order_flow(self, net_flow, S, T_remaining):
        """
        Process net order flow for one time step.

        net_flow > 0 : traders bought from MM → MM sold, inventory decreases
        net_flow < 0 : traders sold to MM   → MM bought, inventory increases

        Returns cash collected this step.
        """
        if net_flow == 0:
            return 0.0

        bid, ask = self.get_quotes(S, T_remaining)

        if net_flow > 0:
            # MM sold net_flow contracts at the ask
            cash_flow       = net_flow * ask
            self.inventory -= net_flow
        else:
            # MM bought abs(net_flow) contracts at the bid
            cash_flow       = net_flow * bid   # negative * negative = positive
            self.inventory -= net_flow         # inventory increases

        self.cash += cash_flow
        return cash_flow