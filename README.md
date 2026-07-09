# Options Market Making Simulator

## Overview

This project simulates an options market maker operating on a single underlying stock. The goal is to model the core P&L dynamics of market making: earning income by quoting bid-ask spreads while managing the inventory risk that accumulates from random order flow.

A market maker is obligated to continuously quote both a bid and an ask price. Traders hit these quotes randomly — some buy, some sell — leaving the MM with a net options position it did not choose. The central challenge is managing this inventory: too much exposure to a large move in the underlying can wipe out weeks of spread income in a single day.

This simulator captures that tradeoff end-to-end:

1. **Stock prices** evolve as Geometric Brownian Motion
2. **Traders** arrive randomly via a Poisson process, hitting the MM's bid or ask
3. **The market maker** quotes spreads around the Black-Scholes theoretical value and adjusts quotes based on current inventory
4. **Delta hedging** neutralizes directional exposure by trading the underlying
5. **P&L** is tracked and decomposed into spread income, theta decay, and gamma losses

The quoting strategy follows the **Avellaneda-Stoikov (2008)** framework, which derives optimal bid-ask spreads analytically as a function of inventory level and time remaining — providing a rigorous benchmark against naive fixed-spread strategies.

## Motivation

Market making is one of the core functions at quantitative trading firms (Citadel Securities, Jane Street, Optiver, DRW). Understanding how a MM balances spread income against inventory risk — and how Greeks like gamma and theta drive that tradeoff — is fundamental to options trading at a quantitative level. This project builds that intuition from the ground up, implementing every component from scratch rather than relying on off-the-shelf backtesting libraries.

## Project Structure

```
options-mm-simulator/
├── src/
│   ├── black_scholes.py   # B-S pricing engine + Greeks (Δ, Γ, Θ, ν)
│   ├── simulator.py       # GBM price paths + Poisson order flow
│   ├── market_maker.py    # Avellaneda-Stoikov quoting strategy
│   ├── hedger.py          # Delta hedging logic
│   └── pnl.py             # P&L tracking and attribution
├── notebooks/
│   └── analysis.ipynb     # Results, visualizations, findings
├── tests/                 # Validation tests for each module
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/options-mm-simulator.git
cd options-mm-simulator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tech Stack

- **Python 3.11+**
- `numpy` — numerical simulation
- `scipy` — statistical distributions
- `matplotlib` — visualization
- `pandas` — data analysis

## 1. Black-Scholes Pricing Engine (`src/black_scholes.py`)

### Overview

The foundation of this simulator is a Black-Scholes pricing engine implemented from scratch. In practice, a market maker continuously reprices their options inventory as the underlying moves — so a fast, accurate theoretical pricer is the starting point for everything else.

Black-Scholes gives the theoretical fair value of a European option under the assumptions of constant volatility, no dividends, and continuous trading. While real markets violate these assumptions (volatility smiles, discrete hedging, jumps), B-S remains the universal quoting benchmark — traders think in terms of B-S implied volatility even when using more complex models.

### Implementation

The pricer computes call and put prices using the closed-form solution:

$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$
$$P = K e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

where:

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

All five Greeks are implemented analytically (not numerically), which is important for a market maker who needs stable, fast sensitivity estimates:

| Greek | Measures | Formula |
|-------|----------|---------|
| Delta (Δ) | Price sensitivity to underlying move | $N(d_1)$ for calls |
| Gamma (Γ) | Rate of change of delta | $\frac{N'(d_1)}{S \sigma \sqrt{T}}$ |
| Theta (Θ) | Daily value decay | See implementation |
| Vega (ν) | Sensitivity to 1% vol change | $S N'(d_1) \sqrt{T} / 100$ |

### Key Relationships

Two properties are validated in tests and are central to the market making logic:

**Put-call parity:** $C - P = S - Ke^{-rT}$. Violations of this are pure arbitrage — any pricing engine that breaks parity is wrong.

**Gamma-theta tradeoff:** A market maker who sells options collects theta (daily time decay) but is exposed to gamma (losses when the stock moves sharply). This tradeoff is the core P&L dynamic the simulator is designed to illustrate.

### Usage

```python
from src.black_scholes import call_price, put_price, delta, gamma, theta, vega

# Price a 3-month ATM call: S=100, K=100, r=5%, sigma=20%
c = call_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20)  # ~5.08
d = delta(S=100, K=100, T=0.25, r=0.05, sigma=0.20)       # ~0.532
g = gamma(S=100, K=100, T=0.25, r=0.05, sigma=0.20)       # ~0.038
```

### Validation

All Greeks are validated against [TradingBlock's options calculator](https://www.tradingblock.com/calculators/option-greeks-calculator) with tolerance ≤ 0.01. Put-call parity is verified algebraically. Run tests with:

```bash
python3 -m tests.test_black_scholes
```

## 2. Market Simulation Engine (`src/simulator.py`)

### Overview

Before building a market maker, you need a realistic environment to run it in. The simulation engine has two components: a stock price model and an order flow model. Together they define the world the market maker operates in — how prices evolve and how traders arrive.

### Stock Price Model: Geometric Brownian Motion

Stock prices are simulated using Geometric Brownian Motion (GBM), the standard model underlying Black-Scholes:

$$S_{t+\Delta t} = S_t \cdot \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right), \quad Z \sim \mathcal{N}(0,1)$$

where $\mu$ is the drift (expected annual return) and $\sigma$ is the annualized volatility.

GBM has two properties that make it suitable for stock price simulation: prices are always positive, and log-returns are normally distributed. The simulator supports generating multiple independent paths simultaneously, which enables Monte Carlo analysis of market maker performance across many market scenarios.

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `S0` | Initial stock price | 100 |
| `mu` | Annual drift | 0.05 |
| `sigma` | Annual volatility | 0.20 |
| `dt` | Time step (1/252 = one trading day) | 1/252 |
| `n_steps` | Number of time steps | 252 |
| `n_paths` | Number of independent simulations | 1 |

### Order Flow Model: Poisson Arrivals

Real market makers face a continuous stream of incoming orders — traders randomly arriving to buy or sell. This is modeled using independent Poisson processes for buy and sell arrivals, which is the standard approach in academic market making literature (Avellaneda-Stoikov, 2008).

At each time step, the number of buy and sell orders are drawn independently:

$$N_{\text{buy}} \sim \text{Poisson}(\lambda_{\text{buy}}), \quad N_{\text{sell}} \sim \text{Poisson}(\lambda_{\text{sell}})$$

The market maker's net inventory change per step is:

$$\Delta q = N_{\text{buy}} - N_{\text{sell}}$$

A positive net flow means more buyers hit the MM's ask (MM sells, accumulates short inventory). A negative net flow means more sellers hit the MM's bid (MM buys, accumulates long inventory).

When $\lambda_{\text{buy}} = \lambda_{\text{sell}}$, the expected net flow is zero and the net flow distribution has standard deviation $\sqrt{\lambda_{\text{buy}} + \lambda_{\text{sell}}}$ — a property verified in tests. In later experiments, asymmetric lambdas simulate directional order flow pressure, a realistic condition market makers must manage.

### Usage

```python
from src.simulator import simulate_gbm, simulate_order_flow

# Simulate 1000 price paths over one trading year
paths = simulate_gbm(S0=100, mu=0.05, sigma=0.20, dt=1/252, n_steps=252, n_paths=1000)

# Simulate daily net order flow
orders = simulate_order_flow(n_steps=252, lambda_buy=3, lambda_sell=3)
# orders[t] = net contracts hitting MM on day t
# positive → MM sold contracts (short inventory)
# negative → MM bought contracts (long inventory)
```

### Validation

GBM paths are validated against the theoretical mean $S_0 e^{\mu T}$ across 1,000 simulations. Order flow is validated against the analytical mean and standard deviation of the Poisson difference distribution. Run tests with:

```bash
python3 -m tests.test_simulator
```

## 3. Avellaneda-Stoikov Market Maker (`src/market_maker.py`)

### Overview

With a pricing engine and an order-flow simulator in place, the next piece is the market maker itself: the logic that decides where to place bid and ask quotes given current inventory and time remaining. A naive market maker would just quote a fixed spread around the Black-Scholes theoretical value regardless of position — but that ignores the central risk of the business, which is accumulating a large directional position that a sharp move in the underlying can turn into a big loss.

The **Avellaneda-Stoikov (2008)** framework solves this with stochastic optimal control: given a market maker maximizing expected utility of terminal wealth, subject to inventory risk and Poisson order arrivals, it derives closed-form expressions for where to center quotes (the *reservation price*) and how wide to make them (the *optimal spread*). This project applies that framework to a single call option, using the Black-Scholes price as the theoretical mid and the underlying's volatility as the risk input.

### Reservation Price: Inventory-Aware Fair Value

Instead of quoting around the raw Black-Scholes price, the market maker quotes around a reservation price that shifts based on current inventory:

$$r(S, t) = C_{BS}(S, K, T-t, r, \sigma) - q \cdot \gamma \sigma^2 (T - t)$$

where $q$ is current inventory (positive = long, negative = short) and $\gamma$ is a risk-aversion parameter. If the market maker is long (bought more than it's sold), the reservation price drops below the theoretical mid — making the ask more attractive to hit and the bid less attractive, nudging inventory back toward flat. Short inventory does the reverse. A flat book leaves the reservation price equal to the Black-Scholes mid.

### Optimal Spread: Balancing Risk Aversion Against Fill Probability

$$\delta = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$

This has two competing components:

- **Time-decay term** (the first term above): grows with volatility, risk aversion, and time remaining. More time left means more opportunity for the underlying to move against an open position, so the model rationally quotes wider.
- **Order-flow term** (the second term above): constant with respect to time, governed by risk aversion and κ (how sharply order arrival probability falls off as quotes move away from the mid). This caps how wide the spread can go before the market maker simply stops getting filled.

| Parameter | Description | Default |
|-----------|--------------|---------|
| `gamma` | Risk aversion — higher values widen spreads and increase inventory skew | 0.1 |
| `kappa` | Order arrival sensitivity — higher values mean traders are more sensitive to spread width | 1.5 |

A notable and non-obvious property of this formula: because the time-decay term shrinks as expiry approaches while the order-flow term stays constant, **the total spread narrows toward expiry**, not widens. That's the correct behavior of the AS model as originally derived — less time left means less accumulated inventory risk. It's also a known simplification in this context: the formula only knows about the underlying's volatility, not option-specific risk like gamma exploding near expiry, so it won't reproduce the spread-widening behavior a real options market maker exhibits late in an option's life.

### Order Flow Processing and Cash Accounting

Each time step, net order flow (buys minus sells, from the Poisson order-flow simulator) is matched against the current quotes:

- **Positive net flow** (more buyers hit the ask): the market maker sells, inventory decreases, cash increases by `net_flow * ask`.
- **Negative net flow** (more sellers hit the bid): the market maker buys, inventory increases, cash decreases by `net_flow * bid`.

This is what turns quoting decisions into a running P&L: cash accumulates from spread income, while inventory accumulates the directional risk that the reservation price and spread are designed to manage.

### Usage

```python
from src.market_maker import MarketMaker

mm = MarketMaker(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

bid, ask = mm.get_quotes(S=100, T_remaining=0.25)

# Simulate the MM selling 5 contracts to buyers
mm.process_order_flow(net_flow=5, S=100, T_remaining=0.25)
# mm.inventory == -5, mm.cash increased by 5 * ask
```

### Validation

Tests check that quotes are well-formed (ask above bid, both positive), that inventory skews the reservation price in the correct direction for both long and short positions, that the spread narrows toward expiry as derived above, and that order flow updates inventory and cash correctly. Run tests with:

```bash
python3 -m tests.test_market_maker
```

Currently only calls are supported — the reservation price and spread formulas track raw contract count and the underlying's volatility, not option delta, so they implicitly assume inventory behaves directionally like a long/short position in the underlying itself. That assumption holds for calls but would need to be revisited (sign-aware skew, or delta-weighted inventory) before puts could be added to the same book.

Greeks calculator: https://www.tradingblock.com/calculators/option-greeks-calculator