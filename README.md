# options-mm-simulator

# Options Market Making Simulator

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



Greeks calculator: https://www.tradingblock.com/calculators/option-greeks-calculator