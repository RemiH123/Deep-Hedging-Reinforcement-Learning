# Deep Hedging with Reinforcement Learning

## Project Overview
This project implements a **Deep Reinforcement Learning (DRL)** agent capable of hedging financial derivatives (European Call Options) under realistic market frictions.

In a frictionless market, the **Black-Scholes** model provides a perfect hedge. However, in the real world, **Transaction Costs** make continuous rebalancing prohibitively expensive. This project builds an AI agent that learns to balance **Risk Reduction** (Hedging) against **Cost Minimization**.

### Objective
Beat the Black-Scholes benchmark by developing a strategy that minimizes the variance of the P&L while reducing trading costs.

---

## Methodology & The "Story"

This notebook documents a rigorous 4-step scientific approach to solving the hedging problem.

### Chapter 1: The Benchmark (Black-Scholes)
We established the baseline using Geometric Brownian Motion simulations.
* **Observation:** Under **1% transaction costs**, the Black-Scholes strategy bleeds capital due to over-trading, shifting the P&L distribution significantly into the negative.

### Chapter 2: The "Naive" RL Failure
We attempted to train a DRL agent from scratch in a high-cost environment.
* **Result:** The agent fell into a **Local Minimum ("Static Hedging")**.
* **Analysis:** The penalty for trading was so high initially that the agent "gave up" on hedging to avoid costs. While this saved money, it failed to reduce risk (high variance). This proved that a naive approach is insufficient.

### Chapter 3: Model Validation (Warm Start)
To solve the "Cold Start" problem, we used **Supervised Learning** in a zero-cost environment.
* **Method:** We forced the Neural Network to mimic the Black-Scholes Delta.
* **Result:** The model successfully "rediscovered" the theoretical curve. This validated our network architecture (Dense + BatchNormalization + LeakyReLU).

### Chapter 4: The Final Solution (Transfer Learning)
We used **Transfer Learning**: we took the "smart" pre-trained brain from Chapter 3 and fine-tuned it with **Reinforcement Learning** under real market costs (1%).
* **Result:** The agent diverged from the theoretical curve to adopt a **"Bandwidth Hedging"** strategy. It learned to tolerate small price movements ("inertia zones") to save costs, only rebalancing when necessary.

---

## Key Results

### 1. Risk Analysis: Why Optimization Matters
We compared three strategies under 1% transaction costs.

| Metric | Black-Scholes | Naive AI (Static) | Optimized AI (Final) |
| :--- | :--- | :--- | :--- |
| **Strategy** | Continuous Trading | Lazy Trading | **Bandwidth Trading** |
| **Cost Impact** | Massive Losses (-2.2) | Few Cost (-0.8) | **Controlled Cost** (-0.9) |
| **Risk (Volatility)** | Low (Theoretical) | **Extreme (High)** | **Low (Optimized)** |

* **Black-Scholes:** Fails due to costs.
* **Naive AI:** Profitable on average but **dangerous** (high variance/risk).
* **Optimized AI:** The best trade-off. It accepts a small cost to drastically reduce risk, achieving a P&L distribution centered and closer to zero.

<img width="1264" height="678" alt="image" src="https://github.com/user-attachments/assets/3a3b871c-9041-4358-aab0-5cd076ad1c83" />

Figure 1: P&L Distribution. The Optimized AI (Green) achieves a narrow "Bell Curve" like Black-Scholes, but with significantly lower costs.

### 2. Strategy Visualization (The "Bandwidth" Effect)
The difference is visible in the policy heatmaps.
* **Black-Scholes (Left):** Smooth gradient = Trades on every small price change.
* **Deep Hedging (Right):** Stepped gradient = **Bandwidth Hedging**. The AI creates "plateaus" where it holds its position to avoid fees.

<img width="1669" height="688" alt="image" src="https://github.com/user-attachments/assets/c7cc9d21-0f3b-4478-a334-1b21ada418a3" />
Figure 2: Learned Policy. Note the "steps" in the Deep Hedging strategy indicating inertia zones.

---

## Technical Stack
* **Python** (NumPy, Matplotlib)
* **TensorFlow / Keras**
* **Custom Training Loop** with `GradientTape`
* **Techniques:**
    * Batch Normalization (Input Scaling)
    * LeakyReLU (Gradient flow)
    * Supervised Pre-training (Warm Start)
    * Transfer Learning

## Acknowledgments & References
* **Paper:** *Deep Hedging*, Hans Buehler et al. (2019).
