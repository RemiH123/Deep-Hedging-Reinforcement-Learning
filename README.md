# Deep Hedging with Reinforcement Learning

## Project Overview
This project implements a **Deep Reinforcement Learning (DRL)** agent capable of hedging financial derivatives (European Call Options) under realistic market frictions.

Unlike the traditional **Black-Scholes model** which assumes cost-free trading, this agent learns to manage **Transaction Costs** by optimizing a trade-off between risk reduction and cost minimization.

The project demonstrates a rigorous quantitative approach, moving from theoretical validation to real-world optimization.

---

## Key Results

### 1. Performance vs Benchmark
Under high transaction costs (**1%**), the Deep Hedging agent significantly outperforms the Black-Scholes benchmark.
* **Black-Scholes Loss:** -2.20 (avg) - *Bleeds money due to frequent rebalancing.*
* **Deep Hedging Loss:** -1.06 (avg) - *Saves capital by trading smarter.*
* **Improvement:** **~52% reduction in losses.**
<img width="1067" height="666" alt="image" src="https://github.com/user-attachments/assets/ea703777-3cb7-4cad-b07e-312f4d35cb1a" />

Figure 1: Final P&L Distribution. The AI (Green) is centered closer to 0 compared to Black-Scholes (Red), showing better risk management under friction

### 2. Strategy Analysis (Bandwidth Hedging)
The AI learned to trade sparsely. Unlike Black-Scholes, which adjusts positions continuously (smooth gradient), the AI creates "no-trade zones" (plateaus) to avoid unnecessary costs.

<img width="1658" height="685" alt="image" src="https://github.com/user-attachments/assets/440deb14-a96f-4cbb-94a5-a9bc828449cb" />

Figure 2: Learned Policy Heatmap. Left: Black-Scholes (Continuous) / Right: Deep Hedging (Stepped/Bandwidth)

---

## Methodology & Technical Architecture

This project follows a rigorous progression from financial theory to applied deep learning, incorporating advanced neural network techniques to ensure convergence.

### 1. The Foundation: Financial Simulation (Chapter 1)
Following the standard approach for pricing derivatives, we simulate the asset price paths using a **Geometric Brownian Motion** (GBM).
* **Parameters:** We set a fixed random seed to ensure the reproducibility of the stochastic paths and comparing results across different models.
* **Baseline:** We calculate the theoretical Delta using the Black-Scholes formula ($N(d_1)$) to establish a "perfect world" benchmark.

### 2. The "Translation" to TensorFlow (Chapter 2)
The core concept is to replace the analytical Black-Scholes formula with a Neural Network.
* **Logic:** Instead of an explicit formula, the network receives the state ($S_t, t$) and outputs a hedging decision ($\delta_t$).
* **Objective:** The model is trained to minimize the difference between the final hedged portfolio value and zero (Target: $Cost + Payoff - Premium \approx 0$).
* **Initial Failure:** A naive implementation often fails to converge or gets stuck in local minima (static hedging), resulting in a non-centered error distribution.

### 3. Engineering the Solution (Chapters 3 & 4)
To solve the convergence issues observed in the naive approach, we implemented specific architectural improvements recommended for Deep Hedging:

* **Deep Architecture:** We upgraded from a simple regression layer to a deeper network with **3 hidden layers of 32 nodes** to capture the convexity of the option price.
* **Stabilization (Batch Normalization):** Financial time-series data (Prices $\approx 100$, Time $\approx 0.08$) have vastly different scales. We added **Batch Normalization** layers to stabilize the learning process and prevent gradient saturation.
* **Activation Functions:**
    * **LeakyReLU:** Used in hidden layers to prevent the "dying ReLU" problem and allow better gradient flow.
    * **Sigmoid (Output):** Used for the final layer because the Delta of a Long Call Option is mathematically bounded between **0 and 1**.

### 4. Validation & Optimization
* **Warm Start (Transfer Learning):** We pre-trained the network under zero-cost conditions using Supervised Learning. This ensured the model could theoretically reproduce Black-Scholes dynamics before tackling frictions.
* **Real-World Frictions:** Finally, we introduced **1% transaction costs**. The optimized architecture successfully learned a **"Bandwidth Hedging"** strategy, trading sparsely to minimize costs while maintaining a robust hedge.

---

## Technical Stack
* **Python**
* **TensorFlow / Keras** (Custom Training Loops, GradientTape)
* **NumPy** (Vectorized Monte Carlo Simulations)
* **Matplotlib** (Visualization)

## File Structure
* `Deep_Hedging_Project.ipynb`: The complete Jupyter Notebook containing simulation, model definition, training, and visualization.

## References
* *Deep Hedging*, Hans Buehler et al. (2019).
* *Reinforcement Learning for Finance*, J.P. Morgan AI Research.
