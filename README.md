# Deep Hedging with Reinforcement Learning

## Project Overview
This project implements a **Deep Reinforcement Learning (DRL)** agent capable of hedging financial derivatives (European Call Options) under realistic market frictions.

Unlike the traditional **Black-Scholes model** which assumes cost-free trading, this agent learns to manage **Transaction Costs** by optimizing a trade-off between risk reduction and cost minimization.

## Key Results

### 1. Performance vs Benchmark
Under high transaction costs (**1%**), the Deep Hedging agent significantly outperforms the Black-Scholes benchmark.
* **Black-Scholes Loss:** -2.20 (avg)
* **Deep Hedging Loss:** -1.04 (avg)
* **Improvement:** **~52% reduction in losses.**

<img width="1057" height="661" alt="image" src="https://github.com/user-attachments/assets/e96bea18-0f17-476e-b8fd-85d287153e25" />
Figure 1: Final P&L Distribution. The AI (Green) is centered closer to 0 compared to Black-Scholes (Red)

### 2. Strategy Analysis (Bandwidth Hedging)
The AI learned to trade sparsely. Unlike Black-Scholes, which adjusts positions continuously (smooth gradient), the AI creates "no-trade zones" (plateaus) to avoid unnecessary costs.

<img width="1598" height="656" alt="image" src="https://github.com/user-attachments/assets/a9a4bb71-9f34-4710-a5e7-af6286843121" />
Figure 2: Learned Policy Heatmap. Left: Black-Scholes / Right: Deep Hedging with "step-like" behavior

## Methodology (The "Story")

This project follows a rigorous 3-step scientific validation:

1.  **Baseline Failure Analysis:** We demonstrated that Black-Scholes fails in the presence of 1% transaction costs (P&L shifts to negative).
2.  **Theoretical Validation (Warm Start):** We verified the Neural Network's capacity by training it with **Zero Costs**. It successfully "rediscovered" the exact Black-Scholes formula from scratch (Validation Heatmap matches theoretical one).
3.  **Optimization:** We re-trained the agent with **1% costs**. The agent diverged from the theoretical curve to adopt a "Bandwidth Hedging" strategy (trading less often), which proved to be a more optimal solution.

## Technical Stack
* **Python**
* **TensorFlow / Keras** (Custom Training Loops, GradientTape)
* **NumPy** (Vectorized Monte Carlo Simulations)
* **Matplotlib** (Visualization)

## File Structure
* `Deep_Hedging_Project.ipynb`: The complete Jupyter Notebook containing simulation, model definition, training, and visualization.

## References
* *Deep Hedging*, Hans Buehler et al. (2019).
