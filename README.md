# Deep Hedging with Reinforcement Learning

## Project Overview
This project implements a **Deep Reinforcement Learning** agent to optimize the hedging of European Call Options in a market with **transaction costs**. 

Traditional models like Black-Scholes assume continuous, cost-free trading. In reality, frequent rebalancing leads to significant transaction costs, eroding profits. This project demonstrates how an AI agent learns to balance the trade-off between **risk reduction** and **cost minimization**.

## Key Results
The AI agent was trained using a custom Utility Loss Function on Monte Carlo simulated paths (Geometric Brownian Motion).

- **Benchmark:** Black-Scholes Delta Hedging (incurs high costs due to frequent rebalancing).
- **AI Solution:** Deep Hedging Agent (learns to trade sparsely, only when necessary).
- **Outcome:** The AI reduced the global loss by approx **80%** compared to the naive Black-Scholes strategy under 1% transaction costs.

## Technical Stack
- **Python**
- **TensorFlow / Keras** (Custom Training Loop & Loss Function)
- **NumPy** (Vectorized Monte Carlo Simulations)
- **Matplotlib** (Visualization)

## Architecture
- **Input:** Market State (Log-Moneyness, Time to Maturity).
- **Model:** Neural Network approximating the optimal hedge ratio $\delta_t$.
- **Optimization:** Minimizing $-E[PnL] + \lambda \cdot Std[PnL]$ (Risk-Adjusted Return).

## File Structure
- `Deep_Hedging_Project.ipynb`: The main Jupyter Notebook containing the simulation, training, and visualization code.

## 📚 References
Based on the "Deep Hedging" framework by Hans Buehler et al. (2019).
