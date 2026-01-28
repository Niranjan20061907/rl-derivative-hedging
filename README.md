# Deep Hedging with Reinforcement Learning

## Overview
This project studies derivative hedging using reinforcement learning under market frictions and model uncertainty.

## Motivation
Black–Scholes delta hedging is optimal only under idealized assumptions. Real markets exhibit stochastic volatility and transaction costs.

## Methodology
- Custom Gymnasium environment for option hedging
- PPO agent (Stable-Baselines3)
- GBM and Heston market simulators
- Risk-aware reward function

## Results
- Delta hedging performs well under GBM
- Under Heston dynamics, delta hedging exhibits large tail-risk spikes
- PPO adapts to realized dynamics and reduces extreme hedging errors

## Key Takeaways
- Model risk dominates in realistic markets
- Reinforcement learning enables adaptive hedging under uncertainty
