# BankNifty Risk Manager OpenEnv

A real-world reinforcement learning environment built for the Meta PyTorch OpenEnv Hackathon. 

## Overview
This environment simulates Indian stock market trading using historical OHLCV and Options Greeks data for BankNifty. It forces AI agents to balance aggressive ROI goals with strict risk management.

### Observation Space (Size: 10)
A continuous array containing:
`[close, volume, oi_put, oi_call, delta_atm, gamma_atm, theta_atm, vega_atm, current_balance, shares_held]`

### Action Space (Discrete: 3)
* `0`: Hold
* `1`: Buy 1 Share
* `2`: Sell 1 Share

## The Tasks
1. **Easy:** Execute valid buy/sell interface commands without crashing.
2. **Medium:** Achieve a net-positive ROI over a short-term simulated window.
3. **Hard:** Maximize returns over an extended window while keeping **maximum drawdown strictly under 10%**.

## How to Run Baseline Inference
```bash
python inference.py