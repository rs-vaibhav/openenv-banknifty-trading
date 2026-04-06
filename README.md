---
title: BankNifty Risk Manager
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---
# 📈 BankNifty Risk Manager (OpenEnv)

## Environment Description & Motivation
The **BankNifty Risk Manager** is a real-world quantitative finance environment designed to test frontier AI agents on capital preservation and risk-adjusted returns. Unlike standard trading environments that only reward absolute profit (PnL), this environment simulates strict institutional trading desks. Agents must navigate volatile options data while adhering to a hard **5% Maximum Drawdown guardrail**. If the agent takes excessive risk, it is heavily penalized, forcing models to balance aggressive momentum trading with strict capital preservation.

## Action & Observation Spaces
**Action Space (Discrete: 3)**
The agent outputs a single integer to interact with the market:
* `0`: HOLD (Take no action / Maintain current state)
* `1`: BUY (Enter a long position)
* `2`: SELL (Liquidate current position)

**Observation Space (Vector)**
The environment returns a comprehensive state vector containing real-time market data and internal portfolio risk metrics:
* Current Asset Price
* Position Status (0 = Flat, 1 = Long)
* Current Account Balance
* Realized PnL
* Current Max Drawdown % (Crucial for risk guardrails)

## Tasks & Graders
This environment implements three distinct tasks of increasing difficulty. The internal grader calculates a deterministic score between `0.0` and `1.0` at the end of every episode based on capital survival and ROI.

* **Easy (api-compliance):** The agent must successfully connect to the environment, parse the observation space, and execute 5 valid actions without crashing. (Score: 1.0 for completion).
* **Medium (short-term-roi):** The agent must navigate 10 market steps and generate a net-positive PnL. (Score: 0.5 for breaking even, scaling up to 1.0 for high profit. 0.0 for losses).
* **Hard (risk-adjusted):** The agent must trade through 15 volatile steps, generating profit while keeping Max Drawdown strictly under 5%. (Score: 0.0 if drawdown exceeds 5%, regardless of final PnL).

## Setup & Usage Instructions
To evaluate an agent locally using the official Hugging Face Router:

1. Clone the repository and install dependencies using `uv`.
2. Export your API credentials and target model:
```bash
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"