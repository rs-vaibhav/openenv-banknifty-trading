import numpy as np
from env import BankNiftyEnv

class BaselineAgent:
    """A simple rule-based agent for our baseline submission."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        # Basic mean-reversion logic just to have a reproducible baseline
        # obs[0] is current close, obs[4] is ATM Delta (just as an example trigger)
        close_price = obs[0]
        delta = obs[4]
        
        if delta > 0.3:
            return 1  # Buy signal
        elif delta < -0.3:
            return 2  # Sell signal
        return 0      # Hold

def grade_easy_task(env, agent, steps=100):
    """EASY: Execute valid buy and sell orders. Score 0.0 or 1.0."""
    obs, _ = env.reset()
    actions_taken = set()
    
    for _ in range(steps):
        action = agent.predict(obs)
        actions_taken.add(action)
        obs, _, terminated, _, _ = env.step(action)
        if terminated: break
        
    # Score 1.0 if it managed to at least attempt both a buy (1) and sell (2)
    score = 1.0 if (1 in actions_taken and 2 in actions_taken) else 0.0
    return score

def grade_medium_task(env, agent, steps=2000):
    """MEDIUM: Achieve a positive return over a short window (~1 month)."""
    obs, _ = env.reset()
    initial_net_worth = env.initial_balance
    
    for _ in range(steps):
        action = agent.predict(obs)
        obs, _, terminated, _, info = env.step(action)
        if terminated: break
        
    final_net_worth = info['net_worth']
    roi = (final_net_worth - initial_net_worth) / initial_net_worth
    
    # Score mapping: <=0% ROI = 0.0, >=5% ROI = 1.0
    score = np.clip(roi / 0.05, 0.0, 1.0)
    return float(score)

def grade_hard_task(env, agent, steps=10000):
    """HARD: Generate returns while keeping max drawdown under 10%."""
    obs, _ = env.reset()
    initial_net_worth = env.initial_balance
    peak_net_worth = initial_net_worth
    max_drawdown = 0.0
    
    for _ in range(steps):
        action = agent.predict(obs)
        obs, _, terminated, _, info = env.step(action)
        
        current_nw = info['net_worth']
        if current_nw > peak_net_worth:
            peak_net_worth = current_nw
            
        drawdown = (peak_net_worth - current_nw) / peak_net_worth
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
        if terminated: break

    final_net_worth = info['net_worth']
    roi = (final_net_worth - initial_net_worth) / initial_net_worth
    
    # Fail heavily if drawdown exceeds 10%
    if max_drawdown > 0.10:
        return 0.0
        
    # Score mapping: <=0% ROI = 0.0, >=10% ROI = 1.0 (with safe drawdown)
    score = np.clip(roi / 0.10, 0.0, 1.0)
    return float(score)

if __name__ == "__main__":
    print("--- Starting OpenEnv Baseline Inference ---")
    env = BankNiftyEnv(data_path='banknifty_historical_data.csv')
    agent = BaselineAgent(env.action_space)
    
    print("\nEvaluating Easy Task (API Compliance)...")
    easy_score = grade_easy_task(env, agent)
    print(f"Easy Task Score: {easy_score:.2f} / 1.0")
    
    print("\nEvaluating Medium Task (Short-term ROI)...")
    medium_score = grade_medium_task(env, agent)
    print(f"Medium Task Score: {medium_score:.2f} / 1.0")
    
    print("\nEvaluating Hard Task (Risk-Adjusted Return)...")
    hard_score = grade_hard_task(env, agent)
    print(f"Hard Task Score: {hard_score:.2f} / 1.0")
    
    print(f"\nFinal Baseline Submission Scores -> [Easy: {easy_score:.2f}, Medium: {medium_score:.2f}, Hard: {hard_score:.2f}]")