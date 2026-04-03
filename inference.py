import os
import numpy as np
from openai import OpenAI
from env import BankNiftyEnv

class LLMAgent:
    """A baseline agent that uses an LLM to make trading decisions."""
    def __init__(self, action_space):
        self.action_space = action_space
        # MANDATORY: Must use OpenAI Client and specific variables
        self.client = OpenAI(
            base_url=os.environ.get("API_BASE_URL"),
            api_key=os.environ.get("HF_TOKEN")
        )
        self.model_name = os.environ.get("MODEL_NAME")

    def predict(self, obs):
        close_price = obs[0]
        delta = obs[4]
        
        prompt = f"BankNifty Data: Close={close_price:.2f}, Delta={delta:.2f}. You are an AI agent. Reply with exactly ONE number: 0 to Hold, 1 to Buy, 2 to Sell."
        
        try:
            # We keep max_tokens tiny so the inference runs fast
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            action_str = response.choices[0].message.content.strip()
            
            if '1' in action_str: return 1
            elif '2' in action_str: return 2
            else: return 0
        except Exception as e:
            # Fallback to hold if the API rate limits or drops to ensure the script doesn't crash
            return 0

# REDUCED STEPS to strictly comply with the "under 20min" runtime limit
def grade_easy_task(env, agent, steps=10):
    obs, _ = env.reset()
    actions_taken = set()
    for _ in range(steps):
        action = agent.predict(obs)
        actions_taken.add(action)
        obs, _, terminated, _, _ = env.step(action)
        if terminated: break
    score = 1.0 if (1 in actions_taken and 2 in actions_taken) else 0.0
    return score

def grade_medium_task(env, agent, steps=30):
    obs, _ = env.reset()
    initial_net_worth = env.initial_balance
    for _ in range(steps):
        action = agent.predict(obs)
        obs, _, terminated, _, info = env.step(action)
        if terminated: break
    roi = (info['net_worth'] - initial_net_worth) / initial_net_worth
    score = np.clip(roi / 0.05, 0.0, 1.0)
    return float(score)

def grade_hard_task(env, agent, steps=50):
    obs, _ = env.reset()
    initial_net_worth = env.initial_balance
    peak_net_worth = initial_net_worth
    max_drawdown = 0.0
    for _ in range(steps):
        action = agent.predict(obs)
        obs, _, terminated, _, info = env.step(action)
        
        if info['net_worth'] > peak_net_worth:
            peak_net_worth = info['net_worth']
        drawdown = (peak_net_worth - info['net_worth']) / peak_net_worth
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if terminated: break

    roi = (info['net_worth'] - initial_net_worth) / initial_net_worth
    if max_drawdown > 0.10: return 0.0
    score = np.clip(roi / 0.10, 0.0, 1.0)
    return float(score)

if __name__ == "__main__":
    print("--- Starting OpenEnv LLM Baseline Inference ---")
    
    # Quick check to ensure the evaluator provided the variables
    if not all(os.environ.get(k) for k in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]):
        print("WARNING: Required environment variables are missing! Scores may default to 0.")

    env = BankNiftyEnv(data_path='banknifty_historical_data.csv')
    agent = LLMAgent(env.action_space)
    
    easy_score = grade_easy_task(env, agent)
    print(f"Easy Task Score: {easy_score:.2f} / 1.0")
    
    medium_score = grade_medium_task(env, agent)
    print(f"Medium Task Score: {medium_score:.2f} / 1.0")
    
    hard_score = grade_hard_task(env, agent)
    print(f"Hard Task Score: {hard_score:.2f} / 1.0")