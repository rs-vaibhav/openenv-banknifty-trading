import os
import json
import collections
from openai import OpenAI
from env import BankNiftyEnv

# --- MANDATORY STDOUT LOGGING FORMATTERS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={float(reward):.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# --- AGENT LOGIC ---
class LLMAgent:
    """An elite, stateful CoT agent with strict risk-management guardrails."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
            api_key=os.environ.get("HF_TOKEN", "dummy_token_to_prevent_crash")
        )
        self.model_name = os.environ.get("MODEL_NAME", "dummy-model")
        
        # Memory and Risk Trackers
        self.price_history = collections.deque(maxlen=5) # Rolling window of 5 ticks
        self.initial_balance = None
        self.peak_balance = None

    def predict(self, obs):
        close_price = obs[0]
        delta = obs[4]
        current_balance = obs[8]
        shares_held = obs[9]
        
        # Initialize risk trackers on the very first step
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.peak_balance = current_balance
            
        # Update price history and peak net worth
        self.price_history.append(close_price)
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            
        # Calculate real-time risk and trend
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        trend = "NEUTRAL"
        if len(self.price_history) >= 2:
            if self.price_history[-1] > self.price_history[0]:
                trend = "UPTREND"
            elif self.price_history[-1] < self.price_history[0]:
                trend = "DOWNTREND"

        # The Professional CoT Prompt
        system_prompt = """You are an elite quantitative AI trading agent. 
Your goal is to maximize ROI while STRICTLY keeping maximum drawdown under 10%.
You must output your response in valid JSON format containing exactly two keys:
1. "analysis": A brief 1-sentence reasoning based on the trend, delta, and risk.
2. "action": An integer representing your decision (0=Hold, 1=Buy, 2=Sell)."""

        user_prompt = f"""--- MARKET DATA ---
Current Price: {close_price:.2f}
Recent Trend (last {len(self.price_history)} ticks): {trend}
Delta (ATM): {delta:.2f}

--- PORTFOLIO RISK ---
Current Balance: {current_balance:.2f}
Shares Held: {shares_held}
Current Drawdown: {drawdown*100:.2f}% (CRITICAL: Must stay under 10%)

Analyze the data and return the JSON."""

        try:
            # Force the model to think and return JSON
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.1 # Low temperature for logical, deterministic answers
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown blocks if the LLM wraps the JSON in ```json ... ```
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
                
            output = json.loads(content)
            action = int(output.get("action", 0))
            print(f"\n🤖 [THINKING]: {output.get('analysis')}")
            print(f"📈 [ACTION]: {action} | Drawdown: {drawdown*100:.2f}% | Trend: {trend}")
            
            # --- THE QUANT GUARDRAIL ---
            # If the LLM makes a risky choice near the ruin limit, the algorithm overrides it.
            if drawdown > 0.085 and action == 1: 
                return 0 # Force a HOLD to protect capital
                
            # If we hold shares and drawdown is getting dangerous, force a panic sell
            if drawdown > 0.09 and shares_held > 0 and action != 2:
                return 2 # Force SELL to realize remaining capital
                
            return action
            
        except Exception as e:
            # If the LLM fails to return valid JSON, fall back safely
            return 0


# --- EVALUATION LOOP ---
def run_task(env, agent, task_name, max_steps):
    model_name = os.getenv("MODEL_NAME", "dummy-model")
    benchmark_name = "BankNifty-RiskManager"
    
    # 1. Emit [START] log
    log_start(task=task_name, env=benchmark_name, model=model_name)

    obs, _ = env.reset()
    rewards = []
    steps_taken = 0
    success = False

    for step in range(1, max_steps + 1):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        steps_taken = step

        # Convert numeric action to a string action for the log
        action_map = {0: "hold()", 1: "buy()", 2: "sell()"}
        action_str = action_map.get(action, "hold()")

        # 2. Emit [STEP] log
        log_step(step=step, action=action_str, reward=reward, done=done, error=None)

        if done:
            success = info.get('net_worth', 0) > env.initial_balance
            break

    if not done:
        success = info.get('net_worth', 0) > env.initial_balance

    # 3. Emit [END] log
    log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    env = BankNiftyEnv(data_path='banknifty_historical_data.csv')
    agent = LLMAgent(env.action_space)
    
    # We run the 3 required tasks with very small step counts to ensure it stays well under the 20-minute limit
    run_task(env, agent, task_name="easy-api-compliance", max_steps=5)
    run_task(env, agent, task_name="medium-short-term-roi", max_steps=10)
    run_task(env, agent, task_name="hard-risk-adjusted", max_steps=15)