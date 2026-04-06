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
            base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1"),
            api_key=os.environ.get("HF_TOKEN")
        )
        self.model_name = os.environ.get("MODEL_NAME", "dummy-model")
        
        # Memory and Risk Trackers
        self.price_history = collections.deque(maxlen=40) # Rolling window of 5 ticks
        self.initial_balance = None
        self.peak_balance = None

    def predict(self, obs):
        close_price = obs[0]
        delta = obs[4]
        current_balance = obs[8]
        shares_held = obs[9]
        
        # Initialize risk trackers
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.peak_balance = current_balance
            
        self.price_history.append(close_price)
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        trend = "NEUTRAL"
        # --- UPGRADED QUANT TREND LOGIC (20-Period SMA) ---
        trend = "NEUTRAL"
        
        # We need at least 5 ticks to start establishing a real average
        if len(self.price_history) >= 10:
            # 1. Calculate the Simple Moving Average
            sma = sum(self.price_history) / len(self.price_history)
            
            # 2. Calculate how far the current price is from the average
            distance_from_sma = (close_price - sma) / sma
            
            # 3. The Chop Filter: Price must break away by 0.1% to be a "real" trend
            chop_threshold = 0.001 
            
            if distance_from_sma > chop_threshold:
                trend = "UPTREND"
            elif distance_from_sma < -chop_threshold:
                trend = "DOWNTREND"
            else:
                trend = "NEUTRAL" # It's just noise, ignore it

        # --- THE COUNCIL OF QUANTS PROMPTS ---
        market_data = f"Price: {close_price:.2f} | Trend: {trend} | Delta: {delta:.2f}"
        risk_data = f"Balance: {current_balance:.2f} | Shares: {shares_held} | Drawdown: {drawdown*100:.2f}%"

        # 1. Momentum Agent
        prompt_momentum = f"""You are a Momentum Trader. 
Trend: {trend} | Price: {close_price:.2f}
If UPTREND, output 1. 
If DOWNTREND, output 2. 
If NEUTRAL, output 0. 
Output exactly one digit. Nothing else."""

        # 2. Contrarian Agent
        prompt_contrarian = f"""You are a Contrarian Trader. You fade the market.
Trend: {trend} | Delta: {delta:.2f}
If UPTREND, output 2. 
If DOWNTREND, output 1. 
If NEUTRAL, output 0.
Output exactly one digit. Nothing else."""

        # 3. Chief Risk Officer (CRO)
        prompt_cro = fprompt_cro = f"""You are the Chief Risk Officer.
Drawdown is {drawdown*100:.2f}%.
If Drawdown is over 4.00%, output 2.
If Drawdown is under 4.00%, output 1.
Output exactly one digit. Nothing else."""

        def query_agent(prompt_text):
            """Helper function to query an individual agent"""
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=15, # Gave it a little more room to talk
                    temperature=0.0 
                )
                action_str = response.choices[0].message.content.strip()
                
                # --- NEW PARSING LOGIC ---
                import re
                # Look specifically for isolated digits 0, 1, or 2
                numbers = re.findall(r'\b[012]\b', action_str)
                
                if numbers:
                    # Take the LAST number found, bypassing chatty explanations
                    return int(numbers[-1]) 
                else:
                    # Debug print if it outputs complete nonsense
                    print(f"⚠️ [PARSE ERROR] Model said: {action_str}")
                    return 0
                    
            except Exception as e:
                print(f"🚨 [API ERROR]: {e}")
                return 0

        # --- PARALLEL EXECUTION ---
        momentum_vote = query_agent(prompt_momentum)
        contrarian_vote = query_agent(prompt_contrarian)
        cro_vote = query_agent(prompt_cro)

        print(f"\n🧠 [COUNCIL VOTES] Momentum: {momentum_vote} | Contrarian: {contrarian_vote} | CRO: {cro_vote}")

        # --- THE AGGREGATOR LOGIC ---
        final_action = 0
        
        # Rule 1: The CRO has absolute Veto Power
        if cro_vote == 2 and shares_held > 0:
            final_action = 2
            print("🚨 [AGGREGATOR] CRO Override! Forced Liquidation to protect capital.")
            
        elif cro_vote == 0:
            if shares_held > 0 and (momentum_vote == 2 or contrarian_vote == 2):
                final_action = 2 # Sell if anyone wants out
            else:
                final_action = 0
            print("🛑 [AGGREGATOR] CRO Vetoed buying. Holding or Selling only.")
            
        # Rule 2: If CRO Approves (1), the Traders vote
        else:
            # AGGRESSIVE MODE: No more Gridlock. 
            # If Momentum trader has a signal, we take it immediately.
            if momentum_vote != 0:
                final_action = momentum_vote
                print(f"💥 [AGGREGATOR] Aggressive Mode! Trusting Momentum ({momentum_vote}).")
                
            # If Momentum is quiet but Contrarian wants to trade, we take it.
            elif contrarian_vote != 0:
                final_action = contrarian_vote
                print(f"💥 [AGGREGATOR] Aggressive Mode! Trusting Contrarian ({contrarian_vote}).")
                
            # Only hold if BOTH traders are completely neutral (0)
            else:
                final_action = 0
                print("⚖️ [AGGREGATOR] Both traders are neutral. Holding.")

        return final_action
                


# --- EVALUATION LOOP ---
def run_task(env, agent, task_name, max_steps):
    model_name = os.getenv("MODEL_NAME", "dummy-model")
    benchmark_name = "BankNifty-RiskManager"
    
    # 1. Emit [START] log
    log_start(task=task_name, env=benchmark_name, model=model_name)

    env.task_name = task_name

    obs, _ = env.reset()
    rewards = []
    steps_taken = 0
    success = False

    for step in range(1, max_steps + 1):
        # -----------------------------------------------------
        # NEW: The Time's Up Override
        # If this is the absolute last step and we hold shares, 
        # force the agent to SELL so we can calculate final PnL.
        # -----------------------------------------------------
        shares_held = obs[9] 
        if step == max_steps and shares_held > 0:
            print("⏰ [TIME UP] Final step reached! Forcing SELL to close position.")
            action = 2
        else:
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