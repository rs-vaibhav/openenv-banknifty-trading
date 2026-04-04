import os
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
    """A baseline agent that uses an LLM to make trading decisions."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
            api_key=os.environ.get("HF_TOKEN", "dummy_token_to_prevent_crash")
        )
        self.model_name = os.environ.get("MODEL_NAME", "dummy-model")

    def predict(self, obs):
        close_price = obs[0]
        delta = obs[4]
        prompt = f"BankNifty Data: Close={close_price:.2f}, Delta={delta:.2f}. Reply with ONE number: 0 to Hold, 1 to Buy, 2 to Sell."
        
        try:
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
        except Exception:
            return 0  # Fallback to Hold


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