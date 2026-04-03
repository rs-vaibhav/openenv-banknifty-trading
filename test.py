from env import BankNiftyEnv

def test_environment():
    print("--- Initializing BankNifty OpenEnv ---")
    try:
        # If your CSV is named differently, update the data_path here
        env = BankNiftyEnv(data_path='banknifty_historical_data.csv')
    except FileNotFoundError:
        print("ERROR: Could not find the CSV file. Make sure it's in the same folder as this script!")
        return

    print("\n--- Testing Reset() ---")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial Info: {info}")

    print("\n--- Testing 5 Random Steps ---")
    for i in range(5):
        # The agent picks a random action: 0 (Hold), 1 (Buy), or 2 (Sell)
        action = env.action_space.sample()
        
        # Take the step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_name = ["Hold", "Buy", "Sell"][action]
        print(f"Step {i+1} | Action: {action_name:<4} | Reward: {reward:>8.2f} | Net Worth: ₹{info['net_worth']:>9.2f} | Shares: {info['shares_held']}")
        
        if terminated:
            print(f"\nEnvironment terminated early at step {i+1}!")
            break

    print("\n--- Test Completed Successfully! ---")

if __name__ == "__main__":
    test_environment()