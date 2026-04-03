import numpy as np
import pandas as pd
from gymnasium import spaces
from openenv.env import Env

class BankNiftyEnv(Env):
    def __init__(self, data_path='banknifty_historical_data.csv', initial_balance=100000.0):
        super().__init__()
        
        # 1. Load and prep data
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna().reset_index(drop=True)
        
        # These are the columns the AI agent gets to "look at"
        self.features = [
            'close', 'volume', 'oi_put', 'oi_call', 
            'delta_atm', 'gamma_atm', 'theta_atm', 'vega_atm'
        ]
        
        # 2. Define Spaces
        # Action Space: 0 = Hold, 1 = Buy 1 Share, 2 = Sell 1 Share
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: The 8 market features + our current balance + shares held
        obs_shape = len(self.features) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_shape,), 
            dtype=np.float32
        )
        
        # 3. Environment State Variables
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_steps = len(self.df) - 1

    def reset(self, seed=None, options=None):
        """Resets the environment to the starting state."""
        # Removed super().reset() as openenv.Env does not implement it
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """Constructs the observation vector for the current step."""
        market_data = self.df.loc[self.current_step, self.features].values.astype(np.float32)
        account_info = np.array([self.balance, self.shares_held], dtype=np.float32)
        return np.concatenate((market_data, account_info))
        
    def _get_info(self):
        """Returns extra diagnostic info (not used for agent training, just tracking)."""
        return {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held
        }

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1
        
        # Check if we hit the end of the historical data
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if terminated:
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()

        current_price = self.df.loc[self.current_step, 'close']
        prev_net_worth = self.net_worth
        
        # Execute Action Logic
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
        # Note: Action 0 (Hold) requires no logic

        # Update portfolio value
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Reward Function: Change in net worth
        reward = self.net_worth - prev_net_worth
        
        # Ruin condition: If the agent loses 50% of the portfolio, end the episode early
        if self.net_worth <= self.initial_balance * 0.5:
            terminated = True
            reward -= 10000.0  # Massive penalty for blowing up the account

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()