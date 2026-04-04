import numpy as np
import pandas as pd
from gymnasium import spaces
from openenv.core import Environment

class BankNiftyEnv(Environment):
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

    def state(self):
        """Mandatory method required by the OpenEnv Environment base class."""
        return self._get_obs()

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all our new Quant tracking variables
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0  # 0 = Flat, 1 = Long
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        
        obs = self._get_obs()
        info = {"net_worth": self.balance, "drawdown": 0.0}
        return obs, info

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
        # We assume standard integer actions from the LLM (0=Hold, 1=Buy, 2=Sell)
        price = self.df.iloc[self.current_step]["close"]
        reward = 0.0

        # -------- ACTION LOGIC --------
        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = price

        elif action == 2:  # SELL
            if self.position == 1:
                profit = price - self.entry_price
                self.balance += profit
                reward += profit  # Reward the agent for realized profit
                self.position = 0

        # -------- QUANT RISK MANAGEMENT --------
        # 1. Hard Stop Loss (2%)
        if self.position == 1:
            loss_pct = (price - self.entry_price) / self.entry_price
            if loss_pct < -0.02:
                loss = price - self.entry_price
                self.balance += loss
                self.position = 0  # Forced liquidation to protect capital
                reward += loss     # Penalize the agent for hitting the stop loss

        # 2. Dynamic Drawdown Tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # 3. Drawdown Penalty (Avoid > 5% as per the rules)
        if drawdown > 0.05:
            reward -= 10.0  # Heavy algorithmic penalty for risking ruin

        # -------- NEXT STEP --------
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        obs = self._get_obs()
        info = {"net_worth": self.balance, "drawdown": drawdown}

        # MANDATORY: Return the standard Gymnasium tuple so the web server doesn't crash!
        return obs, float(reward), terminated, truncated, info