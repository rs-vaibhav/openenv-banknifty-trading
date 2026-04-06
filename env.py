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
        self.task_name = "hard-risk-adjusted"

    def state(self):
        """Mandatory method required by the OpenEnv Environment base class."""
        return self._get_obs()
    
    def score(self) -> float:
        """
        Official Grader: Returns a normalized score between 0.0 and 1.0 based on the active task.
        """
        net_profit = self.balance - self.initial_balance
        roi_percentage = (net_profit / self.initial_balance) * 100

        # TASK 1: EASY (Just survive and execute valid API calls)
        if self.task_name == "easy-api-compliance":
            return 1.0 # If the code didn't crash and reached the end, perfect score.

        # TASK 2: MEDIUM (Make money, no strict risk rules)
        elif self.task_name == "medium-short-term-roi":
            if net_profit <= 0:
                return 0.0 # Failed to make money
            else:
                return min(1.0, float(0.5 + (roi_percentage * 0.1)))

        # TASK 3: HARD (Make money BUT strictly manage risk)
        else:
            # Hard Failure: If you breach the 5% drawdown limit, you instantly fail.
            if self.max_drawdown > 0.05:
                return 0.0 
                
            # Lost money but survived without breaching drawdown
            if net_profit < 0:
                return 0.2 
                
            # Broke even (Did no harm)
            elif net_profit == 0.0:
                return 0.5 
                
            # Profitable AND kept drawdown < 5% (Success)
            else:
                final_score = 0.5 + (roi_percentage * 0.1) 
                return min(1.0, float(final_score))

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all our new Quant tracking variables
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0  # 0 = Flat, 1 = Long
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.trade_history = []    # Tracks profit/loss of every closed trade
        self.balance_history = []
        
        obs = self._get_obs()
        info = {"net_worth": self.balance, "drawdown": 0.0}
        return obs, info

    def _get_obs(self):
        """Constructs the observation vector for the current step."""
        market_data = self.df.loc[self.current_step, self.features].values.astype(np.float32)
        
        # FIXED: Pass 'self.position' instead of the dead 'self.shares_held' variable!
        account_info = np.array([self.balance, self.position], dtype=np.float32) 
        
        return np.concatenate((market_data, account_info))
        
    def _get_info(self):
        """Returns extra diagnostic info (not used for agent training, just tracking)."""
        return {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.position # <--- Fixed here too
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
                self.trade_history.append(profit)

        # -------- QUANT RISK MANAGEMENT --------
        # 1. Hard Stop Loss (2%)
        if self.position == 1:
            loss_pct = (price - self.entry_price) / self.entry_price
            if loss_pct < -0.02:
                loss = price - self.entry_price
                self.balance += loss
                self.position = 0  # Forced liquidation to protect capital
                reward += loss   
                self.trade_history.append(loss)  # Penalize the agent for hitting the stop loss

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
        
        if terminated:
            # --- FORCE LIQUIDATION ON THE FINAL BELL ---
            # If the episode ends and the agent is still holding the asset, sell it.
            if self.position == 1:
                final_price = self.df.iloc[self.current_step]["close"]
                profit = final_price - self.entry_price
                self.balance += profit
                reward += profit  # Finally give the reward!
                self.position = 0
                self.trade_history.append(profit)
                
            self.print_backtest_report()
            
        truncated = False
        
        obs = self._get_obs()

        info = {
            "net_worth": self.balance, 
            "drawdown": drawdown,
            "score": self.score() if terminated else 0.0,
            "success": self.score() >= 0.5 if terminated else False
        }
        self.balance_history.append(self.balance)

        return obs, float(reward), terminated, truncated, info
    
    def print_backtest_report(self):
        print("\n" + "="*50)
        print("📊 BACKTEST SUMMARY & QUANT METRICS")
        print("="*50)

        # 1. Win Rate
        winning_trades = sum(1 for p in self.trade_history if p > 0)
        total_trades = len(self.trade_history)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # 2. Net Return
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100

        # 3. Sharpe Ratio
        if len(self.balance_history) > 1:
            # Calculate step-by-step percentage returns
            returns = np.diff(self.balance_history) / self.balance_history[:-1]
            if np.std(returns) != 0:
                # Calculate Sharpe (assuming approx 25 steps per day, annualized)
                sharpe = np.mean(returns) / np.std(returns)
                sharpe_ratio = sharpe * np.sqrt(252 * 25) 
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        print(f"Total Trades Taken : {total_trades}")
        print(f"Win Rate           : {win_rate:.2f}% ({winning_trades} Wins, {total_trades - winning_trades} Losses)")
        print(f"Net Return         : {total_return:.2f}%")
        print(f"Max Drawdown       : {self.max_drawdown * 100:.2f}%")
        print(f"Sharpe Ratio (Est) : {sharpe_ratio:.2f}")
        print("="*50 + "\n")




 