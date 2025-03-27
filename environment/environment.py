import numpy as np
import gym
from gym import spaces

class Environment(gym.Env):
    def __init__(self, data: dict, window_size: int, initial_balance: float, verbose: bool = False):
        self.window_size = window_size
        self.current_step = self.window_size
        self.history_prices = data
        self.current_prices = self.history_prices[self.current_step]
        self.max_steps = len(self.history_prices) - 1 - self.window_size
        self.tickers = list(self.history_prices[0].keys())

        self.initial_balance = initial_balance
        self.initial_value = self.initial_balance
        self.initial_shares = {t: 0 for t in self.tickers}

        # Historiques initialisés avec les valeurs fixes pour les premiers steps
        self.history_balance = {i: self.initial_balance for i in range(self.current_step + 1)}
        self.history_shares = {i: self.initial_shares.copy() for i in range(self.current_step + 1)}
        self.history_value = {i: self.initial_value for i in range(self.current_step + 1)}

        self.current_balance = self.initial_balance
        self.current_shares = self.initial_shares.copy()
        self.current_value = self.initial_value

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.tickers),))
        self.observation_dimension = self.window_size * (2 * len(self.tickers) + 2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dimension,))

        self.done = False
        self.verbose = verbose

    def reset(self):
        self.current_step = self.window_size
        self.current_prices = self.history_prices[self.current_step]

        self.history_balance = {i: self.initial_balance for i in range(self.current_step + 1)}
        self.history_shares = {i: {t: 0 for t in self.tickers} for i in range(self.current_step + 1)}
        self.history_value = {i: self.initial_balance for i in range(self.current_step + 1)}

        self.current_balance = self.initial_balance
        self.current_shares = {t: 0 for t in self.tickers}
        self.current_value = self.initial_balance

        self.done = False

        if self.verbose:
            print(f"\n📈 Step: {self.current_step}")
            print(f"🟦 Prices: {[round(self.current_prices[t], 2) for t in self.tickers]}")
            print(f"💰 Balance: {self.current_balance:.2f}")
            print(f"📊 Shares: { {t: round(self.current_shares[t], 2) for t in self.tickers} }")
            print(f"📦 Value: {self.current_value:.2f}")

        return self._get_state()
    
    def _get_state(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step

        window_prices = [self.history_prices[i] for i in range(start, end)]
        window_balance = [self.history_balance[i] for i in range(start, end)]
        window_shares = [self.history_shares[i] for i in range(start, end)]
        window_value = [self.history_value[i] for i in range(start, end)]

        return window_prices, window_balance, window_shares, window_value
        
    def step(self, action: np.ndarray):
        if self.done:
            return self._get_state(), 0, self.done, {}

        if np.sum(action) > 1.0:
            raise ValueError(f"Invalid action: total buy fraction = {np.sum(action):.2f} > 1.0")
        
        if any([a < -1.0 for a in action]):
            raise ValueError(f"Invalid action: sell fraction < -1.0")

        for i, ticker in enumerate(self.tickers):
            act = action[i]
            if act < 0:
                shares_to_sell = self.current_shares[ticker] * (-act)
                proceeds = shares_to_sell * self.current_prices[ticker]
                self.current_balance += proceeds
                self.current_shares[ticker] -= shares_to_sell

            elif act > 0:
                amount_to_invest = self.current_balance * act
                shares_to_buy = amount_to_invest / self.current_prices[ticker]
                cost = shares_to_buy * self.current_prices[ticker]
                self.current_balance -= cost
                self.current_shares[ticker] += shares_to_buy

        previous_value = self.history_value[self.current_step]
        self.current_value = self.current_balance + sum(self.current_shares[t] * self.current_prices[t] for t in self.tickers)
        reward = self.current_value - previous_value

        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        self.current_prices = self.history_prices[self.current_step]
        self.history_balance[self.current_step] = self.current_balance
        self.history_shares[self.current_step] = self.current_shares.copy()
        self.history_value[self.current_step] = self.current_value

        if self.verbose:
            print(f"\n📈 Step: {self.current_step}")
            print(f"🟦 Prices: {[round(self.current_prices[t], 2) for t in self.tickers]}")
            print(f"💰 Balance: {self.current_balance:.2f}")
            print(f"📊 Shares: { {t: round(self.current_shares[t], 2) for t in self.tickers} }")
            print(f"📦 Value: {self.current_value:.2f}")
            print(f"🔄 Reward: {reward:.2f}")
            print(f"🎯 Action taken: {np.round(action, 2)}")

        return self._get_state(), reward, self.done, {}