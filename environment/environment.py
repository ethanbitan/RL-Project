# import numpy as np
# import gym
# from gym import spaces

# class Environment(gym.Env):
#     def __init__(self, data: dict, window_size: int, initial_balance: float, verbose: bool = False):
#         self.history_prices = data
#         self.window_size = window_size
#         self.initial_balance = initial_balance


#         self.current_step = self.window_size
#         self.current_prices = self.history_prices[self.current_step]
#         self.max_steps = len(self.history_prices) - 1
#         self.tickers = list(self.history_prices[0].keys())
#         self.initial_value = self.initial_balance
#         self.initial_shares = {t: 0 for t in self.tickers}

#         # Historiques initialisés avec les valeurs fixes pour les premiers steps
#         self.history_balance = {i: self.initial_balance for i in range(self.current_step + 1)}
#         self.history_shares = {i: self.initial_shares.copy() for i in range(self.current_step + 1)}
#         self.history_value = {i: self.initial_value for i in range(self.current_step + 1)}

#         self.current_balance = self.initial_balance
#         self.current_shares = self.initial_shares.copy()
#         self.current_value = self.initial_value

#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.tickers),))
#         self.observation_dimension = self.window_size * (2 * len(self.tickers) + 2)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dimension,))

#         self.done = False
#         self.verbose = verbose

#     def reset(self):
#         self.current_step = self.window_size
#         self.current_prices = self.history_prices[self.current_step]

#         self.history_balance = {i: self.initial_balance for i in range(self.current_step + 1)}
#         self.history_shares = {i: {t: 0 for t in self.tickers} for i in range(self.current_step + 1)}
#         self.history_value = {i: self.initial_balance for i in range(self.current_step + 1)}

#         self.current_balance = self.initial_balance
#         self.current_shares = {t: 0 for t in self.tickers}
#         self.current_value = self.initial_balance

#         self.done = False

#         if self.verbose:
#             print(f"Step: {self.current_step}")

#         return self._get_state()
    
#     def _get_state(self):
#         start = max(0, self.current_step - self.window_size)
#         end = self.current_step

#         window_prices = [self.history_prices[i] for i in range(start, end)]
#         window_balance = [self.history_balance[i] for i in range(start, end)]
#         window_shares = [self.history_shares[i] for i in range(start, end)]
#         window_value = [self.history_value[i] for i in range(start, end)]

#         return window_prices, window_balance, window_shares, window_value
        
#     def step(self, action: np.ndarray):
        # if self.done:
        #     return self._get_state(), 0, self.done, {}

        # if np.sum(action) > 1.0:
        #     raise ValueError(f"Invalid action: total buy fraction = {np.sum(action):.2f} > 1.0")
        
        # if any([a < -1.0 for a in action]):
        #     raise ValueError(f"Invalid action: sell fraction < -1.0")

#         for i, ticker in enumerate(self.tickers):
#             act = action[i]
#             if act < 0:
#                 shares_to_sell = self.current_shares[ticker] * (-act)
#                 proceeds = shares_to_sell * self.current_prices[ticker]
#                 self.current_balance += proceeds
#                 self.current_shares[ticker] -= shares_to_sell

#             elif act > 0:
#                 amount_to_invest = self.current_balance * act
#                 shares_to_buy = amount_to_invest / self.current_prices[ticker]
#                 cost = shares_to_buy * self.current_prices[ticker]
#                 self.current_balance -= cost
#                 self.current_shares[ticker] += shares_to_buy

#         previous_value = self.history_value[self.current_step]
#         self.current_value = self.current_balance + sum(self.current_shares[t] * self.current_prices[t] for t in self.tickers)
#         reward = self.current_value - previous_value

#         self.current_step += 1
#         self.done = self.current_step >= self.max_steps

#         self.current_prices = self.history_prices[self.current_step]
#         self.history_balance[self.current_step] = self.current_balance
#         self.history_shares[self.current_step] = self.current_shares.copy()
#         self.history_value[self.current_step] = self.current_value

#         if self.verbose:
#             print(f"Step: {self.current_step}")

#         return self._get_state(), reward, self.done, {}


import numpy as np
import gym
from gym import spaces

class Environment(gym.Env):
    def __init__(self, data: dict, window_size: int, initial_balance: float, 
                 train_mode: bool = True, test_split: int = 100, verbose: bool = False):
        
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.verbose = verbose
        self.test_split = test_split
        self.train_mode = train_mode

        self.tickers = list(self.data[0].keys())
        self.max_steps = len(self.data) - 1
        self.train_limit = self.max_steps - test_split  # exclusive

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.tickers),))
        self.observation_dimension = self.window_size * (2 * len(self.tickers) + 2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dimension,))

        self.initial_shares = {t: 0 for t in self.tickers}
        self.reset()

    def reset(self):
        if self.train_mode:
            max_start = self.train_limit - self.window_size - 50  # leave margin for episode length
            self.start_index = np.random.randint(0, max_start)
            self.episode_length = np.random.randint(50, 150)
        else:
            self.start_index = self.train_limit
            self.episode_length = self.test_split - self.window_size

        self.current_step = self.start_index + self.window_size
        self.episode_limit = self.current_step + self.episode_length

        # Reset prices
        self.current_prices = self.data[self.current_step]

        # Reset balances and shares
        self.current_balance = self.initial_balance
        self.current_value = self.initial_balance
        self.current_shares = {t: 0 for t in self.tickers}

        # History initialized for window
        self.history_balance = {i: self.initial_balance for i in range(self.current_step + 1)}
        self.history_value = {i: self.initial_balance for i in range(self.current_step + 1)}
        self.history_shares = {i: {t: 0 for t in self.tickers} for i in range(self.current_step + 1)}

        self.done = False

        if self.verbose:
            print(f"🔁 Reset to step {self.current_step} (start index: {self.start_index}, length: {self.episode_length})")

        return self._get_state()

    def reset_with_range(self, start_index: int, episode_length: int):
        self.start_index = start_index
        self.episode_length = episode_length
        self.current_step = start_index + self.window_size
        self.episode_limit = self.current_step + self.episode_length

        self.current_prices = self.data[self.current_step]
        self.current_balance = self.initial_balance
        self.current_value = self.initial_balance
        self.current_shares = {t: 0 for t in self.tickers}

        self.history_balance = {i: self.initial_balance for i in range(self.current_step + 1)}
        self.history_value = {i: self.initial_balance for i in range(self.current_step + 1)}
        self.history_shares = {i: {t: 0 for t in self.tickers} for i in range(self.current_step + 1)}

        self.done = False
        return self._get_state()

    def _get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step

        prices = [self.data[i] for i in range(start, end)]
        balance = [self.history_balance[i] for i in range(start, end)]
        shares = [self.history_shares[i] for i in range(start, end)]
        value = [self.history_value[i] for i in range(start, end)]

        return prices, balance, shares, value

    def step(self, action: np.ndarray):
        if self.done:
            return self._get_state(), 0, self.done, {}

        if np.sum(action) > 1.0:
            raise ValueError(f"Invalid action: total buy fraction = {np.sum(action):.2f} > 1.0")
        
        if any([a < -1.0 for a in action]):
            raise ValueError(f"Invalid action: sell fraction < -1.0")

        for i, ticker in enumerate(self.tickers):
            act = action[i]
            price = self.current_prices[ticker]

            if act < 0:
                shares_to_sell = self.current_shares[ticker] * (-act)
                proceeds = shares_to_sell * price
                self.current_balance += proceeds
                self.current_shares[ticker] -= shares_to_sell

            elif act > 0:
                invest_amount = self.current_balance * act
                shares_to_buy = invest_amount / price
                self.current_balance -= invest_amount
                self.current_shares[ticker] += shares_to_buy

        previous_value = self.history_value[self.current_step]
        self.current_value = self.current_balance + sum(self.current_shares[t] * self.current_prices[t] for t in self.tickers)
        reward = self.current_value - previous_value

        # Update step and histories
        self.current_step += 1
        self.done = self.current_step >= self.episode_limit or self.current_step >= self.max_steps
        self.current_prices = self.data[self.current_step]

        self.history_balance[self.current_step] = self.current_balance
        self.history_shares[self.current_step] = self.current_shares.copy()
        self.history_value[self.current_step] = self.current_value

        return self._get_state(), reward, self.done, {}