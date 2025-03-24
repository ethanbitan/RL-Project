import numpy as np
from environment.environment import Environment
from agents.base_agent import Base_Agent

class MC_Agent(Base_Agent):
    def __init__(self, env: Environment, epsilon: float = 0.1, gamma: float = 1.0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tickers = env.tickers
        self.window_size = env.window_size

        buy_actions = np.eye(len(self.tickers))
        sell_action = -np.ones((1, len(self.tickers)))
        hold_action = np.zeros((1, len(self.tickers)))
        self.action_templates = np.vstack([buy_actions, sell_action, hold_action]).astype(np.float32)

        self.Q = {}           # (state, action_id) → Q-value
        self.returns = {}     # (state, action_id) → list of returns
        self.trajectory = []  # list of (state, action_id, reward)

    def reset(self):
        self.trajectory = []

    def _state_to_key(self, state):
        prices, balance, shares, _ = state
        prices_diff_key = tuple(
            tuple(round(prices[i+1][t] - prices[i][t], 0) for t in self.tickers)
            for i in range(len(prices) - 1)
        )
        balance_key = (balance[-1] > 0,)
        shares_key = tuple(shares[-1][t] > 0 for t in self.tickers)
        print(prices_diff_key + balance_key + shares_key)
        return prices_diff_key + balance_key + shares_key
    
    def act(self, state):
        state_key = self._state_to_key(state)
        if np.random.rand() < self.epsilon:
            action_id = np.random.randint(len(self.action_templates))
        else:
            q_vals = [self.Q.get((state_key, a), 0.0) for a in range(len(self.action_templates))]
            action_id = np.argmax(q_vals)
        return self.action_templates[action_id], action_id

    def update(self):
        # Monte Carlo every-visit
        G = 0
        visited = set()

        for t in reversed(range(len(self.trajectory))):
            state, action_id, reward = self.trajectory[t]
            G = self.gamma * G + reward
            key = (state, action_id)

            # every-visit : on stocke G à chaque passage
            if key not in self.returns:
                self.returns[key] = []
            self.returns[key].append(G)

            # moyenne des retours
            self.Q[key] = np.mean(self.returns[key])

        self.trajectory = []  # Clear trajectory after update

    def remember(self, state, action_id, reward):
        state_key = self._state_to_key(state)
        self.trajectory.append((state_key, action_id, reward))