import numpy as np
from environment.environment import Environment
from agents.base_agent import Base_Agent

class SARSA_Agent(Base_Agent):
    def __init__(self, env: Environment, epsilon: float = 0.1, gamma: float = 0.9, alpha: float = 0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.tickers = env.tickers

        buy_actions = np.eye(len(self.tickers))
        sell_action = -np.ones((1, len(self.tickers)))
        hold_action = np.zeros((1, len(self.tickers)))
        self.action_templates = np.vstack([buy_actions, sell_action, hold_action]).astype(np.float32)

        self.Q = {}  # dictionnaire des Q-valeurs : (state, action_id) → valeur

    def reset(self):
        pass

    def _state_to_key(self, state):
        prices, balance, shares, _ = state
        prices_diff_key = tuple(
            tuple(round(prices[i+1][t] - prices[i][t], 0) for t in self.tickers)
            for i in range(len(prices) - 1)
        )
        balance_key = (balance[-1] > 0,)
        shares_key = tuple(shares[-1][t] > 0 for t in self.tickers)
        return prices_diff_key + balance_key + shares_key

    def act(self, state):
        state_key = self._state_to_key(state)

        if np.random.rand() < self.epsilon:
            action_id = np.random.randint(len(self.action_templates))
        else:
            q_vals = [self.Q.get((state_key, a), 0.0) for a in range(len(self.action_templates))]
            action_id = np.argmax(q_vals)

        return self.action_templates[action_id], action_id

    def update(self, prev_state, prev_action, reward, next_state, next_action):
        key = (self._state_to_key(prev_state), prev_action)
        next_key = (self._state_to_key(next_state), next_action)

        q_sa = self.Q.get(key, 0.0)
        q_s_next_a_next = self.Q.get(next_key, 0.0)

        self.Q[key] = q_sa + self.alpha * (reward + self.gamma * q_s_next_a_next - q_sa)