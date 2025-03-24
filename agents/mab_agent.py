import numpy as np
from environment.environment import Environment
from agents.base_agent import Base_Agent

class MAB_Agent(Base_Agent):
    def __init__(self, env: Environment, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.tickers = env.tickers

        buy_actions = np.eye(len(self.tickers))
        sell_action = -np.ones((1, len(self.tickers)))
        hold_action = np.zeros((1, len(self.tickers)))
        self.action_templates = np.vstack([buy_actions, sell_action, hold_action]).astype(np.float32)

        self.Q = np.zeros(len(self.action_templates))
        self.N = np.zeros(len(self.action_templates))

    def reset(self):
        self.Q = np.zeros(len(self.action_templates))
        self.N = np.zeros(len(self.action_templates))

    def select_action_id(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_templates))
        else:
            return np.argmax(self.Q)

    def act(self, state=None):
        action_id = self.select_action_id()
        action = self.action_templates[action_id]
        return action, action_id

    def update(self, action_id, reward):
        self.N[action_id] += 1
        alpha = 1 / self.N[action_id]
        self.Q[action_id] += alpha * (reward - self.Q[action_id])