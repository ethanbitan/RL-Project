import numpy as np
from environment.environment import Environment
from agents.base_agent import Base_Agent

class MC_Agent(Base_Agent):
    def __init__(self, env: Environment, epsilon: float = 0.1, gamma: float = 1.0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tickers = env.tickers
        self.n_actions = len(self.tickers) + 2  # buy one asset + sell all + hold
        self.action_templates = np.vstack([
            np.eye(len(self.tickers)),
            -np.ones((1, len(self.tickers))),
            np.zeros((1, len(self.tickers)))
        ]).astype(np.float32)

        self.Q = {}           # (state, action_id) → Q-value
        self.returns = {}     # (state, action_id) → list of returns
        self.trajectory = []  # list of (state, action_id, reward)

    def _state_to_key(self, state):
        # Convertit un état complexe en clé hashable (tuple)
        # Ici on simplifie en flattenant tous les éléments en un seul vecteur
        prices, balance, shares, value = state
        flat = []
        for d in prices + balance + shares + value:
            if isinstance(d, dict):
                flat.extend(list(d.values()))
            else:
                flat.extend(d if isinstance(d, list) else [d])
        return tuple(np.round(flat, 2))  # On arrondit pour limiter les variations

    def act(self, state):
        state_key = self._state_to_key(state)
        if np.random.rand() < self.epsilon:
            action_id = np.random.randint(self.n_actions)
        else:
            q_vals = [self.Q.get((state_key, a), 0.0) for a in range(self.n_actions)]
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

    def reset(self):
        self.trajectory = []