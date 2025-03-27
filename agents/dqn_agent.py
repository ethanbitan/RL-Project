import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import Base_Agent
from annexe.replay_buffer import ReplayBuffer
from annexe.q_network import QNetwork
from environment.environment import Environment

class DQN_Agent(Base_Agent):
    def __init__(
        self,
        env: Environment,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = 0.95,
        alpha: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 100
    ):
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.n_actions = len(env.tickers) + 2  # buy each, sell all, hold

        # ε-greedy parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Learning & RL parameters
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0

        # Devices & models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_q_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # Action template (one-hot)
        buy_actions = np.eye(len(env.tickers), dtype=np.float32)
        sell_action = -np.ones((1, len(env.tickers)), dtype=np.float32)
        hold_action = np.zeros((1, len(env.tickers)), dtype=np.float32)
        self.action_templates = np.vstack([buy_actions, sell_action, hold_action])

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, state_dim=self.state_dim)

    def reset(self):
        pass

    def act(self, state: np.ndarray) -> tuple[np.ndarray, int]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if np.random.rand() < self.epsilon:
            action_id = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            action_id = torch.argmax(q_values, dim=1).item()

        return self.action_templates[action_id], action_id

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # r + γ max Q′(s′, a′)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)