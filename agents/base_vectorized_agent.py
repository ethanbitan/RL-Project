from abc import ABC, abstractmethod
import numpy as np
from agents.base_agent import Base_Agent

class Base_VectorizedAgent(Base_Agent):
    def __init__(self, env):
        self.env = env
        self.tickers = env.tickers

    def flatten_state(self, state) -> np.ndarray:
        prices, balance, shares, value = state
        flat = []

        for day_prices in prices:
            flat.extend([day_prices[t] for t in self.tickers])
        for b in balance:
            flat.append(b)
        for day_shares in shares:
            flat.extend([day_shares[t] for t in self.tickers])
        for v in value:
            flat.append(v)

        return np.array(flat, dtype=np.float32)