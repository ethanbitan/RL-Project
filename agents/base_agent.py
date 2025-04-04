from abc import ABC, abstractmethod
import numpy as np

class Base_Agent(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> tuple[np.ndarray, int | None]:
        pass

    @abstractmethod
    def update(self, *args):
        pass

    def train(self, mode: bool = True):
        pass