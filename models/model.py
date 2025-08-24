from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseAgent(ABC):
    """
    Base class for all RL agents.
    """
    
    @abstractmethod
    def train(self) -> None:
        """Main training loop"""
        pass

    @abstractmethod
    def act(self, state) -> int:
        """Select action given current state"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint"""
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate agent performance"""
        pass
