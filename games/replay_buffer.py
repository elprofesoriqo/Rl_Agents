import random
from collections import deque
from typing import Deque, Tuple
import numpy as np

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning algorithms."""
    
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


