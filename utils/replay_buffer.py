from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


class SumTree:
    """Efficient sum tree data structure for prioritized sampling.

    This implementation uses a binary tree where each leaf contains a priority value
    and each internal node contains the sum of its children's values.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize sum tree.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree structure
        self.data = np.zeros(capacity, dtype=object)  # Store actual experiences
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float, data: Any) -> None:
        """Add new experience with given priority.

        Args:
            priority: Priority value for the experience
            data: Experience data to store
        """
        tree_index = self.data_pointer + self.capacity - 1

        # Store data
        self.data[self.data_pointer] = data

        # Update tree
        self.update(tree_index, priority)

        # Update pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index: int, priority: float) -> None:
        """Update priority of experience at given tree index.

        Args:
            tree_index: Index in the tree structure
            priority: New priority value
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate change up the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> tuple[int, float, Any]:
        """Retrieve experience based on priority value.

        Args:
            value: Random value for proportional sampling

        Returns:
            Tuple of (tree_index, priority, data)
        """
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, select leaf
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # Select child based on priority
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self) -> float:
        """Get total priority (root of tree)."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """GPU-optimized prioritized experience replay buffer.

    This implementation stores experiences on GPU for maximum performance
    and uses prioritized sampling to focus on important experiences.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 100000,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            obs_shape: Shape of observations
            device: Device to store tensors on (preferably GPU)
            alpha: How much prioritization is used (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction factor
            beta_steps: Number of steps to anneal beta to 1.0
            epsilon: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.epsilon = epsilon

        self.tree = SumTree(capacity)

        # Pre-allocate GPU tensors for maximum performance
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.position = 0
        self.size = 0

        self.beta_schedule = np.linspace(beta_start, 1.0, beta_steps)
        self.beta_step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        priority: float | None = None,
    ) -> None:
        """Add experience to replay buffer.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
            priority: Priority for this experience (if None, uses max priority)
        """
        if priority is None:
            if self.tree.size == 0:
                priority = 1.0
            else:
                max_priority = self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.size].max()
                priority = max_priority if max_priority > 0 else 1.0

        self.observations[self.position] = obs.to(self.device)
        self.actions[self.position] = action.to(self.device)
        self.rewards[self.position] = reward.to(self.device)
        self.next_observations[self.position] = next_obs.to(self.device)
        self.dones[self.position] = done.to(self.device)

        experience_data = self.position  # Store index instead of full data
        self.tree.add(priority**self.alpha, experience_data)

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample batch of experiences with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones, weights, indices)
        """
        if self.size == 0:
            msg = 'Cannot sample from empty buffer'
            raise ValueError(msg)

        # Calculate current beta
        beta = self.beta_schedule[min(self.beta_step, self.beta_steps - 1)]
        self.beta_step += 1

        # Sample experiences
        batch_indices = []
        batch_priorities = []

        # Divide priority range into segments
        priority_segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            # Sample from each segment
            segment_start = priority_segment * i
            segment_end = priority_segment * (i + 1)
            sample_value = random.uniform(segment_start, segment_end)

            tree_index, priority, data_index = self.tree.get_leaf(sample_value)
            batch_indices.append(data_index)
            batch_priorities.append(priority)

        # Convert to tensors
        indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        priorities = torch.tensor(batch_priorities, dtype=torch.float32, device=self.device)

        # Calculate importance sampling weights
        sampling_probabilities = priorities / self.tree.total_priority
        weights = (self.size * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize weights

        # Gather experiences from GPU tensors
        batch_obs = self.observations[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_obs = self.next_observations[indices]
        batch_dones = self.dones[indices]

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, weights, indices

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Update priorities of sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        indices = indices.cpu().numpy()
        priorities = priorities.cpu().numpy()

        for idx, priority in zip(indices, priorities):
            # Add epsilon to prevent zero priorities
            priority = abs(priority) + self.epsilon

            # Find tree index for this data index
            tree_index = idx + self.tree.capacity - 1
            self.tree.update(tree_index, priority**self.alpha)

    def get_max_priority(self) -> float:
        """Get maximum priority in the buffer."""
        if self.tree.size == 0:
            return 1.0
        max_priority = self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.size].max()
        return max_priority if max_priority > 0 else 1.0

    def __len__(self) -> int:
        """Get current size of buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size >= batch_size


class UniformReplayBuffer:
    """Standard uniform replay buffer for comparison/fallback.

    This is a simpler implementation without prioritization,
    useful for ablation studies or as a fallback option.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device) -> None:
        """Initialize uniform replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            obs_shape: Shape of observations
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device

        # Pre-allocate GPU tensors
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.position = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        priority: float | None = None,  # Ignored for uniform buffer
    ) -> None:
        """Add experience to replay buffer."""
        # Store experience in GPU tensors
        self.observations[self.position] = obs.to(self.device)
        self.actions[self.position] = action.to(self.device)
        self.rewards[self.position] = reward.to(self.device)
        self.next_observations[self.position] = next_obs.to(self.device)
        self.dones[self.position] = done.to(self.device)

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample batch of experiences uniformly."""
        if self.size == 0:
            msg = 'Cannot sample from empty buffer'
            raise ValueError(msg)

        # Sample random indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Gather experiences
        batch_obs = self.observations[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_obs = self.next_observations[indices]
        batch_dones = self.dones[indices]

        # Return uniform weights
        weights = torch.ones(batch_size, dtype=torch.float32, device=self.device)

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, weights, indices

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """No-op for uniform buffer."""
        pass

    def __len__(self) -> int:
        """Get current size of buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size >= batch_size


def create_replay_buffer(
    buffer_type: str,
    capacity: int,
    obs_shape: tuple[int, ...],
    device: torch.device,
    **kwargs: Any,
) -> PrioritizedReplayBuffer | UniformReplayBuffer:
    """Factory function to create replay buffer.

    Args:
        buffer_type: Type of buffer ('prioritized' or 'uniform')
        capacity: Buffer capacity
        obs_shape: Observation shape
        device: Device to store on
        **kwargs: Additional arguments for prioritized buffer

    Returns:
        Replay buffer instance
    """
    if buffer_type.lower() == 'prioritized':
        return PrioritizedReplayBuffer(capacity, obs_shape, device, **kwargs)
    elif buffer_type.lower() == 'uniform':
        return UniformReplayBuffer(capacity, obs_shape, device)
    else:
        msg = f"Unknown buffer type: {buffer_type}. Use 'prioritized' or 'uniform'"
        raise ValueError(msg)
