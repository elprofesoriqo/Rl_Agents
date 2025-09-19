from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import gymnasium as gym
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """Convert continuous CarRacing actions to discrete action space."""

    def __init__(self, env: Any) -> None:
        """Initialize discrete action wrapper."""
        super().__init__(env)

        self.discrete_actions = [
            [0.0, 0.0, 0.0],  # 0: no-op
            [-0.6, 0.0, 0.0],  # 1: left
            [0.6, 0.0, 0.0],  # 2: right
            [0.0, 1.0, 0.0],  # 3: gas
            [0.0, 0.0, 0.8],  # 4: brake
            [-0.6, 1.0, 0.0],  # 5: left + gas
            [0.6, 1.0, 0.0],  # 6: right + gas
            [-0.3, 1.0, 0.0],  # 7: slight left + gas
            [0.3, 1.0, 0.0],  # 8: slight right + gas
        ]

        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action."""
        if not isinstance(action, (int, np.integer)):
            action = int(action)

        if action < 0 or action >= len(self.discrete_actions):
            action = 0  # Default to no-op

        return np.array(self.discrete_actions[action], dtype=np.float32)


class FramePreprocessor:
    """Preprocess frames for the CarRacing environment."""

    def __init__(self, img_size: tuple[int, int] = (84, 84), grayscale: bool = True, normalize: bool = True) -> None:
        """Initialize frame preprocessor."""
        self.img_size = img_size
        self.grayscale = grayscale
        self.normalize = normalize

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        # Convert to RGB if needed (CarRacing returns RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if self.grayscale:
                # Convert to grayscale using proper weights
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                # Keep RGB but ensure correct format
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize frame
        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)

        # Convert to float32 first
        frame = frame.astype(np.float32)

        # Normalize pixel values
        if self.normalize:
            frame = frame / 255.0

        # Add channel dimension if grayscale
        if self.grayscale and len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=0)  # Shape: (1, H, W)
        elif not self.grayscale and len(frame.shape) == 3:
            frame = frame.transpose(2, 0, 1)  # Shape: (C, H, W)

        return frame


class FrameStackWrapper(gym.ObservationWrapper):
    """Stack multiple frames together."""

    def __init__(self, env: Any, num_stack: int = 4) -> None:
        """Initialize frame stacking wrapper."""
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Update observation space
        low = env.observation_space.low
        high = env.observation_space.high

        if len(low.shape) == 3:  # (C, H, W)
            new_low = np.repeat(low, num_stack, axis=0)
            new_high = np.repeat(high, num_stack, axis=0)
        else:  # (H, W) - grayscale
            new_low = np.stack([low] * num_stack, axis=0)
            new_high = np.stack([high] * num_stack, axis=0)

        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: Any) -> Any:
        """Stack observations."""
        self.frames.append(observation)
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, **kwargs: Any) -> Any:
        """Reset environment and fill frame buffer."""
        observation, info = self.env.reset(**kwargs)

        # Fill frame buffer with initial observation
        for _ in range(self.num_stack):
            self.frames.append(observation)

        return self.observation(observation), info


class RewardShaper:
    """Shape rewards for better training performance."""

    def __init__(self, config: Any) -> None:
        """Initialize reward shaper."""
        self.config = config
        self.prev_reward = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        self.speed_bonus_threshold = 50.0
        self.progress_bonus = 0.1
        self.time_penalty = -0.01
        self.crash_penalty = -10.0
        self.off_track_penalty = -1.0

    def shape_reward(self, reward: float, done: bool, info: dict) -> float:
        """Apply reward shaping."""
        if not self.config.env.reward_shaping:
            return reward

        shaped_reward = reward
        self.step_count += 1

        # Progress bonus for positive rewards
        if reward > 0:
            shaped_reward += self.progress_bonus

        # Time penalty to encourage faster completion
        shaped_reward += self.time_penalty

        # Penalty for going off track (negative rewards in CarRacing)
        if reward < -0.1:
            shaped_reward += self.off_track_penalty

        # Large penalty for episode termination due to poor performance
        if done and self.step_count < 200:  # Ended too early
            shaped_reward += self.crash_penalty

        # Bonus for maintaining speed (inferred from consistent positive rewards)
        if reward > 0 and self.prev_reward > 0:
            shaped_reward += 0.05

        self.prev_reward = reward
        self.total_reward += shaped_reward

        return shaped_reward

    def reset(self) -> None:
        """Reset reward shaper state."""
        self.prev_reward = 0.0
        self.step_count = 0
        self.total_reward = 0.0


class CarRacingEnvironment:
    """Complete CarRacing environment wrapper with all preprocessing."""

    def __init__(self, config: Any) -> None:
        """Initialize CarRacing environment."""
        self.config = config

        # Create base environment
        self.env = gym.make(
            config.env.name,
            render_mode=config.env.render_mode,
            max_episode_steps=config.env.max_episode_steps,
        )

        # Apply discrete action wrapper if needed
        if config.env.discrete_actions:
            self.env = DiscreteActionWrapper(self.env)

        # Set up frame preprocessing
        self.preprocessor = FramePreprocessor(
            img_size=tuple(config.env.img_size),
            grayscale=config.env.grayscale,
            normalize=config.env.normalize,
        )

        # Update observation space based on preprocessing
        obs_shape = (1, *tuple(config.env.img_size)) if config.env.grayscale else (3, *tuple(config.env.img_size))

        if config.env.normalize:
            self.env.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=obs_shape,
                dtype=np.float32,
            )
        else:
            self.env.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=obs_shape,
                dtype=np.uint8,
            )

        # Apply frame stacking
        if config.env.frame_stack > 1:
            self.env = FrameStackWrapper(self.env, config.env.frame_stack)

        # Set up reward shaping
        self.reward_shaper = RewardShaper(config)

        # Frame skipping
        self.frame_skip = config.env.frame_skip

        # Environment state
        self._last_observation = None
        self._episode_step = 0
        self._episode_reward = 0.0

    @property
    def observation_space(self) -> Any:
        """Get observation space."""
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        """Get action space."""
        return self.env.action_space

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Reset environment."""
        # Reset reward shaper
        self.reward_shaper.reset()
        self._episode_step = 0
        self._episode_reward = 0.0

        # Reset base environment
        observation, info = self.env.reset(**kwargs)

        # Preprocess observation
        observation = self.preprocessor.process(observation)
        self._last_observation = observation

        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step environment with frame skipping."""
        total_reward = 0.0

        # Frame skipping
        for _ in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Preprocess observation
        observation = self.preprocessor.process(observation)
        self._last_observation = observation

        # Apply reward shaping
        shaped_reward = self.reward_shaper.shape_reward(
            total_reward,
            terminated or truncated,
            info,
        )

        self._episode_step += 1
        self._episode_reward += shaped_reward

        # Add episode statistics to info
        info.update(
            {
                'episode_step': self._episode_step,
                'episode_reward': self._episode_reward,
                'raw_reward': total_reward,
                'shaped_reward': shaped_reward,
            }
        )

        return observation, shaped_reward, terminated, truncated, info

    def render(self, mode: str = 'human') -> Any:
        """Render environment."""
        return self.env.render()

    def close(self) -> None:
        """Close environment."""
        self.env.close()

    def seed(self, seed: int | None = None) -> Any:
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        if hasattr(self.env, 'reset'):
            # For newer gymnasium versions
            return self.env.reset(seed=seed)
        return None

    def get_episode_info(self) -> dict:
        """Get current episode information."""
        return {
            'episode_step': self._episode_step,
            'episode_reward': self._episode_reward,
            'total_shaped_reward': self.reward_shaper.total_reward,
        }
