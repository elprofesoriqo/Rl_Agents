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
            [0.0, 0.0, 0.0],  # 0: Do-Nothing (coast)
            [-1.0, 0.0, 0.0],  # 1: Turn-Left (hard left)
            [1.0, 0.0, 0.0],  # 2: Turn-Right (hard right)
            [0.0, 1.0, 0.0],  # 3: Accelerate (full throttle)
            [0.0, 0.0, 0.8],  # 4: Brake (hard brake)
        ]

        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action."""
        if not isinstance(action, (int, np.integer)):
            action = int(action)

        if action < 0 or action >= len(self.discrete_actions):
            action = 0

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
        if frame.shape[0] == 96:
            frame = frame[:84, :, :]

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32)

        if self.normalize:
            frame = frame / 255.0

        if self.grayscale and len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=0)
        elif not self.grayscale and len(frame.shape) == 3:
            frame = frame.transpose(2, 0, 1)

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
    """Reward shaper for training performance."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.prev_reward = 0.0
        self.step_count = 0
        self.total_reward = 0.0

        try:
            from utils.reward_shaping import create_reward_shaper

            shaper_type = getattr(config.env, 'reward_shaper_type', 'basic')
            self.advanced_shaper = create_reward_shaper(config, shaper_type)
            self.use_advanced = True
        except ImportError:
            self.use_advanced = False
            self.speed_bonus_threshold = 50.0
            self.progress_bonus = 0.1
            self.time_penalty = -0.01
            self.crash_penalty = -10.0
            self.off_track_penalty = -1.0

    def shape_reward(
        self, reward: float, done: bool, info: dict, observation: np.ndarray = None, action: int = None
    ) -> tuple[float, dict]:
        """Apply reward shaping."""
        if not self.config.env.reward_shaping:
            return reward, {}

        if self.use_advanced and observation is not None and action is not None:
            shaped_reward, shaping_components = self.advanced_shaper.shape_reward(
                reward, observation, action, info, done
            )
            self.total_reward += shaped_reward
            return shaped_reward, shaping_components

        shaped_reward = reward
        self.step_count += 1
        shaping_components = {'base_reward': reward}

        if reward > 0:
            progress_bonus = self.progress_bonus
            shaped_reward += progress_bonus
            shaping_components['progress_bonus'] = progress_bonus

        time_penalty = self.time_penalty
        shaped_reward += time_penalty
        shaping_components['time_penalty'] = time_penalty

        if reward < -0.1:
            off_track_penalty = self.off_track_penalty
            shaped_reward += off_track_penalty
            shaping_components['off_track_penalty'] = off_track_penalty

        if done and self.step_count < 200:
            crash_penalty = self.crash_penalty
            shaped_reward += crash_penalty
            shaping_components['crash_penalty'] = crash_penalty

        if reward > 0 and self.prev_reward > 0:
            speed_bonus = 0.05
            shaped_reward += speed_bonus
            shaping_components['speed_bonus'] = speed_bonus

        self.prev_reward = reward
        self.total_reward += shaped_reward
        shaping_components['total_shaping'] = shaped_reward - reward

        return shaped_reward, shaping_components

    def reset(self) -> None:
        """Reset reward shaper state."""
        self.prev_reward = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        if self.use_advanced:
            self.advanced_shaper.reset()


class CarRacingEnvironment:
    """CarRacing environment wrapper with preprocessing."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.env_id = getattr(config, '_env_id', 0)
        render_mode = config.env.render_mode if self.env_id == 0 else None

        self.env = gym.make(
            config.env.name,
            render_mode=render_mode,
            max_episode_steps=config.env.max_episode_steps,
        )

        if config.env.discrete_actions:
            self.env = DiscreteActionWrapper(self.env)

        class PreprocessingWrapper(gym.ObservationWrapper):
            def __init__(self, env, preprocessor):
                super().__init__(env)
                self.preprocessor = preprocessor
                obs_shape = (
                    (1, *tuple(config.env.img_size)) if config.env.grayscale else (3, *tuple(config.env.img_size))
                )

                if config.env.normalize:
                    self.observation_space = gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=obs_shape,
                        dtype=np.float32,
                    )
                else:
                    self.observation_space = gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=obs_shape,
                        dtype=np.uint8,
                    )

            def observation(self, observation):
                return self.preprocessor.process(observation)

        self.preprocessor = FramePreprocessor(
            img_size=tuple(config.env.img_size),
            grayscale=config.env.grayscale,
            normalize=config.env.normalize,
        )

        self.env = PreprocessingWrapper(self.env, self.preprocessor)

        if config.env.frame_stack > 1:
            self.env = FrameStackWrapper(self.env, config.env.frame_stack)

        self.reward_shaper = RewardShaper(config)
        self.frame_skip = config.env.frame_skip
        self._last_observation = None
        self._episode_step = 0
        self._episode_reward = 0.0
        self._off_track_steps = 0
        self._max_off_track_steps = 100

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Reset environment."""
        self.reward_shaper.reset()
        self._episode_step = 0
        self._episode_reward = 0.0
        self._off_track_steps = 0
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step environment with frame skipping."""
        total_reward = 0.0

        for _ in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        self._last_observation = observation
        custom_terminated = terminated
        custom_truncated = truncated

        if total_reward < -0.1:
            self._off_track_steps += 1
        else:
            self._off_track_steps = 0

        if self._off_track_steps >= self._max_off_track_steps:
            custom_terminated = True
            info['termination_reason'] = 'off_track_timeout'
        elif total_reward < -5.0:
            custom_terminated = True
            info['termination_reason'] = 'severe_off_track'
        elif self._episode_step >= self.config.env.max_episode_steps:
            custom_truncated = True
            info['termination_reason'] = 'max_steps'

        clipped_reward = np.clip(total_reward, a_min=None, a_max=1.0)

        shaped_reward, shaping_components = self.reward_shaper.shape_reward(
            clipped_reward,
            custom_terminated or custom_truncated,
            info,
            observation,
            action,
        )

        info.update({f'shaping/{key}': value for key, value in shaping_components.items()})
        self._episode_step += 1
        self._episode_reward += shaped_reward

        info.update(
            {
                'episode_step': self._episode_step,
                'episode_reward': self._episode_reward,
                'raw_reward': total_reward,
                'shaped_reward': shaped_reward,
            }
        )

        return observation, shaped_reward, custom_terminated, custom_truncated, info

    def close(self) -> None:
        """Close environment."""
        self.env.close()
