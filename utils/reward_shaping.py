from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


class CarRacingRewardShaper:
    """Reward shaping for CarRacing environment.

    Implements multiple reward shaping techniques based on racing game research:
    - Progress-based rewards for forward movement
    - Speed-based rewards for maintaining velocity
    - Track position rewards for staying on track
    - Steering smoothness rewards for stable driving
    - Lap completion bonuses
    - Penalty shaping for crashes and off-track behavior
    """

    def __init__(self, config: Any) -> None:
        """Initialize reward shaper.

        Args:
            config: Configuration object with reward shaping parameters
        """
        self.config = config

        # Reward shaping weights
        self.progress_weight = getattr(config.env, 'progress_reward_weight', 1.0)
        self.speed_weight = getattr(config.env, 'speed_reward_weight', 0.1)
        self.position_weight = getattr(config.env, 'position_reward_weight', 0.5)
        self.steering_weight = getattr(config.env, 'steering_reward_weight', 0.1)
        self.lap_bonus = getattr(config.env, 'lap_completion_bonus', 100.0)
        self.crash_penalty = getattr(config.env, 'crash_penalty', -10.0)
        self.off_track_penalty = getattr(config.env, 'off_track_penalty', -1.0)

        # State tracking for progress calculation
        self.last_position = None
        self.last_speed = 0.0
        self.last_steering = 0.0
        self.total_progress = 0.0
        self.episode_start_time = 0
        self.tiles_visited = set()
        self.consecutive_off_track = 0
        self.max_consecutive_off_track = 50  # Reset after this many off-track steps

        # Track geometry (estimated from CarRacing environment)
        self.track_width = 40.0  # Approximate track width in pixels
        self.track_center_tolerance = 15.0  # Tolerance for being "on center"

        # Speed thresholds
        self.min_speed_threshold = 0.1  # Below this is considered stopped
        self.optimal_speed_range = (20.0, 50.0)  # Optimal speed range

        # Progress tracking
        self.progress_smoothing = 0.9  # Exponential smoothing for progress
        self.smoothed_progress = 0.0

    def reset(self) -> None:
        """Reset state for new episode."""
        self.last_position = None
        self.last_speed = 0.0
        self.last_steering = 0.0
        self.total_progress = 0.0
        self.episode_start_time = 0
        self.tiles_visited.clear()
        self.consecutive_off_track = 0
        self.smoothed_progress = 0.0

    def shape_reward(
        self, base_reward: float, observation: np.ndarray, action: int | np.ndarray, info: dict[str, Any], done: bool
    ) -> tuple[float, dict[str, float]]:
        """Apply reward shaping to base reward.

        Args:
            base_reward: Original reward from environment
            observation: Current observation
            action: Action taken
            info: Environment info dictionary
            done: Whether episode is done

        Returns:
            Tuple of (shaped_reward, shaping_components)
        """
        shaping_components = {}
        total_shaping = 0.0

        # Extract information from observation and info
        current_speed = self._extract_speed(info)
        current_position = self._extract_position(info)
        on_track = self._is_on_track(observation, info)
        steering_action = self._extract_steering(action)

        # 1. Progress-based reward
        progress_reward = self._compute_progress_reward(current_position)
        shaping_components['progress'] = progress_reward
        total_shaping += progress_reward * self.progress_weight

        # 2. Speed-based reward
        speed_reward = self._compute_speed_reward(current_speed, on_track)
        shaping_components['speed'] = speed_reward
        total_shaping += speed_reward * self.speed_weight

        # 3. Track position reward
        position_reward = self._compute_position_reward(observation, on_track)
        shaping_components['position'] = position_reward
        total_shaping += position_reward * self.position_weight

        # 4. Steering smoothness reward
        steering_reward = self._compute_steering_reward(steering_action)
        shaping_components['steering'] = steering_reward
        total_shaping += steering_reward * self.steering_weight

        # 5. Special bonuses and penalties
        bonus_penalty = self._compute_bonuses_penalties(info, done, on_track)
        shaping_components['bonus_penalty'] = bonus_penalty
        total_shaping += bonus_penalty

        # 6. Exploration reward (encourage visiting new track tiles)
        exploration_reward = self._compute_exploration_reward(current_position)
        shaping_components['exploration'] = exploration_reward
        total_shaping += exploration_reward * 0.05  # Small weight for exploration

        # Update state for next step
        self._update_state(current_position, current_speed, steering_action, on_track)

        # Combine base reward with shaping
        shaped_reward = base_reward + total_shaping
        shaping_components['total_shaping'] = total_shaping
        shaping_components['base_reward'] = base_reward

        return shaped_reward, shaping_components

    def _extract_speed(self, info: dict[str, Any]) -> float:
        """Extract speed from environment info."""
        # Try to get speed from info, fallback to estimation
        if 'speed' in info:
            return float(info['speed'])

        # Estimate speed from position change if available
        if self.last_position is not None and 'position' in info:
            current_pos = np.array(info['position'])
            last_pos = np.array(self.last_position)
            speed = np.linalg.norm(current_pos - last_pos)
            return float(speed)

        return 0.0

    def _extract_position(self, info: dict[str, Any]) -> np.ndarray | None:
        """Extract position from environment info."""
        if 'position' in info:
            return np.array(info['position'])
        return None

    def _extract_steering(self, action: int | np.ndarray) -> float:
        """Extract steering component from action."""
        if isinstance(action, (int, np.integer)):
            steering_map = {
                0: 0.0,  # no-op
                1: -0.6,  # left
                2: 0.6,  # right
                3: 0.0,  # gas
                4: 0.0,  # brake
                5: -0.6,  # left + gas
                6: 0.6,  # right + gas
                7: -0.3,  # slight left + gas
                8: 0.3,  # slight right + gas
            }
            return steering_map.get(int(action), 0.0)
        elif isinstance(action, np.ndarray) and len(action) >= 1:
            return float(action[0])  # First component is steering
        return 0.0

    def _is_on_track(self, observation: np.ndarray, info: dict[str, Any]) -> bool:
        """Determine if agent is on track based on observation."""
        # Method 1: Use info if available
        if 'on_track' in info:
            return bool(info['on_track'])

        # Method 2: Analyze observation to detect track
        if len(observation.shape) >= 2:
            # Convert to grayscale if needed
            if len(observation.shape) == 3:
                if observation.shape[0] in [3, 4]:  # Channel first
                    obs = observation[0] if observation.shape[0] == 4 else np.mean(observation, axis=0)
                else:  # Channel last
                    obs = observation[:, :, 0] if observation.shape[2] == 4 else np.mean(observation, axis=2)
            else:
                obs = observation

            # Check center region for track (green pixels typically indicate track)
            center_h, center_w = obs.shape[0] // 2, obs.shape[1] // 2
            center_region = obs[center_h - 10 : center_h + 10, center_w - 10 : center_w + 10]

            # Normalize to 0-1 if needed
            if center_region.max() > 1.0:
                center_region = center_region / 255.0

            # Track is typically darker than grass, lighter than off-road
            track_pixels = np.sum((center_region > 0.3) & (center_region < 0.7))
            total_pixels = center_region.size

            return track_pixels / total_pixels > 0.5

        return True  # Default to on-track if can't determine

    def _compute_progress_reward(self, current_position: np.ndarray | None) -> float:
        """Compute reward based on forward progress."""
        if current_position is None or self.last_position is None:
            return 0.0

        # Calculate distance moved
        distance_moved = np.linalg.norm(current_position - self.last_position)

        # Only reward forward progress (not just any movement)
        # This is a simplified version - in practice, you'd want to use track waypoints
        progress = min(distance_moved, 5.0)  # Cap to prevent exploitation

        # Apply exponential smoothing
        self.smoothed_progress = (
            self.progress_smoothing * self.smoothed_progress + (1 - self.progress_smoothing) * progress
        )

        return self.smoothed_progress

    def _compute_speed_reward(self, current_speed: float, on_track: bool) -> float:
        """Compute reward based on speed maintenance."""
        if not on_track:
            # Penalize high speed when off track
            return -abs(current_speed) * 0.1

        # Reward optimal speed range
        if self.optimal_speed_range[0] <= current_speed <= self.optimal_speed_range[1]:
            return 1.0
        elif current_speed < self.min_speed_threshold:
            # Penalize being stopped
            return -0.5
        elif current_speed > self.optimal_speed_range[1]:
            # Penalize excessive speed (but less than being stopped)
            excess = current_speed - self.optimal_speed_range[1]
            return -0.1 * excess
        else:
            # Below optimal range but moving
            return 0.5 * (current_speed / self.optimal_speed_range[0])

    def _compute_position_reward(self, observation: np.ndarray, on_track: bool) -> float:
        """Compute reward based on track position."""
        if not on_track:
            return -1.0

        # Reward staying near track center
        # This is a simplified version - ideally you'd compute distance from track centerline
        return 0.5  # Base reward for being on track

    def _compute_steering_reward(self, steering_action: float) -> float:
        """Compute reward for smooth steering."""
        # Penalize excessive steering changes
        steering_change = abs(steering_action - self.last_steering)

        # Reward smooth steering
        if steering_change < 0.1:
            return 0.1
        elif steering_change > 0.5:
            return -0.2
        else:
            return 0.0

    def _compute_bonuses_penalties(self, info: dict[str, Any], done: bool, on_track: bool) -> float:
        """Compute special bonuses and penalties."""
        bonus_penalty = 0.0

        # Lap completion bonus
        if 'lap_completed' in info and info['lap_completed']:
            bonus_penalty += self.lap_bonus

        # Crash penalty
        if done and 'crash' in info and info['crash']:
            bonus_penalty += self.crash_penalty

        # Off-track penalty
        if not on_track:
            self.consecutive_off_track += 1
            # Increasing penalty for staying off track
            penalty = self.off_track_penalty * (1 + self.consecutive_off_track * 0.1)
            bonus_penalty += penalty

            # Reset if off track too long
            if self.consecutive_off_track > self.max_consecutive_off_track:
                bonus_penalty += self.crash_penalty  # Treat as crash
        else:
            self.consecutive_off_track = 0

        return bonus_penalty

    def _compute_exploration_reward(self, current_position: np.ndarray | None) -> float:
        """Compute reward for exploring new areas of the track."""
        if current_position is None:
            return 0.0

        # Discretize position to track tiles
        tile_x = int(current_position[0] // 10)  # 10-unit tiles
        tile_y = int(current_position[1] // 10)
        tile = (tile_x, tile_y)

        if tile not in self.tiles_visited:
            self.tiles_visited.add(tile)
            return 1.0  # Reward for visiting new tile

        return 0.0

    def _update_state(
        self, current_position: np.ndarray | None, current_speed: float, steering_action: float, on_track: bool
    ) -> None:
        """Update internal state for next step."""
        self.last_position = current_position
        self.last_speed = current_speed
        self.last_steering = steering_action

        if current_position is not None:
            if self.last_position is not None:
                self.total_progress += np.linalg.norm(current_position - self.last_position)


class AdaptiveRewardShaper(CarRacingRewardShaper):
    """Adaptive reward shaper that adjusts weights based on training progress.

    This version dynamically adjusts reward shaping weights based on the agent's
    performance, gradually reducing shaping as the agent improves.
    """

    def __init__(self, config: Any) -> None:
        """Initialize adaptive reward shaper."""
        super().__init__(config)

        # Adaptive parameters
        self.initial_weights = {
            'progress': self.progress_weight,
            'speed': self.speed_weight,
            'position': self.position_weight,
            'steering': self.steering_weight,
        }

        self.min_weight_factor = 0.1  # Minimum weight factor
        self.adaptation_episodes = getattr(config.env, 'adaptation_episodes', 1000)
        self.current_episode = 0
        self.performance_history = []
        self.performance_window = 100  # Episodes to consider for performance

    def update_episode(self, episode_reward: float) -> None:
        """Update adaptive weights based on episode performance.

        Args:
            episode_reward: Total reward achieved in the episode
        """
        self.current_episode += 1
        self.performance_history.append(episode_reward)

        # Keep only recent performance
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)

        # Calculate adaptation factor
        if self.current_episode > self.performance_window:
            # Use performance improvement to adjust weights
            recent_performance = np.mean(self.performance_history[-50:])
            early_performance = np.mean(self.performance_history[:50])

            if recent_performance > early_performance:
                # Agent is improving, reduce shaping
                progress_factor = max(self.min_weight_factor, 1.0 - (self.current_episode / self.adaptation_episodes))
            else:
                # Agent struggling, maintain shaping
                progress_factor = 1.0

            # Update weights
            self.progress_weight = self.initial_weights['progress'] * progress_factor
            self.speed_weight = self.initial_weights['speed'] * progress_factor
            self.position_weight = self.initial_weights['position'] * progress_factor
            self.steering_weight = self.initial_weights['steering'] * progress_factor


class CurriculumRewardShaper(CarRacingRewardShaper):
    """Curriculum-based reward shaper that implements progressive difficulty.

    This version starts with heavy shaping for basic behaviors and gradually
    shifts focus to more advanced racing techniques.
    """

    def __init__(self, config: Any) -> None:
        """Initialize curriculum reward shaper."""
        super().__init__(config)

        # Curriculum stages
        self.curriculum_stages = [
            'basic_movement',  # Learn to move and stay on track
            'speed_control',  # Learn optimal speed
            'smooth_steering',  # Learn smooth steering
            'advanced_racing',  # Learn racing lines and advanced techniques
        ]

        self.current_stage = 0
        self.stage_thresholds = [50, 150, 300, float('inf')]  # Episode thresholds for stage progression
        self.episodes_completed = 0

        # Stage-specific weight multipliers
        self.stage_weights = {
            'basic_movement': {'position': 2.0, 'progress': 1.5, 'speed': 0.5, 'steering': 0.5},
            'speed_control': {'position': 1.5, 'progress': 1.5, 'speed': 2.0, 'steering': 0.5},
            'smooth_steering': {'position': 1.0, 'progress': 1.0, 'speed': 1.5, 'steering': 2.0},
            'advanced_racing': {'position': 1.0, 'progress': 2.0, 'speed': 1.0, 'steering': 1.0},
        }

    def update_episode(self, episode_reward: float) -> None:
        """Update curriculum stage based on progress.

        Args:
            episode_reward: Total reward achieved in the episode
        """
        self.episodes_completed += 1

        # Check for stage progression
        if (
            self.current_stage < len(self.curriculum_stages) - 1
            and self.episodes_completed >= self.stage_thresholds[self.current_stage]
        ):
            self.current_stage += 1
            print(f'Curriculum progression: Stage {self.current_stage} - {self.curriculum_stages[self.current_stage]}')

    def shape_reward(
        self, base_reward: float, observation: np.ndarray, action: int | np.ndarray, info: dict[str, Any], done: bool
    ) -> tuple[float, dict[str, float]]:
        """Apply curriculum-based reward shaping."""
        # Get base shaping
        shaped_reward, components = super().shape_reward(base_reward, observation, action, info, done)

        # Apply curriculum-specific weight adjustments
        stage_name = self.curriculum_stages[self.current_stage]
        stage_multipliers = self.stage_weights[stage_name]

        # Adjust components based on current stage
        adjusted_components = {}
        total_adjustment = 0.0

        for component, value in components.items():
            if component in stage_multipliers:
                multiplier = stage_multipliers[component]
                adjusted_value = value * multiplier
                adjustment = adjusted_value - value
                total_adjustment += adjustment
                adjusted_components[component] = adjusted_value
            else:
                adjusted_components[component] = value

        adjusted_components['curriculum_stage'] = float(self.current_stage)
        adjusted_components['curriculum_adjustment'] = total_adjustment

        return shaped_reward + total_adjustment, adjusted_components


def create_reward_shaper(config: Any, shaper_type: str = 'basic') -> CarRacingRewardShaper:
    """Factory function to create reward shaper.

    Args:
        config: Configuration object
        shaper_type: Type of shaper ('basic', 'adaptive', 'curriculum')

    Returns:
        Reward shaper instance
    """
    if shaper_type == 'adaptive':
        return AdaptiveRewardShaper(config)
    elif shaper_type == 'curriculum':
        return CurriculumRewardShaper(config)
    else:
        return CarRacingRewardShaper(config)
