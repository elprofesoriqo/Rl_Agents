from __future__ import annotations

import multiprocessing as mp
import time
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch
from torch.multiprocessing import Process, Queue, Event

from utils.environment import CarRacingEnvironment


class ParallelEnvironmentWorker:
    """Worker process for running a single environment."""

    def __init__(
        self,
        worker_id: int,
        config: Any,
        command_queue: Queue,
        result_queue: Queue,
        ready_event: Event,
        stop_event: Event,
    ) -> None:
        self.worker_id = worker_id
        self.config = config
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.env = None
        self.last_obs = None
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0

    def run(self) -> None:
        """Main worker loop."""
        try:
            self.env = CarRacingEnvironment(self.config)
            self.ready_event.set()

            while not self.stop_event.is_set():
                try:
                    command = self.command_queue.get(timeout=0.1)
                    if command['type'] == 'reset':
                        self._handle_reset(command)
                    elif command['type'] == 'step':
                        self._handle_step(command)
                    elif command['type'] == 'close':
                        break
                except Exception as e:
                    if 'Empty' in str(e) or 'queue.Empty' in str(type(e)):
                        continue
                    else:
                        import traceback

                        error_details = f'{str(e)}\n{traceback.format_exc()}'
                        self.result_queue.put({'worker_id': self.worker_id, 'type': 'error', 'error': error_details})
        except Exception as e:
            import traceback

            error_details = f'{str(e)}\n{traceback.format_exc()}'
            self.result_queue.put({'worker_id': self.worker_id, 'type': 'init_error', 'error': error_details})
        finally:
            if self.env:
                self.env.close()

    def _handle_reset(self, command: dict[str, Any]) -> None:
        """Handle environment reset command."""
        obs, info = self.env.reset()
        self.last_obs = obs
        self.episode_reward = 0.0
        self.episode_length = 0
        self.result_queue.put({'worker_id': self.worker_id, 'type': 'reset', 'obs': obs, 'info': info})

    def _handle_step(self, command: dict[str, Any]) -> None:
        """Handle environment step command."""
        action = command['action']
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        done = terminated or truncated
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length

        if done:
            self.episode_count += 1
            info['episode_count'] = self.episode_count
            info['final_episode_reward'] = self.episode_reward
            info['final_episode_length'] = self.episode_length

        self.result_queue.put(
            {
                'worker_id': self.worker_id,
                'type': 'step',
                'obs': obs,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'done': done,
                'info': info,
            }
        )
        self.last_obs = obs


class ParallelEnvironments:
    """Manages multiple parallel environments for efficient rollout collection."""

    def __init__(self, config: Any, num_envs: int | None = None) -> None:
        self.config = config
        self.num_envs = num_envs or config.training.n_parallel_envs

        import platform

        if platform.system() == 'Windows':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
        else:
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

        self.command_queues = [Queue() for _ in range(self.num_envs)]
        self.result_queue = Queue()
        self.ready_events = [Event() for _ in range(self.num_envs)]
        self.stop_event = Event()
        self.workers = []
        self.processes = []
        self.observations = [None] * self.num_envs
        self.episode_stats = defaultdict(list)
        self.total_steps = 0
        self.total_episodes = 0

        from utils.environment import CarRacingEnvironment

        temp_env = CarRacingEnvironment(config)
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()
        self._start_workers()

    def _start_workers(self) -> None:
        """Start all worker processes."""
        for i in range(self.num_envs):
            worker = ParallelEnvironmentWorker(
                worker_id=i,
                config=self.config,
                command_queue=self.command_queues[i],
                result_queue=self.result_queue,
                ready_event=self.ready_events[i],
                stop_event=self.stop_event,
            )
            process = Process(target=worker.run)
            process.start()
            self.workers.append(worker)
            self.processes.append(process)

        for event in self.ready_events:
            event.wait(timeout=30)

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        for i in range(self.num_envs):
            self.command_queues[i].put({'type': 'reset'})

        observations = [None] * self.num_envs
        for _ in range(self.num_envs):
            result = self.result_queue.get(timeout=10)
            if result['type'] == 'error':
                raise RuntimeError(f'Worker {result["worker_id"]} error: {result["error"]}')
            worker_id = result['worker_id']
            observations[worker_id] = result['obs']

        self.observations = observations
        return np.array(observations)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments with given actions."""
        if len(actions) != self.num_envs:
            raise ValueError(f'Expected {self.num_envs} actions, got {len(actions)}')

        for i, action in enumerate(actions):
            self.command_queues[i].put({'type': 'step', 'action': action})

        observations = [None] * self.num_envs
        rewards = [None] * self.num_envs
        terminated = [None] * self.num_envs
        truncated = [None] * self.num_envs
        infos = [None] * self.num_envs

        for _ in range(self.num_envs):
            result = self.result_queue.get(timeout=10)
            if result['type'] == 'error':
                raise RuntimeError(f'Worker {result["worker_id"]} error: {result["error"]}')

            worker_id = result['worker_id']
            observations[worker_id] = result['obs']
            rewards[worker_id] = result['reward']
            terminated[worker_id] = result['terminated']
            truncated[worker_id] = result['truncated']
            infos[worker_id] = result['info']

            if result['done']:
                self.total_episodes += 1
                self.episode_stats['rewards'].append(result['info']['final_episode_reward'])
                self.episode_stats['lengths'].append(result['info']['final_episode_length'])

        self.observations = observations
        self.total_steps += self.num_envs
        return (np.array(observations), np.array(rewards), np.array(terminated), np.array(truncated), infos)

    def get_episode_statistics(self) -> dict[str, Any]:
        """Get episode statistics since last call."""
        stats = {}

        if self.episode_stats['rewards']:
            rewards = self.episode_stats['rewards']
            lengths = self.episode_stats['lengths']

            stats = {
                'episode_count': len(rewards),
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
            }

            # Clear statistics after reporting
            self.episode_stats = defaultdict(list)

        return stats

    def close(self) -> None:
        """Close all environments and stop worker processes."""
        self.stop_event.set()
        for i in range(self.num_envs):
            try:
                self.command_queues[i].put({'type': 'close'})
            except:
                pass

        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()


class VectorizedEnvironments:
    """Vectorized environment wrapper for batch processing.

    This provides a unified interface for both parallel and single environments,
    making it easy to switch between different parallelization strategies.
    """

    def __init__(
        self,
        config: Any,
        num_envs: int | None = None,
        env_type: str = 'parallel',
    ) -> None:
        """Initialize vectorized environments.

        Args:
            config: Configuration object
            num_envs: Number of environments
            env_type: Type of environment ('parallel', 'async', 'single')
        """
        self.config = config
        self.num_envs = num_envs or config.training.n_parallel_envs
        self.env_type = env_type

        if env_type == 'parallel':
            self.envs = ParallelEnvironments(config, num_envs)
        elif env_type == 'async':
            self.envs = AsyncParallelEnvironments(config, num_envs)
        elif env_type == 'single':
            # Single environment wrapped to look like parallel
            self.envs = self._create_single_env_wrapper(config)
        else:
            raise ValueError(f'Unknown env_type: {env_type}')

        # Common interface
        # Get observation space from the actual environment
        if hasattr(self.envs, 'observation_space'):
            self.observation_space = self.envs.observation_space
        elif hasattr(self.envs, 'sync_envs') and hasattr(self.envs.sync_envs, 'observation_space'):
            self.observation_space = self.envs.sync_envs.observation_space
        else:
            # Fallback: create a dummy environment to get the observation space
            from utils.environment import CarRacingEnvironment

            temp_env = CarRacingEnvironment(config)
            self.observation_space = temp_env.observation_space
            temp_env.close()

        self.action_space = 5  # Mike's proven 5-action discrete space

    def _create_single_env_wrapper(self, config: Any) -> Any:
        """Create single or pseudo-parallel environment wrapper."""

        class SingleEnvWrapper:
            def __init__(self, config):
                from utils.environment import CarRacingEnvironment

                # Create multiple environments with smart rendering (only first env renders)
                self.num_envs = config.training.n_parallel_envs
                self.envs = []
                for i in range(self.num_envs):
                    # Create a copy of config with environment ID
                    env_config = type(config)()
                    env_config.__dict__.update(config.__dict__)
                    env_config._env_id = i  # Add environment ID for rendering control
                    self.envs.append(CarRacingEnvironment(env_config))
                self.observation_space = self.envs[0].observation_space
                self.action_space = self.envs[0].action_space

                # Episode tracking for each environment
                self.episode_rewards = [0.0] * self.num_envs
                self.episode_lengths = [0] * self.num_envs
                self.episode_count = 0

            def reset(self):
                # Reset all environments
                observations = []
                for i in range(self.num_envs):
                    obs, info = self.envs[i].reset()
                    observations.append(obs)
                    # Reset episode tracking
                    self.episode_rewards[i] = 0.0
                    self.episode_lengths[i] = 0
                return np.array(observations)

            def step(self, actions):
                # Step all environments
                observations = []
                rewards = []
                terminated_list = []
                truncated_list = []
                infos = []

                for i in range(self.num_envs):
                    action = actions[i] if len(actions) > i else actions[0]
                    obs, reward, terminated, truncated, info = self.envs[i].step(action)

                    # Track episode progress
                    self.episode_rewards[i] += reward
                    self.episode_lengths[i] += 1

                    # Add current episode stats to info
                    info['episode_reward'] = self.episode_rewards[i]
                    info['episode_length'] = self.episode_lengths[i]

                    # If episode is done, add final stats and reset
                    done = terminated or truncated
                    if done:
                        self.episode_count += 1
                        info['episode_count'] = self.episode_count
                        info['final_episode_reward'] = self.episode_rewards[i]
                        info['final_episode_length'] = self.episode_lengths[i]

                        # Auto-reset for next episode
                        next_obs, _ = self.envs[i].reset()
                        obs = next_obs  # Use the reset observation

                        # Reset episode tracking after storing final stats
                        self.episode_rewards[i] = 0.0
                        self.episode_lengths[i] = 0

                    observations.append(obs)
                    rewards.append(reward)
                    terminated_list.append(terminated)
                    truncated_list.append(truncated)
                    infos.append(info)

                return (
                    np.array(observations),
                    np.array(rewards),
                    np.array(terminated_list),
                    np.array(truncated_list),
                    infos,
                )

            def get_episode_statistics(self):
                # Return simple stats - the trainer will get episode info from the step() method
                return {
                    'episode_count': self.episode_count,
                    'mean_reward': 0.0,  # Will be calculated by trainer from episode completions
                }

            def close(self):
                for env in self.envs:
                    env.close()

        return SingleEnvWrapper(config)

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        return self.envs.reset()

    def step(self, actions: np.ndarray | list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments."""
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        return self.envs.step(actions)

    def get_episode_statistics(self) -> dict[str, Any]:
        """Get episode statistics."""
        return self.envs.get_episode_statistics()

    def close(self) -> None:
        """Close all environments."""
        self.envs.close()


def create_parallel_envs(
    config: Any,
    num_envs: int | None = None,
    env_type: str = 'parallel',
) -> VectorizedEnvironments:
    """Factory function to create parallel environments.

    Args:
        config: Configuration object
        num_envs: Number of parallel environments
        env_type: Type of parallelization ('parallel', 'async', 'single')

    Returns:
        VectorizedEnvironments instance
    """
    return VectorizedEnvironments(config, num_envs, env_type)
