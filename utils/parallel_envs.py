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
        """Initialize environment worker.

        Args:
            worker_id: Unique ID for this worker
            config: Configuration object
            command_queue: Queue for receiving commands
            result_queue: Queue for sending results
            ready_event: Event to signal worker is ready
            stop_event: Event to signal worker should stop
        """
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
                    self.result_queue.put({'worker_id': self.worker_id, 'type': 'error', 'error': str(e)})

        except Exception as e:
            self.result_queue.put({'worker_id': self.worker_id, 'type': 'init_error', 'error': str(e)})
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
        """Initialize parallel environments.

        Args:
            config: Configuration object
            num_envs: Number of parallel environments (uses config if None)
        """
        self.config = config
        self.num_envs = num_envs or config.training.n_parallel_envs

        # Multiprocessing setup
        mp.set_start_method('spawn', force=True)

        # Communication queues and events
        self.command_queues = [Queue() for _ in range(self.num_envs)]
        self.result_queue = Queue()
        self.ready_events = [Event() for _ in range(self.num_envs)]
        self.stop_event = Event()

        # Worker processes
        self.workers = []
        self.processes = []

        # State tracking
        self.observations = [None] * self.num_envs
        self.episode_stats = defaultdict(list)
        self.total_steps = 0
        self.total_episodes = 0

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

        # Wait for all workers to be ready
        for event in self.ready_events:
            event.wait(timeout=30)  # 30 second timeout

    def reset(self) -> np.ndarray:
        """Reset all environments.

        Returns:
            Array of initial observations
        """
        # Send reset commands to all workers
        for i in range(self.num_envs):
            self.command_queues[i].put({'type': 'reset'})

        # Collect reset results
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
        """Step all environments with given actions.

        Args:
            actions: Array of actions for each environment

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        if len(actions) != self.num_envs:
            raise ValueError(f'Expected {self.num_envs} actions, got {len(actions)}')

        # Send step commands to all workers
        for i, action in enumerate(actions):
            self.command_queues[i].put({'type': 'step', 'action': action})

        # Collect step results
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

            # Track episode statistics
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
        # Signal all workers to stop
        self.stop_event.set()

        # Send close commands
        for i in range(self.num_envs):
            try:
                self.command_queues[i].put({'type': 'close'})
            except:
                pass  # Queue might be full or closed

        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()

    def __enter__(self) -> ParallelEnvironments:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class AsyncParallelEnvironments:
    """Asynchronous parallel environments for maximum throughput.

    This version doesn't wait for all environments to complete each step,
    allowing for better utilization when environments have varying step times.
    """

    def __init__(self, config: Any, num_envs: int | None = None) -> None:
        """Initialize async parallel environments.

        Args:
            config: Configuration object
            num_envs: Number of parallel environments
        """
        self.config = config
        self.num_envs = num_envs or config.training.n_parallel_envs

        # Use synchronous implementation as base for now
        # This can be extended for true async behavior
        self.sync_envs = ParallelEnvironments(config, num_envs)

        # Async-specific state
        self.pending_steps = {}
        self.completed_episodes = []

    def reset(self) -> np.ndarray:
        """Reset all environments asynchronously."""
        return self.sync_envs.reset()

    def step_async(self, actions: np.ndarray) -> None:
        """Submit actions to environments asynchronously."""
        # For now, delegate to sync version
        # Future implementation could track pending operations
        self.last_actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for and collect results from async steps."""
        return self.sync_envs.step(self.last_actions)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Synchronous step (for compatibility)."""
        return self.sync_envs.step(actions)

    def get_episode_statistics(self) -> dict[str, Any]:
        """Get episode statistics."""
        return self.sync_envs.get_episode_statistics()

    def close(self) -> None:
        """Close all environments."""
        self.sync_envs.close()

    def __enter__(self) -> AsyncParallelEnvironments:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


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
        self.observation_space = self.envs.sync_envs.config.env if hasattr(self.envs, 'sync_envs') else config.env
        self.action_space = 9  # Discrete action space size

    def _create_single_env_wrapper(self, config: Any) -> Any:
        """Create single environment wrapper that mimics parallel interface."""

        class SingleEnvWrapper:
            def __init__(self, config):
                from utils.environment import CarRacingEnvironment

                self.env = CarRacingEnvironment(config)
                self.num_envs = 1

            def reset(self):
                obs, info = self.env.reset()
                return np.array([obs])

            def step(self, actions):
                action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions
                obs, reward, terminated, truncated, info = self.env.step(action)
                return (np.array([obs]), np.array([reward]), np.array([terminated]), np.array([truncated]), [info])

            def get_episode_statistics(self):
                return {}

            def close(self):
                self.env.close()

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

    def __enter__(self) -> VectorizedEnvironments:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


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
