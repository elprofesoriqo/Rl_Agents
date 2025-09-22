from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.ppo_agent import PPOAgent
from utils.parallel_envs import create_parallel_envs
from utils.replay_buffer import create_replay_buffer
from utils.logger import Logger


class PPOTrainer:
    """High-performance PPO trainer with HPC optimizations."""

    def __init__(self, config: Any, logger: Logger) -> None:
        """Initialize PPO trainer.

        Args:
            config: Configuration object
            logger: Logger instance for metrics and progress tracking
        """
        self.config = config
        self.logger = logger

        self.device = torch.device(config.hpc.device)
        logger.info(f'Using device: {self.device}')

        logger.log_system_info()

        env_type = 'single' if config.training.n_parallel_envs == 1 else 'parallel'
        actual_num_envs = config.training.n_parallel_envs

        self.envs = create_parallel_envs(
            config=config,
            num_envs=actual_num_envs,
            env_type=env_type,
        )
        logger.info(f'Created {actual_num_envs} parallel environments (requested: {config.training.n_parallel_envs})')

        self.agent = PPOAgent(
            config=config, observation_space=self.envs.observation_space, action_space=self.envs.action_space
        )
        logger.log_model_info(self.agent.network)

        self.replay_buffer = None
        if config.replay_buffer.buffer_type in ['prioritized', 'uniform']:
            obs_shape = (config.env.frame_stack, *config.env.img_size)
            buffer_device = torch.device(
                'cuda' if config.replay_buffer.gpu_buffer and torch.cuda.is_available() else 'cpu'
            )
            logger.info(
                f'Creating replay buffer on device: {buffer_device} (gpu_buffer={config.replay_buffer.gpu_buffer})'
            )
            self.replay_buffer = create_replay_buffer(
                buffer_type=config.replay_buffer.buffer_type,
                capacity=config.replay_buffer.capacity,
                obs_shape=obs_shape,
                device=buffer_device,
                alpha=config.replay_buffer.alpha,
                beta_start=config.replay_buffer.beta_start,
                beta_steps=config.replay_buffer.beta_steps,
                epsilon=config.replay_buffer.epsilon,
            )
            logger.info(
                f'Created {config.replay_buffer.buffer_type} replay buffer with capacity {config.replay_buffer.capacity} on {buffer_device}'
            )

        self.total_timesteps = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.training_start_time = time.time()

        self.performance_metrics = defaultdict(list)
        self.rollout_times = []
        self.update_times = []

        self._log_hyperparameters()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f'Loading checkpoint: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.agent.network.load_state_dict(checkpoint['model_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.total_timesteps = checkpoint.get('timesteps', 0)
        self.episode_count = checkpoint.get('episodes', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))

        if hasattr(self.agent, 'scaler') and self.agent.scaler and 'scaler_state_dict' in checkpoint:
            self.agent.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f'Checkpoint loaded: timesteps={self.total_timesteps}, episodes={self.episode_count}')

    def _log_hyperparameters(self) -> None:
        """Log hyperparameters to TensorBoard."""
        hparams = {
            'learning_rate': self.config.ppo.learning_rate,
            'gamma': self.config.ppo.gamma,
            'gae_lambda': self.config.ppo.gae_lambda,
            'clip_epsilon': self.config.ppo.clip_epsilon,
            'value_loss_coeff': self.config.ppo.value_loss_coeff,
            'entropy_coeff': self.config.ppo.entropy_coeff,
            'batch_size': self.config.ppo.batch_size,
            'n_parallel_envs': self.config.training.n_parallel_envs,
            'rollout_steps': self.config.training.rollout_steps,
            'ppo_epochs': self.config.ppo.ppo_epochs,
            'max_grad_norm': self.config.ppo.max_grad_norm,
            'mixed_precision': self.config.training.mixed_precision,
            'compile_model': self.config.hpc.compile_model,
        }

        if self.replay_buffer:
            hparams.update(
                {
                    'buffer_type': self.config.replay_buffer.buffer_type,
                    'buffer_capacity': self.config.replay_buffer.capacity,
                    'per_alpha': self.config.replay_buffer.alpha,
                    'per_beta_start': self.config.replay_buffer.beta_start,
                }
            )

        self.logger.log_hyperparameters(hparams)

    def train(self) -> None:
        """Main training loop."""
        self.logger.info('Starting PPO training')
        self.logger.info(f'Target timesteps: {self.config.training.total_timesteps:,}')
        self.logger.info(f'Rollout steps: {self.config.training.rollout_steps}')
        self.logger.info(f'Parallel environments: {self.config.training.n_parallel_envs}')

        try:
            observations = self.envs.reset()
            self.logger.info('Environments reset successfully')

            while self.total_timesteps < self.config.training.total_timesteps:
                rollout_start_time = time.time()
                rollout_data = self._collect_rollouts(observations)
                rollout_time = time.time() - rollout_start_time
                self.rollout_times.append(rollout_time)

                update_start_time = time.time()
                if getattr(self.config.logging, 'profiler', False):
                    try:
                        import torch.profiler as profiler

                        with profiler.profile(
                            activities=[profiler.ProfilerActivity.CPU]
                            + ([profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
                            record_shapes=False,
                            with_stack=False,
                        ) as prof:
                            training_metrics = self._update_agent(rollout_data)
                        trace_dir = Path(self.logger.run_dir) / 'traces'
                        trace_dir.mkdir(exist_ok=True)
                        prof.export_chrome_trace(trace_dir / f'update_{self.total_timesteps}.json')
                    except Exception as e:
                        self.logger.warning(f'Profiler failed: {e}')
                        training_metrics = self._update_agent(rollout_data)
                else:
                    training_metrics = self._update_agent(rollout_data)
                update_time = time.time() - update_start_time
                self.update_times.append(update_time)

                observations = rollout_data['next_observations']

                self._log_training_metrics(training_metrics, rollout_time, update_time)

                if (
                    self.total_timesteps % self.config.training.save_interval == 0
                    or self.total_timesteps >= self.config.training.total_timesteps
                ):
                    self._save_checkpoint()

                if self.total_timesteps % self.config.training.eval_interval == 0:
                    self._evaluate_agent()

        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user')
        except Exception as e:
            self.logger.error(f'Training error: {e}')
            raise
        finally:
            self._cleanup()

        self.logger.info('Training completed successfully')

    def _collect_rollouts(self, initial_observations: np.ndarray) -> dict[str, Any]:
        """Collect rollouts from parallel environments.

        Args:
            initial_observations: Initial observations from environments

        Returns:
            Dictionary containing rollout data
        """
        rollout_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_observations': None,
            'episode_stats': [],
        }

        observations = initial_observations

        for step in range(self.config.training.rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(self.device)
                actions, log_probs, values = self.agent.network.get_action_and_value(obs_tensor)

            next_observations, rewards, terminated, truncated, infos = self.envs.step(actions.cpu().numpy())
            dones = terminated | truncated

            rollout_data['observations'].append(observations.copy())
            rollout_data['actions'].append(actions.cpu().numpy())
            rollout_data['rewards'].append(rewards)
            rollout_data['values'].append(values.cpu().numpy())
            rollout_data['log_probs'].append(log_probs.cpu().numpy())
            rollout_data['dones'].append(dones)

            for i, info in enumerate(infos):
                if dones[i] and 'final_episode_reward' in info:
                    episode_stats = {
                        'episode_reward': info['final_episode_reward'],
                        'episode_length': info['final_episode_length'],
                        'timestep': self.total_timesteps + (step + 1) * self.config.training.n_parallel_envs,
                    }
                    rollout_data['episode_stats'].append(episode_stats)
                    self.episode_count += 1

                    # Log to console
                    self.logger.info(
                        f'Episode {self.episode_count} completed! '
                        f'Reward: {info["final_episode_reward"]:.2f}, '
                        f'Length: {info["final_episode_length"]} steps'
                    )

                    # Log to metrics files
                    episode_metrics = {
                        'episode_reward': info['final_episode_reward'],
                        'episode_length': info['final_episode_length'],
                        'episode_number': self.episode_count,
                        'timesteps': self.total_timesteps + (step + 1) * self.config.training.n_parallel_envs,
                    }
                    self.logger.log_metrics(episode_metrics)

            # Add to replay buffer if using PER
            if self.replay_buffer:
                for i in range(len(observations)):
                    if step > 0:
                        obs_tensor = torch.FloatTensor(rollout_data['observations'][step - 1][i])
                        action_tensor = torch.tensor(rollout_data['actions'][step - 1][i])
                        reward_tensor = torch.tensor(rollout_data['rewards'][step - 1][i])
                        next_obs_tensor = torch.FloatTensor(observations[i])
                        done_tensor = torch.tensor(rollout_data['dones'][step - 1][i])

                        self.replay_buffer.add(obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor)

            observations = next_observations
            self.total_timesteps += self.config.training.n_parallel_envs

        rollout_data['next_observations'] = observations

        for key in ['observations', 'actions', 'rewards', 'values', 'log_probs', 'dones']:
            rollout_data[key] = np.array(rollout_data[key])

        return rollout_data

    def _update_agent(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Update PPO agent using collected rollouts.

        Args:
            rollout_data: Rollout data from environments

        Returns:
            Dictionary of training metrics
        """
        observations = torch.FloatTensor(rollout_data['observations']).to(self.device)
        actions = torch.LongTensor(rollout_data['actions']).to(self.device)
        rewards = torch.FloatTensor(rollout_data['rewards']).to(self.device)
        values = torch.FloatTensor(rollout_data['values']).to(self.device)
        log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
        dones = torch.BoolTensor(rollout_data['dones']).to(self.device)

        # Get next value for GAE computation
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(rollout_data['next_observations']).to(self.device)
            next_values = self.agent.network.get_value(next_obs_tensor)

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones, next_values)

        observations = observations.reshape(-1, *observations.shape[2:])
        actions = actions.reshape(-1)
        log_probs = log_probs.reshape(-1)
        values = values.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        training_metrics = self.agent.update_policy(observations, actions, log_probs, values, advantages, returns)

        if self.replay_buffer and self.replay_buffer.is_ready(self.config.replay_buffer.batch_size):
            self._update_replay_buffer_priorities(training_metrics.get('td_errors'))

        return training_metrics

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor [rollout_steps, n_envs]
            values: Value estimates [rollout_steps, n_envs]
            dones: Done flags [rollout_steps, n_envs]
            next_values: Next value estimates [n_envs]

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        values_extended = torch.cat([values, next_values.unsqueeze(0)], dim=0)

        for t in reversed(range(self.config.training.rollout_steps)):
            if t == self.config.training.rollout_steps - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_values_t = values_extended[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_values_t = values_extended[t + 1]

            delta = rewards[t] + self.config.ppo.gamma * next_values_t * next_non_terminal - values[t]
            advantages[t] = last_gae = (
                delta + self.config.ppo.gamma * self.config.ppo.gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values

        if getattr(self.config.ppo, 'normalize_advantages', True):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _update_replay_buffer_priorities(self, td_errors: torch.Tensor | None) -> None:
        """Update replay buffer priorities using TD errors.

        Args:
            td_errors: TD errors from policy update
        """
        if td_errors is None or not self.replay_buffer.is_ready(32):
            return

        try:
            batch = self.replay_buffer.sample(32)
            observations, actions, rewards, next_observations, dones, weights, indices = batch

            with torch.no_grad():
                current_values = self.agent.network.get_value(observations)
                next_values = self.agent.network.get_value(next_observations)

                target_values = rewards + self.config.ppo.gamma * next_values * (~dones).float()
                td_errors = torch.abs(target_values - current_values)

            self.replay_buffer.update_priorities(indices, td_errors)

        except Exception as e:
            self.logger.warning(f'Failed to update replay buffer priorities: {e}')

    def _log_training_metrics(
        self, training_metrics: dict[str, float], rollout_time: float, update_time: float
    ) -> None:
        """Log training metrics and performance statistics.

        Args:
            training_metrics: Metrics from agent update
            rollout_time: Time taken for rollout collection
            update_time: Time taken for policy update
        """
        fps = (self.config.training.rollout_steps * self.config.training.n_parallel_envs) / rollout_time
        total_time = time.time() - self.training_start_time

        metrics = {
            'timesteps': self.total_timesteps,
            'episodes': self.episode_count,
            'fps': fps,
            'rollout_time': rollout_time,
            'update_time': update_time,
            'total_time': total_time,
            **training_metrics,
        }

        env_stats = self.envs.get_episode_statistics()
        if env_stats:
            metrics.update({f'env/{key}': value for key, value in env_stats.items()})

            if 'mean_reward' in env_stats:
                if env_stats['mean_reward'] > self.best_reward:
                    self.best_reward = env_stats['mean_reward']
                    metrics['best_reward'] = self.best_reward

        if self.replay_buffer:
            metrics.update(
                {
                    'replay_buffer/size': len(self.replay_buffer),
                    'replay_buffer/capacity': self.replay_buffer.capacity,
                }
            )

        self.logger.log_metrics(metrics)

        if self.total_timesteps % (self.config.training.rollout_steps * self.config.training.n_parallel_envs) == 0:
            self.logger.info(
                f'Timesteps: {self.total_timesteps:,}/{self.config.training.total_timesteps:,} '
                f'| Episodes: {self.episode_count} | FPS: {fps:.0f} | '
                f'Mean Reward: {env_stats.get("mean_reward", 0):.2f} | '
                f'Policy Loss: {training_metrics.get("policy_loss", 0):.4f} | '
                f'Value Loss: {training_metrics.get("value_loss", 0):.4f}'
            )

    def _evaluate_agent(self) -> None:
        """Evaluate agent performance."""
        self.logger.info('Running evaluation...')

        eval_rewards = []
        eval_lengths = []

        eval_env = create_parallel_envs(config=self.config, num_envs=1, env_type='single')

        try:
            for episode in range(self.config.training.eval_episodes):
                obs = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False

                while not done:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).to(self.device)
                        actions, _, _ = self.agent.network.get_action_and_value(obs_tensor)

                    obs, rewards, terminated, truncated, _ = eval_env.step(actions.cpu().numpy())
                    episode_reward += rewards[0]
                    episode_length += 1
                    done = terminated[0] or truncated[0]

                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)

            eval_metrics = {
                'eval/mean_reward': np.mean(eval_rewards),
                'eval/std_reward': np.std(eval_rewards),
                'eval/min_reward': np.min(eval_rewards),
                'eval/max_reward': np.max(eval_rewards),
                'eval/mean_length': np.mean(eval_lengths),
            }

            self.logger.log_metrics(eval_metrics)
            self.logger.info(
                f'Evaluation completed: Mean Reward: {eval_metrics["eval/mean_reward"]:.2f} ± '
                f'{eval_metrics["eval/std_reward"]:.2f}'
            )

        finally:
            eval_env.close()

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.logger.run_dir) / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f'checkpoint_{self.total_timesteps}.pth'

        checkpoint = {
            'timesteps': self.total_timesteps,
            'episodes': self.episode_count,
            'best_reward': self.best_reward,
            'model_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'scheduler_state_dict': self.agent.scheduler.state_dict(),
            'config': self.config,
        }

        if hasattr(self.agent, 'scaler') and self.agent.scaler:
            checkpoint['scaler_state_dict'] = self.agent.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')

        if hasattr(self, 'best_reward') and len(self.performance_metrics.get('mean_reward', [])) > 0:
            current_reward = self.performance_metrics['mean_reward'][-1]
            if current_reward >= self.best_reward:
                best_model_path = checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, best_model_path)
                self.logger.info(f'Best model updated: {best_model_path}')

    def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info('Cleaning up resources...')

        try:
            self.envs.close()
            self.logger.info('Environments closed')
        except Exception as e:
            self.logger.warning(f'Error closing environments: {e}')

        self.logger.save_excel_metrics()

        total_training_time = time.time() - self.training_start_time
        avg_fps = self.total_timesteps / total_training_time if total_training_time > 0 else 0

        final_stats = {
            'total_training_time': total_training_time,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.episode_count,
            'average_fps': avg_fps,
            'best_reward': self.best_reward,
        }

        if self.rollout_times:
            final_stats.update(
                {
                    'avg_rollout_time': np.mean(self.rollout_times),
                    'avg_update_time': np.mean(self.update_times),
                }
            )

        self.logger.log_metrics(final_stats)
        self.logger.info(f'Training completed in {total_training_time:.2f}s with average FPS: {avg_fps:.0f}')


def create_trainer(config: Any, logger: Logger) -> PPOTrainer:
    """Factory function to create PPO trainer.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        PPOTrainer instance
    """
    return PPOTrainer(config, logger)
