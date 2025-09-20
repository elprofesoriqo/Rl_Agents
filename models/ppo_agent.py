import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import optim

from utils.environment import CarRacingEnvironment

from .networks import create_ppo_network


class PPOMemory:
    """Memory buffer for storing PPO rollout data."""

    def __init__(self, rollout_steps: int, n_envs: int, obs_shape: tuple, device: torch.device) -> None:
        """Initialize PPO memory buffer.

        Args:
            rollout_steps: Number of steps per rollout
            n_envs: Number of parallel environments
            obs_shape: Shape of observations
            device: Device to store tensors on

        """
        self.rollout_steps = rollout_steps
        self.n_envs = n_envs
        self.device = device

        # Initialize storage tensors
        self.observations = torch.zeros((rollout_steps, n_envs, *obs_shape), device=device)
        self.actions = torch.zeros((rollout_steps, n_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((rollout_steps, n_envs), device=device)
        self.values = torch.zeros((rollout_steps, n_envs), device=device)
        self.rewards = torch.zeros((rollout_steps, n_envs), device=device)
        self.dones = torch.zeros((rollout_steps, n_envs), dtype=torch.bool, device=device)
        self.advantages = torch.zeros((rollout_steps, n_envs), device=device)
        self.returns = torch.zeros((rollout_steps, n_envs), device=device)

        self.step = 0

    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Store a step of experience."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.step += 1

    def compute_advantages_and_returns(self, next_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Compute advantages and returns using GAE."""
        with torch.no_grad():
            values_with_next = torch.cat([self.values, next_value.unsqueeze(0)], dim=0)

            advantages = torch.zeros_like(self.advantages)
            last_gae = 0

            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    next_non_terminal = 1.0 - self.dones[t].float()
                    next_value_t = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t].float()
                    next_value_t = values_with_next[t + 1]

                delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
                advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            self.returns = advantages + self.values
            self.advantages = advantages

            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """Get mini-batches for training."""
        total_samples = self.rollout_steps * self.n_envs
        indices = torch.randperm(total_samples, device=self.device)

        # Flatten all tensors
        flat_obs = self.observations.view(-1, *self.observations.shape[2:])
        flat_actions = self.actions.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_values = self.values.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                'observations': flat_obs[batch_indices],
                'actions': flat_actions[batch_indices],
                'old_log_probs': flat_log_probs[batch_indices],
                'old_values': flat_values[batch_indices],
                'advantages': flat_advantages[batch_indices],
                'returns': flat_returns[batch_indices],
            }

    def reset(self) -> None:
        """Reset memory buffer."""
        self.step = 0


class PPOAgent:
    def __init__(self, config, observation_space, action_space) -> None:
        """Initialize PPO agent.

        Args:
            config: Configuration object
            observation_space: Environment observation space
            action_space: Environment action space

        """
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

        self.device = torch.device(config.hpc.device)

        self.network = create_ppo_network(config).to(self.device)

        self.use_mixed_precision = config.training.mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')

        self.network = torch.compile(self.network)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.ppo.learning_rate,
            eps=1e-5,
        )

        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.training.total_timesteps
            // (config.training.rollout_steps * config.training.n_parallel_envs),
        )

        self.gamma = config.ppo.gamma
        self.gae_lambda = config.ppo.gae_lambda
        self.clip_epsilon = config.ppo.clip_epsilon
        self.value_loss_coeff = config.ppo.value_loss_coeff
        self.entropy_coeff = config.ppo.entropy_coeff
        self.max_grad_norm = config.ppo.max_grad_norm
        self.ppo_epochs = config.ppo.ppo_epochs

        self.total_timesteps = 0
        self.episode_count = 0
        self.best_reward = float('-inf')

        obs_shape = observation_space.shape
        self.memory = PPOMemory(
            rollout_steps=config.training.rollout_steps,
            n_envs=config.training.n_parallel_envs,
            obs_shape=obs_shape,
            device=self.device,
        )

        self.training_metrics = defaultdict(list)

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        """Get action from observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic action selection

        Returns:
            Tuple of (action, log_prob, value)

        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            if deterministic:
                action_logits, value = self.network(obs_tensor)
                action = torch.argmax(action_logits, dim=-1)
                log_prob = torch.zeros_like(action, dtype=torch.float)
            else:
                action, log_prob, value = self.network.get_action_and_value(obs_tensor)

            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def collect_rollouts(self, envs: list[CarRacingEnvironment]) -> dict[str, float]:
        """Collect rollouts from parallel environments.

        Args:
            envs: List of parallel environments

        Returns:
            Dictionary of episode statistics

        """
        self.network.eval()
        episode_stats = defaultdict(list)

        self.memory.reset()

        observations = []
        for env in envs:
            try:
                obs, _ = env.reset()
                observations.append(obs)
            except:
                observations.append(env._last_observation)

        observations = torch.FloatTensor(np.array(observations)).to(self.device)

        for _step in range(self.config.training.rollout_steps):
            with torch.no_grad():
                actions, log_probs, values = self.network.get_action_and_value(observations)

            next_observations = []
            rewards = []
            dones = []

            for i, (env, action) in enumerate(zip(envs, actions.cpu().numpy())):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done)

                if done:
                    episode_stats['episode_reward'].append(info.get('episode_reward', 0))
                    episode_stats['episode_length'].append(info.get('episode_step', 0))
                    obs, _ = env.reset()
                    next_observations[i] = obs

            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.BoolTensor(dones).to(self.device)

            self.memory.store(
                observations,
                actions,
                log_probs,
                values,
                rewards_tensor,
                dones_tensor,
            )

            observations = torch.FloatTensor(np.array(next_observations)).to(self.device)
            self.total_timesteps += len(envs)

        with torch.no_grad():
            next_values = self.network.get_value(observations)

        self.memory.compute_advantages_and_returns(
            next_values,
            self.gamma,
            self.gae_lambda,
        )

        return dict(episode_stats)

    def update_policy(
        self,
        observations: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        old_values: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Update policy using PPO algorithm.

        Args:
            observations: Observations tensor (optional, uses memory if None)
            actions: Actions tensor (optional, uses memory if None)
            old_log_probs: Old log probabilities (optional, uses memory if None)
            old_values: Old value estimates (optional, uses memory if None)
            advantages: Advantage estimates (optional, uses memory if None)
            returns: Return estimates (optional, uses memory if None)

        Returns:
            Dictionary of training metrics

        """
        self.network.train()

        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []

        # Use direct tensors if provided, otherwise use memory
        if observations is not None:
            # Direct tensor mode - create single batch
            batch_size = self.config.ppo.mini_batch_size
            total_samples = observations.shape[0]

            batches = []
            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_data = {
                    'observations': observations[start_idx:end_idx],
                    'actions': actions[start_idx:end_idx],
                    'old_log_probs': old_log_probs[start_idx:end_idx],
                    'old_values': old_values[start_idx:end_idx],
                    'advantages': advantages[start_idx:end_idx],
                    'returns': returns[start_idx:end_idx],
                }
                batches.append(batch_data)
        else:
            # Memory mode - use existing memory system
            batches = list(self.memory.get_batches(self.config.ppo.mini_batch_size))

        for _epoch in range(self.ppo_epochs):
            for batch in batches:
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs, values, entropy = self.network.evaluate_actions(
                            batch['observations'],
                            batch['actions'],
                        )

                        ratio = torch.exp(log_probs - batch['old_log_probs'])
                        surr1 = ratio * batch['advantages']
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantages']
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_pred_clipped = batch['old_values'] + torch.clamp(
                            values - batch['old_values'],
                            -self.clip_epsilon,
                            self.clip_epsilon,
                        )
                        value_loss1 = (values - batch['returns']).pow(2)
                        value_loss2 = (value_pred_clipped - batch['returns']).pow(2)
                        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                        total_loss = (
                            policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy.mean()
                        )

                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    log_probs, values, entropy = self.network.evaluate_actions(
                        batch['observations'],
                        batch['actions'],
                    )

                    ratio = torch.exp(log_probs - batch['old_log_probs'])
                    surr1 = ratio * batch['advantages']
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantages']
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = batch['old_values'] + torch.clamp(
                        values - batch['old_values'],
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss1 = (values - batch['returns']).pow(2)
                    value_loss2 = (value_pred_clipped - batch['returns']).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                    total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy.mean()

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_fraction.item())

        self.scheduler.step()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.scheduler.get_last_lr()[0],
        }

    def train(self, env: CarRacingEnvironment, logger) -> None:
        """Main training loop.

        Args:
            env: Environment instance (will create parallel envs)
            logger: Logger instance

        """
        logger.info('Starting PPO training')
        logger.log_system_info()
        logger.log_model_info(self.network)

        envs = [CarRacingEnvironment(self.config) for _ in range(self.config.training.n_parallel_envs)]

        update_count = 0
        start_time = time.time()

        while self.total_timesteps < self.config.training.total_timesteps:
            episode_stats = self.collect_rollouts(envs)

            training_metrics = self.update_policy()

            update_count += 1

            if episode_stats['episode_reward']:
                avg_reward = np.mean(episode_stats['episode_reward'])
                avg_length = np.mean(episode_stats['episode_length'])

                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.save_checkpoint(logger.run_dir / 'best_model.pt')

                logger.log_metrics(
                    {
                        'episode_reward_mean': avg_reward,
                        'episode_length_mean': avg_length,
                        'episode_reward_std': np.std(episode_stats['episode_reward']),
                        'best_reward': self.best_reward,
                        'total_timesteps': self.total_timesteps,
                        'fps': self.total_timesteps / (time.time() - start_time),
                        **training_metrics,
                    }
                )

                logger.info(f'Update {update_count}: Reward={avg_reward:.2f}, Steps={self.total_timesteps}')

            if update_count % (self.config.training.save_interval // self.config.training.rollout_steps) == 0:
                self.save_checkpoint(logger.run_dir / f'checkpoint_{update_count}.pt')

        self.save_checkpoint(logger.run_dir / 'final_model.pt')
        logger.info('Training completed')

        for env in envs:
            env.close()

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'total_timesteps': self.total_timesteps,
                'best_reward': self.best_reward,
                'config': self.config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.best_reward = checkpoint['best_reward']

    def test(self, env: CarRacingEnvironment, logger, num_episodes: int = 10) -> None:
        """Test the trained agent."""
        logger.info(f'Testing agent for {num_episodes} episodes')

        self.network.eval()
        episode_rewards = []

        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _, _ = self.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if hasattr(env, 'render'):
                    env.render()

            episode_rewards.append(episode_reward)
            logger.info(f'Test Episode {episode + 1}: Reward = {episode_reward:.2f}')

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        logger.info(f'Test Results: Average Reward = {avg_reward:.2f} ± {std_reward:.2f}')
        logger.log_metrics(
            {
                'test_reward_mean': avg_reward,
                'test_reward_std': std_reward,
                'test_episodes': num_episodes,
            }
        )

    def demo(self, env: CarRacingEnvironment, logger) -> None:
        """Run a demonstration."""
        logger.info('Running demonstration')
        self.test(env, logger, num_episodes=1)
