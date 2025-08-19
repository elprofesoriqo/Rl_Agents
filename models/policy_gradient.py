import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Tuple
import os
from datetime import datetime
import gymnasium as gym

from models.network import FCNetwork

class PolicyNetwork(FCNetwork):
    """
    Policy Network for REINFORCE algorithm from:
    "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
    https://doi.org/10.1007/BF00992696
        """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(state_dim, hidden_dim)
        self.action_dim = action_dim
        self.layers = self._create_fc_layers(state_dim, action_dim, [hidden_dim, hidden_dim])
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use base class method with softmax activation
        return self._forward_through_layers(x, self.layers, final_activation='softmax')


class ValueNetwork(FCNetwork):
    """
    Value Network for baseline in policy gradient methods.
    Used to reduce variance in REINFORCE algorithm.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__(state_dim, hidden_dim)
        self.layers = self._create_fc_layers(state_dim, 1, [hidden_dim, hidden_dim])
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_through_layers(x, self.layers)

from models.model import PolicyBasedAgent
from games.experiment_logger import ExperimentLogger


class PolicyGradientAgent(PolicyBasedAgent):
    """
    REINFORCE (Vanilla Policy Gradient) Agent implementation from:
    "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
    https://doi.org/10.1007/BF00992696

    Uses Monte Carlo sampling with baseline for variance reduction.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        render_enabled = self.train_cfg.get("render_every", 0) > 0
        render_mode = "human" if render_enabled else None
        self.env = gym.make(self.env_cfg["id"], render_mode=render_mode)
            
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        hidden_dim = self.model_cfg.get("hidden_dim", 128)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim).to_device()
        
        self.policy_optimizer = self._setup_optimizer(
            self.policy_net, 
            self.train_cfg.get("policy_lr", 1e-3)
        )
        
        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim, hidden_dim).to_device()
            self.value_optimizer = self._setup_optimizer(
                self.value_net, 
                self.train_cfg.get("value_lr", 1e-3)
            )
        
        experiment_name = f"pg_{self.env_cfg['id']}"
        self._setup_experiment_logging(experiment_name)
    
    def act(self, state) -> int:
        """Select action using current policy"""
        state_tensor = self._to_tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
            action_dist = Categorical(probs)
            action = action_dist.sample()
        return action.item()
    
    def _collect_episode(self, render: bool = False) -> Tuple[List[np.ndarray], List[int], List[float], float]:
        """Collect a full episode of experience"""
        states, actions, rewards = [], [], []
        obs = self.env.reset()
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        total_reward = 0.0
        
        for step in range(self.max_episode_length):
            states.append(state)
            next_state, action, reward, done = self._episode_step(self.env, state)
            
            if render:
                self.env.render()
            
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        return states, actions, rewards, total_reward
    
    
    def _update_policy(self, states: List[np.ndarray], actions: List[int], 
                      returns: torch.Tensor) -> float:
        """Update policy using REINFORCE algorithm"""
        states_tensor = self._to_tensor(np.array(states), dtype=torch.float)
        actions_tensor = self._to_tensor(actions, dtype=torch.long)
        
        if self.use_baseline:
            values = self.value_net(states_tensor).squeeze()
            advantages = returns - values.detach()
            
            # Update value network
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        else:
            advantages = returns
        
        # Compute policy loss
        probs = self.policy_net(states_tensor)
        action_dist = Categorical(probs)
        log_probs = action_dist.log_prob(actions_tensor)
        
        # REINFORCE loss: -log π(a|s) * advantage
        policy_loss = -(log_probs * advantages).mean()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        self._apply_gradient_clipping(self.policy_net)
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def _log_episode_stats(self, episode: int, reward: float, loss: float, 
                          best_reward: float, worst_reward: float,
                          rolling_avg: float, rolling_std: float):
        """Log episode statistics"""
        episode_data = {
            'reward': reward,
            'loss': loss,
            'best_reward': best_reward,
            'worst_reward': worst_reward,
            'rolling_avg': rolling_avg,
            'rolling_std': rolling_std,
            'algorithm': 'policy_gradient',
            'policy_lr': self.train_cfg.get('policy_lr', 1e-3),
            'use_baseline': self.use_baseline,
        }
        
        super()._log_episode_stats(episode, **episode_data)
    
    def train(self) -> None:
        """Main training loop"""
        num_episodes = self.train_cfg.get("num_episodes", 1000)
        save_every = self.train_cfg.get("save_every", 100)
        stats_window = self.train_cfg.get("stats_window", 100)
        render_every = self.train_cfg.get("render_every", 0)
        
        recent_rewards = []
        best_reward = float("-inf")
        worst_reward = float("inf")
        
        for episode in range(num_episodes):
            render_this = render_every and (episode % render_every == 0)
            states, actions, rewards, total_reward = self._collect_episode(render=render_this)

            returns = self._compute_returns(rewards)
            policy_loss = self._update_policy(states, actions, returns)
            recent_rewards.append(total_reward)
            if len(recent_rewards) > stats_window:
                recent_rewards.pop(0)
            
            best_reward = max(best_reward, total_reward)
            worst_reward = min(worst_reward, total_reward)
            rolling_avg = float(np.mean(recent_rewards))
            rolling_std = float(np.std(recent_rewards))
            
            # tensorboard
            self.writer.add_scalar("episode/reward", total_reward, episode)
            self.writer.add_scalar("episode/policy_loss", policy_loss, episode)
            self.writer.add_scalar("episode/episode_length", len(rewards), episode)
            self.writer.add_scalar("episode/rolling_avg", rolling_avg, episode)
            
            self._log_episode_stats(
                episode, total_reward, policy_loss, best_reward, worst_reward,
                rolling_avg, rolling_std
            )
            
            print(f"Ep {episode:4d} | R {total_reward:8.2f} | "
                  f"len {len(rewards):3d} | loss {policy_loss:7.4f} | "
                  f"avg {rolling_avg:6.2f}")
            
            if save_every and episode > 0 and episode % save_every == 0:
                checkpoint_path = os.path.join(
                    self.experiment_logger.checkpoints_dir, 
                    f"checkpoint_ep_{episode}.pth"
                )
                self.save(checkpoint_path)
            
            self.episode += 1

        self.cleanup()
    
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        networks = {"policy_net": self.policy_net}
        optimizers = {"policy_optimizer": self.policy_optimizer}
        
        if self.use_baseline:
            networks["value_net"] = self.value_net
            optimizers["value_optimizer"] = self.value_optimizer
        
        self.save_agent(path, networks, optimizers)
    
    def load(self, path: str) -> None:
        """Load model checkpoint"""
        network_names = ["policy_net"]
        optimizer_names = ["policy_optimizer"]
        
        if self.use_baseline:
            network_names.append("value_net")
            optimizer_names.append("value_optimizer")
        
        self.load_agent(path, network_names, optimizer_names)
    
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate the agent"""
        networks_to_eval = [self.policy_net]
        if self.use_baseline:
            networks_to_eval.append(self.value_net)
        
        return self.evaluate_agent(num_episodes, networks_to_eval)

