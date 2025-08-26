import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, List, Tuple

from ..network import FCNetwork, ConvNetwork
from .base_actor_critic import ActorCriticModel


class ActorCriticNetwork(ConvNetwork):
    """Shared Actor-Critic Network"""
    def __init__(self, input_shape: Tuple[int, ...], action_dim: int, hidden_dim: int = 512):
        super().__init__(input_shape)
        
        self.actor_head = nn.Linear(self.conv_out_size, action_dim)
        self.critic_head = nn.Linear(self.conv_out_size, 1)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_features = self._forward_conv_to_flat(x)
        
        actor_logits = self.actor_head(conv_features)
        action_probs = F.softmax(actor_logits, dim=-1)
        
        value = self.critic_head(conv_features)
        
        return action_probs, value


class ActorCriticFC(FCNetwork):
    """Shared Actor-Critic Network for A2C (Classic Control)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(state_dim, hidden_dim)
        
        self.shared_layers = self._create_fc_layers(state_dim, hidden_dim, [hidden_dim])
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self._forward_through_layers(x, self.shared_layers)
        
        actor_logits = self.actor_head(shared_features)
        action_probs = F.softmax(actor_logits, dim=-1)
        
        value = self.critic_head(shared_features)
        
        return action_probs, value


class A2CModel(ActorCriticModel):
    """
    - Actor network for policy π(a|s)
    - Critic network for value function V(s)
    - Advantage estimation A(s,a) = Q(s,a) - V(s)
    - n-step returns for learning
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        if hasattr(self.env, 'action_space_n'):
            self.action_dim = self.env.action_space_n
        else:
            self.action_dim = self.env.action_space.n
        
        if hasattr(self.env, 'observation_space'):
            obs_space = self.env.observation_space
            if len(obs_space.shape) == 3:  # Image observations (H, W, C)
                self.input_shape = tuple(self.model_cfg.get("input_shape", obs_space.shape))
                hidden_dim = self.model_cfg.get("hidden_dim", 512)
                self.network = ActorCriticNetwork(self.input_shape, self.action_dim, hidden_dim).to(self.device)
            else:  # Vector observations
                state_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else np.prod(obs_space.shape)
                hidden_dim = self.model_cfg.get("hidden_dim", 128)
                self.network = ActorCriticFC(state_dim, self.action_dim, hidden_dim).to(self.device)
        else:
            # Default to convolutional for backwards compatibility
            self.input_shape = tuple(self.model_cfg.get("input_shape", (4, 84, 84)))
            hidden_dim = self.model_cfg.get("hidden_dim", 512)
            self.network = ActorCriticNetwork(self.input_shape, self.action_dim, hidden_dim).to(self.device)
        
        lr = self.train_cfg.get("learning_rate", 3e-4)
        self.optimizer = self._setup_optimizer(self.network, lr)

    def get_algorithm_name(self) -> str:
        return "a2c"

    def act_and_value(self, state) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy and compute value"""
        state_tensor = self._to_tensor(state, dtype=torch.float).unsqueeze(0)
        
        probs, value = self.network(state_tensor)
        
        action_dist = Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()

    def _update_networks(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """Update actor and critic networks using A2C algorithm"""
        states = self._to_tensor(np.array(trajectory['states']), dtype=torch.float)
        actions = self._to_tensor(trajectory['actions'], dtype=torch.long)
        rewards = trajectory['rewards']
        values = trajectory['values']
        log_probs = trajectory['log_probs']
        next_value = trajectory['next_value']
        
        returns, advantages = self._compute_advantages(rewards, values, next_value)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        current_probs, current_values = self.network(states)
        current_values = current_values.squeeze()
        
        returns = returns.to(self.device)
        if current_values.dim() != returns.dim():
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)
            elif current_values.dim() == 0:
                current_values = current_values.unsqueeze(0)
        
        value_loss = F.mse_loss(current_values, returns)
        
        current_dist = Categorical(current_probs)
        current_log_probs = current_dist.log_prob(actions)
        entropy = current_dist.entropy().mean()
        
        actor_loss = -(current_log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        
        total_loss = actor_loss + self.value_coef * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self._apply_gradient_clipping(self.network)
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        episode_data = super().get_episode_metrics(episode_data)
        episode_data.update({
            'learning_rate': self.train_cfg.get('learning_rate', 3e-4),
        })
        return episode_data

    def save(self, path: str) -> None:
        """Save A2C checkpoint"""
        networks = {"network": self.network}
        optimizers = {"optimizer": self.optimizer}
        self.save_agent(path, networks, optimizers)

    def load(self, path: str) -> None:
        """Load A2C checkpoint"""
        network_names = ["network"]
        optimizer_names = ["optimizer"]
        self.load_agent(path, network_names, optimizer_names)

    def _get_eval_action(self, state) -> int:
        """Get greedy action for evaluation"""
        with torch.no_grad():
            state_tensor = self._to_tensor(state, dtype=torch.float).unsqueeze(0)
            probs, _ = self.network(state_tensor)
            return int(probs.argmax().item())

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate the A2C agent"""
        networks_to_eval = [self.network]
        return self.evaluate_agent(num_episodes, networks_to_eval)