import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, List, Tuple

from ..network import FCNetwork, ConvNetwork
from .base_actor_critic import ActorCriticModel



class ActorNetwork(nn.Module):
    """Actor Network for Adversarial A2C"""
    def __init__(self, input_shape: Tuple[int, ...] = None, state_dim: int = None, action_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        
        if input_shape is not None:  # Convolutional network for image inputs
            self.conv_net = ConvNetwork(input_shape)
            self.actor_head = nn.Linear(self.conv_net.conv_out_size, action_dim)
            self.is_conv = True
        elif state_dim is not None:  # Fully connected network for vector inputs
            self.fc_net = FCNetwork(state_dim, hidden_dim)
            self.fc_layers = self.fc_net._create_fc_layers(state_dim, hidden_dim, [hidden_dim])
            self.actor_head = nn.Linear(hidden_dim, action_dim)
            self.is_conv = False
        else:
            raise ValueError("Either input_shape or state_dim must be provided")
            
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            conv_features = self.conv_net._forward_conv_to_flat(x)
            actor_logits = self.actor_head(conv_features)
        else:
            fc_features = self.fc_net._forward_through_layers(x, self.fc_layers)
            actor_logits = self.actor_head(fc_features)
        return F.softmax(actor_logits, dim=-1)
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)


class CriticNetwork(nn.Module):
    """Critic Network for Adversarial A2C"""
    def __init__(self, input_shape: Tuple[int, ...] = None, state_dim: int = None, hidden_dim: int = 512):
        super().__init__()
        
        if input_shape is not None:  # Convolutional network for image inputs
            self.conv_net = ConvNetwork(input_shape)
            self.critic_head = nn.Linear(self.conv_net.conv_out_size, 1)
            self.is_conv = True
        elif state_dim is not None:  # Fully connected network for vector inputs
            self.fc_net = FCNetwork(state_dim, hidden_dim)
            self.fc_layers = self.fc_net._create_fc_layers(state_dim, hidden_dim, [hidden_dim])
            self.critic_head = nn.Linear(hidden_dim, 1)
            self.is_conv = False
        else:
            raise ValueError("Either input_shape or state_dim must be provided")
            
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            conv_features = self.conv_net._forward_conv_to_flat(x)
            return self.critic_head(conv_features)
        else:
            fc_features = self.fc_net._forward_through_layers(x, self.fc_layers)
            return self.critic_head(fc_features)
            
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)


class AdversarialNetwork(nn.Module):
    """Adversarial Network that generates perturbations"""
    def __init__(self, input_shape: Tuple[int, ...] = None, state_dim: int = None, hidden_dim: int = 512):
        super().__init__()
        
        if input_shape is not None:  # Convolutional network for image inputs
            self.conv_net = ConvNetwork(input_shape)
            self.adversarial_head = nn.Linear(self.conv_net.conv_out_size, np.prod(input_shape))
            self.input_shape = input_shape
            self.is_conv = True
        elif state_dim is not None:  # Fully connected network for vector inputs
            self.fc_net = FCNetwork(state_dim, hidden_dim)
            self.fc_layers = self.fc_net._create_fc_layers(state_dim, hidden_dim, [hidden_dim])
            self.adversarial_head = nn.Linear(hidden_dim, state_dim)
            self.is_conv = False
        else:
            raise ValueError("Either input_shape or state_dim must be provided")
            
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        if self.is_conv:
            conv_features = self.conv_net._forward_conv_to_flat(x)
            perturbation_flat = self.adversarial_head(conv_features)
            perturbation = perturbation_flat.view(batch_size, *self.input_shape)
        else:
            fc_features = self.fc_net._forward_through_layers(x, self.fc_layers)
            perturbation = self.adversarial_head(fc_features)
            
        return torch.tanh(perturbation)
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)


class AdversarialA2CModel(ActorCriticModel):
    """
    Adversarial Advantage Actor-Critic (Adversarial A2C) implementation.
    
    This algorithm extends A2C with adversarial training for robustness:
    - Main agent (actor-critic) learns to maximize rewards
    - Adversarial agent learns to generate perturbations that minimize main agent's performance
    - Joint minimax training makes the main agent robust to perturbations
    - Uses adversarial coefficient to balance between performance and robustness
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        if hasattr(self.env, 'action_space_n'):
            self.action_dim = self.env.action_space_n
        else:
            self.action_dim = self.env.action_space.n
        hidden_dim = self.model_cfg.get("hidden_dim", 512)
        
        if hasattr(self.env, 'observation_space'):
            obs_space = self.env.observation_space
            if len(obs_space.shape) == 3:  # Image observations
                self.input_shape = tuple(self.model_cfg.get("input_shape", obs_space.shape))
                self.actor = ActorNetwork(input_shape=self.input_shape, action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
                self.critic = CriticNetwork(input_shape=self.input_shape, hidden_dim=hidden_dim).to(self.device)
                self.adversary = AdversarialNetwork(input_shape=self.input_shape, hidden_dim=hidden_dim).to(self.device)
                self.is_conv = True
            else:  # Vector observations
                self.state_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else np.prod(obs_space.shape)
                self.actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
                self.critic = CriticNetwork(state_dim=self.state_dim, hidden_dim=hidden_dim).to(self.device)
                self.adversary = AdversarialNetwork(state_dim=self.state_dim, hidden_dim=hidden_dim).to(self.device)
                self.is_conv = False
        else:
            # Default to convolutional for backwards compatibility
            self.input_shape = tuple(self.model_cfg.get("input_shape", (4, 84, 84)))
            self.actor = ActorNetwork(input_shape=self.input_shape, action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
            self.critic = CriticNetwork(input_shape=self.input_shape, hidden_dim=hidden_dim).to(self.device)
            self.adversary = AdversarialNetwork(input_shape=self.input_shape, hidden_dim=hidden_dim).to(self.device)
            self.is_conv = True
        
        # Adversarial training parameters
        self.adv_coef = self.train_cfg.get("adversarial_coef", 0.1)  # Balance between performance and robustness
        self.adv_eps = self.train_cfg.get("adversarial_eps", 0.1)   # Maximum perturbation magnitude
        self.adv_update_freq = self.train_cfg.get("adv_update_freq", 1)  # How often to update adversary
        self.adv_warmup_episodes = self.train_cfg.get("adv_warmup_episodes", 50)  # Episodes before adversarial training starts
        
        actor_lr = self.train_cfg.get("actor_lr", 3e-4)
        critic_lr = self.train_cfg.get("critic_lr", 1e-3)
        adv_lr = self.train_cfg.get("adversarial_lr", 1e-4)
        
        self.actor_optimizer = self._setup_optimizer(self.actor, actor_lr)
        self.critic_optimizer = self._setup_optimizer(self.critic, critic_lr)
        self.adversary_optimizer = self._setup_optimizer(self.adversary, adv_lr)
        
        self.episode_count = 0
        self.adversarial_training_active = False

    def get_algorithm_name(self) -> str:
        return "adversarial_a2c"

    def _apply_adversarial_perturbation(self, state: torch.Tensor) -> torch.Tensor:
        """Apply adversarial perturbation if adversarial training is active"""
        if not self.adversarial_training_active:
            return state
            
        with torch.no_grad():
            perturbation = self.adversary(state)
            perturbation = perturbation * self.adv_eps
            perturbed_state = state + perturbation
            
            if self.is_conv:
                # For image observations, clip to [0, 1] (assuming normalized frames)
                perturbed_state = torch.clamp(perturbed_state, 0.0, 1.0)
            else:
                # For vector observations, no specific clipping needed
                pass
            
        return perturbed_state

    def act_and_value(self, state) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy and compute value (with potential adversarial perturbation)"""
        state_tensor = self._to_tensor(state, dtype=torch.float).unsqueeze(0)
        
        if self.training and self.adversarial_training_active:
            state_tensor = self._apply_adversarial_perturbation(state_tensor)
        
        probs = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        action_dist = Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()

    def _update_networks(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """Update actor, critic, and adversarial networks"""
        states = self._to_tensor(np.array(trajectory['states']), dtype=torch.float)
        actions = self._to_tensor(trajectory['actions'], dtype=torch.long)
        rewards = trajectory['rewards']
        values = trajectory['values']
        log_probs = trajectory['log_probs']
        next_value = trajectory['next_value']
        
        returns, advantages = self._compute_advantages(rewards, values, next_value)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        loss_dict = {}
        
        if self.adversarial_training_active:
            perturbed_states = self._apply_adversarial_perturbation(states)
        else:
            perturbed_states = states
        
        current_values = self.critic(perturbed_states).squeeze()
        value_loss = F.mse_loss(current_values, returns)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self._apply_gradient_clipping(self.critic)
        self.critic_optimizer.step()
        
        current_probs = self.actor(perturbed_states)
        current_dist = Categorical(current_probs)
        current_log_probs = current_dist.log_prob(actions)
        entropy = current_dist.entropy().mean()
        
        # Actor loss = -log_prob * advantage - entropy_bonus
        actor_loss = -(current_log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._apply_gradient_clipping(self.actor)
        self.actor_optimizer.step()
        
        loss_dict.update({
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item()
        })
        
        if self.adversarial_training_active and self.episode_count % self.adv_update_freq == 0:
            adversarial_perturbations = self.adversary(states)
            adversarial_perturbations = adversarial_perturbations * self.adv_eps
            adversarial_states = states + adversarial_perturbations
            if self.is_conv:
                adversarial_states = torch.clamp(adversarial_states, 0.0, 1.0)
            
            with torch.no_grad():
                adv_probs = self.actor(adversarial_states)
                adv_values = self.critic(adversarial_states).squeeze()
            
            adversarial_loss = adv_values.mean()
            
            perturbation_penalty = torch.mean(adversarial_perturbations ** 2)
            adversarial_loss += 0.01 * perturbation_penalty
            
            self.adversary_optimizer.zero_grad()
            adversarial_loss.backward()
            self._apply_gradient_clipping(self.adversary)
            self.adversary_optimizer.step()
            
            loss_dict['adversarial_loss'] = adversarial_loss.item()
            loss_dict['perturbation_norm'] = torch.mean(torch.abs(adversarial_perturbations)).item()
        
        return loss_dict

    def run_episode(self, episode: int, render_every: int) -> Tuple[float, int, List[float]]:
        """Run episode with adversarial training logic"""
        self.episode_count = episode
        
        if episode >= self.adv_warmup_episodes:
            self.adversarial_training_active = True
        
        return super().run_episode(episode, render_every)

    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add Adversarial A2C-specific metrics"""
        episode_data = super().get_episode_metrics(episode_data)
        episode_data.update({
            'actor_lr': self.train_cfg.get('actor_lr', 3e-4),
            'critic_lr': self.train_cfg.get('critic_lr', 1e-3),
            'adversarial_lr': self.train_cfg.get('adversarial_lr', 1e-4),
            'adversarial_coef': self.adv_coef,
            'adversarial_eps': self.adv_eps,
            'adversarial_active': self.adversarial_training_active,
        })
        return episode_data

    def print_episode_progress(self, episode: int, reward: float, steps: int, loss: float, rolling_avg: float):
        """Adversarial A2C specific episode progress format"""
        adv_status = "ADV" if self.adversarial_training_active else "WARM"
        print(f"Ep {episode:4d} | R {reward:8.2f} | len {steps:3d} | loss {loss:7.4f} | avg {rolling_avg:6.2f} | {adv_status}")

    def save(self, path: str) -> None:
        """Save Adversarial A2C checkpoint"""
        networks = {
            "actor": self.actor, 
            "critic": self.critic, 
            "adversary": self.adversary
        }
        optimizers = {
            "actor_optimizer": self.actor_optimizer, 
            "critic_optimizer": self.critic_optimizer,
            "adversary_optimizer": self.adversary_optimizer
        }
        additional_data = {
            "episode_count": self.episode_count,
            "adversarial_training_active": self.adversarial_training_active
        }
        self.save_agent(path, networks, optimizers, additional_data)

    def load(self, path: str) -> None:
        """Load Adversarial A2C checkpoint"""
        network_names = ["actor", "critic", "adversary"]
        optimizer_names = ["actor_optimizer", "critic_optimizer", "adversary_optimizer"]
        self.load_agent(path, network_names, optimizer_names)

    def _get_eval_action(self, state) -> int:
        """Get greedy action for evaluation (without adversarial perturbations)"""
        with torch.no_grad():
            state_tensor = self._to_tensor(state, dtype=torch.float).unsqueeze(0)
            probs = self.actor(state_tensor)  # No perturbation during evaluation
            return int(probs.argmax().item())

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate the Adversarial A2C agent"""
        # Temporarily disable adversarial training during evaluation
        original_training_state = self.training
        original_adversarial_state = self.adversarial_training_active
        
        self.adversarial_training_active = False
        networks_to_eval = [self.actor, self.critic]
        
        result = self.evaluate_agent(num_episodes, networks_to_eval)
        
        # Restore original states
        self.training = original_training_state
        self.adversarial_training_active = original_adversarial_state
        
        return result

    def evaluate_robustness(self, num_episodes: int, perturbation_strength: float = None) -> Tuple[float, float]:
        """Evaluate agent's robustness against adversarial perturbations"""
        if perturbation_strength is None:
            perturbation_strength = self.adv_eps
            
        original_eps = self.adv_eps
        original_adversarial_state = self.adversarial_training_active
        
        self.adv_eps = perturbation_strength
        self.adversarial_training_active = True
        
        networks_to_eval = [self.actor, self.critic]
        result = self.evaluate_agent(num_episodes, networks_to_eval)
        
        # Restore original settings
        self.adv_eps = original_eps
        self.adversarial_training_active = original_adversarial_state
        
        return result