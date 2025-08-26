import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, List, Tuple

from ..network import FCNetwork
from .base_policy import PolicyBasedModel


class PolicyNetwork(FCNetwork):
    """Policy Network for REINFORCE algorithm"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(state_dim, hidden_dim)
        self.layers = self._create_fc_layers(state_dim, action_dim, [hidden_dim, hidden_dim])
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_through_layers(x, self.layers, final_activation='softmax')


class ValueNetwork(FCNetwork):
    """Value Network for baseline in policy gradient methods"""
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__(state_dim, hidden_dim)
        self.layers = self._create_fc_layers(state_dim, 1, [hidden_dim, hidden_dim])
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_through_layers(x, self.layers)


class PolicyGradientAgent(PolicyBasedModel):
    """REINFORCE (Vanilla Policy Gradient) Agent"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        hidden_dim = self.model_cfg.get("hidden_dim", 128)
        
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.policy_optimizer = self._setup_optimizer(self.policy_net, self.train_cfg.get("policy_lr", 1e-3))
        
        if self.use_baseline:
            self.value_net = ValueNetwork(self.state_dim, hidden_dim).to(self.device)
            self.value_optimizer = self._setup_optimizer(self.value_net, self.train_cfg.get("value_lr", 1e-3))

    def get_algorithm_name(self) -> str:
        return "policy_gradient"

    def act(self, state) -> int:
        """Select action using current policy"""
        state_tensor = self._to_tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
            action_dist = Categorical(probs)
            action = action_dist.sample()
        return action.item()

    def _update_policy(self, states: List[np.ndarray], actions: List[int], returns: torch.Tensor) -> float:
        """Update policy using REINFORCE"""
        states_tensor = self._to_tensor(np.array(states), dtype=torch.float)
        actions_tensor = self._to_tensor(actions, dtype=torch.long)
        
        if self.use_baseline:
            values = self.value_net(states_tensor).squeeze()
            advantages = returns - values.detach()
            
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        else:
            advantages = returns
        
        probs = self.policy_net(states_tensor)
        action_dist = Categorical(probs)
        log_probs = action_dist.log_prob(actions_tensor)
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self._apply_gradient_clipping(self.policy_net)
        self.policy_optimizer.step()
        
        return policy_loss.item()

    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add policy-specific metrics"""
        episode_data.update({
            'policy_lr': self.train_cfg.get('policy_lr', 1e-3),
            'use_baseline': self.use_baseline,
        })
        return episode_data

    def print_episode_progress(self, episode: int, reward: float, steps: int, loss: float, rolling_avg: float):
        print(f"Ep {episode:4d} | R {reward:8.2f} | len {steps:3d} | loss {loss:7.4f} | avg {rolling_avg:6.2f}")

    def save(self, path: str) -> None:
        """Save checkpoint"""
        networks = {"policy_net": self.policy_net}
        optimizers = {"policy_optimizer": self.policy_optimizer}
        
        if self.use_baseline:
            networks["value_net"] = self.value_net
            optimizers["value_optimizer"] = self.value_optimizer
        
        self.save_agent(path, networks, optimizers)

    def load(self, path: str) -> None:
        """Load checkpoint"""
        network_names = ["policy_net"]
        optimizer_names = ["policy_optimizer"]
        
        if self.use_baseline:
            network_names.append("value_net")
            optimizer_names.append("value_optimizer")
        
        self.load_agent(path, network_names, optimizer_names)

    def _get_eval_action(self, state) -> int:
        """Get greedy action for evaluation"""
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            probs = self.policy_net(state_tensor)
            return int(probs.argmax().item())

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate the agent"""
        networks_to_eval = [self.policy_net]
        if self.use_baseline:
            networks_to_eval.append(self.value_net)
        return self.evaluate_agent(num_episodes, networks_to_eval)