import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from ..network import ConvNetwork, NoisyLinear
from .base_value import ValueBasedModel


class RainbowDQN(ConvNetwork):
    """
    Rainbow DQN combining:
    - Dueling architecture
    - Noisy networks for exploration  
    - Distributional learning (Categorical DQN)
    """
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int, 
                 n_atoms: int = 51, v_min: float = -10, v_max: float = 10, noisy: bool = True):
        super().__init__(input_shape)
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.noisy = noisy
        
        LayerType = NoisyLinear if noisy else nn.Linear
        self.fc_shared = LayerType(self.conv_out_size, 512)
        self.value_stream = LayerType(512, n_atoms)
        self.advantage_stream = LayerType(512, n_actions * n_atoms)
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_to_flat(x)
        shared_features = F.relu(self.fc_shared(x))
        
        value = self.value_stream(shared_features).view(-1, 1, self.n_atoms)
        advantage = self.advantage_stream(shared_features).view(-1, self.n_actions, self.n_atoms)
        
        q_dist = self._dueling_aggregation(value, advantage)
        return F.softmax(q_dist, dim=-1)

    def reset_noise(self):
        """Reset noise parameters for noisy networks"""
        if self.noisy:
            self.fc_shared.reset_noise()
            self.value_stream.reset_noise()
            self.advantage_stream.reset_noise()


class RainbowModel(ValueBasedModel):
    """
    Rainbow DQN implementation from:
    "Rainbow: Combining Improvements in Deep Reinforcement Learning" (2018)
    """
    
    def create_networks(self):
        """Initialize Rainbow Q-networks"""
        n_atoms = self.model_cfg.get("n_atoms", 51)
        v_min = self.model_cfg.get("v_min", -10)
        v_max = self.model_cfg.get("v_max", 10)
        noisy = self.model_cfg.get("noisy", True)
        
        self.q_network = RainbowDQN(self.input_shape, self.n_actions, n_atoms, v_min, v_max, noisy).to(self.device)
        self.target_network = RainbowDQN(self.input_shape, self.n_actions, n_atoms, v_min, v_max, noisy).to(self.device)
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.is_distributional = True
            
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def get_algorithm_name(self) -> str:
        return "rainbow"

    def _custom_act(self, state) -> int:
        """Custom action selection"""
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            q_dist = self.q_network(state_tensor)
            q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=2)
            return int(q_values.argmax(dim=1).item())

    def compute_loss(self, batch):
        """Compute Rainbow (Categorical DQN) loss"""
        states, actions, rewards, next_states, dones = batch
        batch_size = states.shape[0]

        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.long)
        rewards_t = self._to_tensor(rewards, dtype=torch.float)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones.astype('uint8'), dtype=torch.bool)

        current_q_dist = self.q_network(states_t)
        current_q_dist = current_q_dist.gather(2, actions_t.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.n_atoms)).squeeze(1)
        
        next_q_dist = self.q_network(next_states_t)
        next_q_values = (next_q_dist * self.support.view(1, 1, -1)).sum(dim=2)
        next_actions = next_q_values.argmax(1)
        
        target_q_dist = self.target_network(next_states_t)
        target_q_dist = target_q_dist.gather(2, next_actions.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.n_atoms)).squeeze(1)
        
        target_support = rewards_t.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (~dones_t).unsqueeze(1)
        target_support = target_support.clamp(self.v_min, self.v_max)
        
        b = (target_support - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.n_atoms - 1)) * (l == u)] += 1
        
        m = torch.zeros_like(target_q_dist)
        offset = torch.linspace(0, ((batch_size - 1) * self.n_atoms), batch_size).long().unsqueeze(1).expand(batch_size, self.n_atoms).to(self.device)
        
        m.view(-1).index_add_(0, (l + offset).view(-1), (target_q_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (target_q_dist * (b - l.float())).view(-1))
        
        loss = -torch.sum(m * current_q_dist.log(), dim=1).mean()
        return loss