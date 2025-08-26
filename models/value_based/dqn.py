import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from ..network import ConvNetwork
from .base_value import ValueBasedModel


class NatureDQN(ConvNetwork):
    """
    DQN Network Architecture from Nature 2015 paper:
    "Human-level control through deep reinforcement learning"
    """
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super().__init__(input_shape)
        self.fc_head = self._create_fc_head(512, n_actions)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_to_flat(x)
        return self.fc_head(x)


class DQNModel(ValueBasedModel):
    """
    Deep Q-Network Model implementation from:
    "Human-level control through deep reinforcement learning" (Nature, 2015)
    """
    
    def create_networks(self):
        """Initialize Q-networks"""
        self.q_network = NatureDQN(self.input_shape, self.n_actions).to(self.device)
        self.target_network = NatureDQN(self.input_shape, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def get_algorithm_name(self) -> str:
        return "dqn"

    def compute_loss(self, batch):
        """Compute standard DQN loss"""
        states, actions, rewards, next_states, dones = batch

        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.long)
        rewards_t = self._to_tensor(rewards, dtype=torch.float)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones.astype('uint8'), dtype=torch.bool)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states_t).max(1)[0].detach()
        target_q = rewards_t + self.gamma * next_q * (~dones_t)
        
        return F.mse_loss(current_q, target_q)