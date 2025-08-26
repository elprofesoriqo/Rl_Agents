import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from ..network import ConvNetwork
from .base_value import ValueBasedModel


class DuelingDQN(ConvNetwork):
    """
    Dueling DQN Architecture from:
    "Dueling Network Architectures for Deep Reinforcement Learning" (2015)
    
    Separates Q(s,a) into V(s) + A(s,a)
    """
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super().__init__(input_shape)
        self.shared, self.value_head, self.advantage_head = self._create_dueling_heads(512, n_actions)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_to_flat(x)
        shared_features = self.shared(x)
        
        value = self.value_head(shared_features)
        advantage = self.advantage_head(shared_features)
        
        return self._dueling_aggregation(value, advantage)


class DoubleDQNModel(ValueBasedModel):
    """
    Double DQN implementation from:
    "Deep Reinforcement Learning with Double Q-learning" (2016)
    """
    
    def create_networks(self):
        """Initialize Q-networks"""
        self.q_network = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
        self.target_network = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def get_algorithm_name(self) -> str:
        return "ddqn"

    def compute_loss(self, batch):
        """Compute Double DQN loss"""
        states, actions, rewards, next_states, dones = batch

        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.long)
        rewards_t = self._to_tensor(rewards, dtype=torch.float)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones.astype('uint8'), dtype=torch.bool)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_actions = self.q_network(next_states_t).argmax(1).detach()
        next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        target_q = rewards_t + self.gamma * next_q * (~dones_t)
        
        return F.mse_loss(current_q, target_q)