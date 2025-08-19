import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseNetwork(nn.Module, ABC):
    """
    Base class for all neural networks in the RL framework.
    Provides common functionality like weight initialization and device management.
    """
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _initialize_weights(self):
        """Initialize network weights using appropriate initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        """Calculate output size after convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self._forward_conv_layers(dummy_input)
            return int(np.prod(output.size()))
    
    @abstractmethod
    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers only"""
        pass
    
    def to_device(self):
        """Move network to appropriate device"""
        return self.to(self.device)


class ConvNetwork(BaseNetwork):
    """
    Base class for convolutional networks (like DQN variants).
    Implements common Atari CNN architecture.
    """
    
    def __init__(self, input_shape: Tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape
        
        # Standard Atari CNN architecture
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate conv output size
        self.conv_out_size = self._get_conv_out(input_shape)
        
        self._initialize_weights()
    
    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers"""
        # Normalize input to [0,1] if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor"""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return x
    
    def _forward_conv_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Common pattern: conv layers -> flatten"""
        x = self._forward_conv_layers(x)
        return x.view(x.size(0), -1)
    
    def _create_fc_head(self, hidden_dim: int, output_dim: int) -> nn.Sequential:
        """Create a simple FC head: conv_out -> hidden -> output"""
        return nn.Sequential(
            nn.Linear(self.conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _create_dueling_heads(self, hidden_dim: int, n_actions: int) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Create dueling architecture heads: shared -> value & advantage"""
        shared = nn.Sequential(
            nn.Linear(self.conv_out_size, hidden_dim),
            nn.ReLU()
        )
        value_head = nn.Linear(hidden_dim, 1)
        advantage_head = nn.Linear(hidden_dim, n_actions)
        return shared, value_head, advantage_head
        
    def _dueling_aggregation(self, value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """Standard dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))"""
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class FCNetwork(BaseNetwork):
    """
    Base class for fully connected networks (like policy/value networks).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Not applicable for FC networks"""
        return x
    
    def _create_fc_layers(self, input_dim: int, output_dim: int, 
                         hidden_dims: list = None) -> nn.ModuleList:
        """Create fully connected layers with specified dimensions"""
        if hidden_dims is None:
            hidden_dims = [self.hidden_dim, self.hidden_dim]
        
        layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return layers
    
    def _forward_through_layers(self, x: torch.Tensor, layers: nn.ModuleList, 
                               final_activation: Optional[str] = None) -> torch.Tensor:
        """Forward through a list of layers with ReLU between them"""
        for i, layer in enumerate(layers[:-1]):
            x = F.relu(layer(x))
        
        # Final layer
        x = layers[-1](x)
        
        # Optional final activation
        if final_activation == 'softmax':
            return F.softmax(x, dim=-1)
        elif final_activation == 'tanh':
            return torch.tanh(x)
        elif final_activation == 'sigmoid':
            return torch.sigmoid(x)
        
        return x


class NoisyLinear(nn.Module):
    """
    Noisy Networks for Exploration from:
    "Noisy Networks for Exploration" (ICLR, 2018)
    https://arxiv.org/abs/1706.10295
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features**0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features**0.5)

    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorised Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(input, weight, bias)