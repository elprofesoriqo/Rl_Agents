from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def init_weights(module: nn.Module, init_method: str = 'orthogonal') -> None:
    """Initialize network weights using specified method."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_method == 'orthogonal':
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for CarRacing environment.
    Optimized for 84x84 grayscale frames with 4-frame stacking.
    """

    def __init__(self, input_channels: int = 4, features_dim: int = 512) -> None:
        """Initialize CNN feature extractor.

        Args:
            input_channels: Number of input channels (frame stack size)
            features_dim: Dimension of output features

        """
        super().__init__()

        self.input_channels = input_channels
        self.features_dim = features_dim

        # Convolutional layers - based on successful CarRacing architectures
        self.conv_layers = nn.Sequential(
            # First conv layer: 84x84x4 -> 20x20x32
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # Second conv layer: 20x20x32 -> 9x9x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # Third conv layer: 9x9x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            conv_output = self.conv_layers(dummy_input)
            self.conv_output_size = conv_output.numel()

        # Fully connected layer to get desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, features_dim),
            nn.ReLU(inplace=True),
        )

        # Initialize weights
        self.apply(lambda m: init_weights(m, 'orthogonal'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN feature extractor.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature tensor of shape (batch_size, features_dim)

        """
        # Normalize input to [0, 1] if not already
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Pass through convolutional layers
        conv_features = self.conv_layers(x)

        # Flatten and pass through fully connected layer
        flattened = conv_features.view(conv_features.size(0), -1)
        return self.fc(flattened)


class ActorNetwork(nn.Module):
    """Actor network for PPO agent (policy network)."""

    def __init__(self, features_dim: int, action_dim: int, hidden_dims: list | None = None) -> None:
        """Initialize actor network.

        Args:
            features_dim: Dimension of input features from CNN
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions

        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.features_dim = features_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        input_dim = features_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            input_dim = hidden_dim

        # Output layer for action logits
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(lambda m: init_weights(m, 'orthogonal'))

        # Initialize final layer with smaller weights for better exploration
        if len(self.network) > 0:
            nn.init.orthogonal_(self.network[-1].weight, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network.

        Args:
            features: Input features from CNN

        Returns:
            Action logits

        """
        return self.network(features)

    def get_action_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Get action probabilities using softmax."""
        logits = self.forward(features)
        return F.softmax(logits, dim=-1)

    def get_log_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Get log action probabilities."""
        logits = self.forward(features)
        return F.log_softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for PPO agent (value function)."""

    def __init__(self, features_dim: int, hidden_dims: list | None = None) -> None:
        """Initialize critic network.

        Args:
            features_dim: Dimension of input features from CNN
            hidden_dims: List of hidden layer dimensions

        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.features_dim = features_dim

        # Build network layers
        layers = []
        input_dim = features_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            input_dim = hidden_dim

        # Output layer for value estimation
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(lambda m: init_weights(m, 'orthogonal'))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network.

        Args:
            features: Input features from CNN

        Returns:
            Value estimates

        """
        return self.network(features).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network with shared CNN feature extractor.
    This is the main network used by the PPO agent.
    """

    def __init__(
        self,
        input_channels: int = 4,
        action_dim: int = 9,
        features_dim: int = 512,
        hidden_dims: list | None = None,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize Actor-Critic network.

        Args:
            input_channels: Number of input channels (frame stack size)
            action_dim: Number of discrete actions
            features_dim: Dimension of CNN features
            hidden_dims: List of hidden layer dimensions for actor/critic
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0 = no dropout)

        """
        super().__init__()

        self.input_channels = input_channels
        self.action_dim = action_dim
        self.features_dim = features_dim

        # Shared CNN feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            features_dim=features_dim,
        )

        # Actor and Critic networks
        self.actor = ActorNetwork(
            features_dim=features_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )

        self.critic = CriticNetwork(
            features_dim=features_dim,
            hidden_dims=hidden_dims,
        )

        # Optional batch normalization and dropout
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the entire network.

        Args:
            observations: Input observations (batch_size, channels, height, width)

        Returns:
            Tuple of (action_logits, value_estimates)

        """
        # Extract features using shared CNN
        features = self.feature_extractor(observations)

        # Apply optional batch normalization
        if self.use_batch_norm:
            features = self.batch_norm(features)

        # Apply optional dropout
        if self.dropout_rate > 0 and self.training:
            features = self.dropout(features)

        # Get actor and critic outputs
        action_logits = self.actor(features)
        value_estimates = self.critic(features)

        return action_logits, value_estimates

    def get_action_and_value(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value estimate.

        Args:
            observations: Input observations

        Returns:
            Tuple of (action, log_prob, value)

        """
        action_logits, values = self.forward(observations)

        # Sample action from categorical distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

        return actions, log_probs, values

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Get only value estimates."""
        features = self.feature_extractor(observations)
        if self.use_batch_norm:
            features = self.batch_norm(features)
        return self.critic(features)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions.

        Args:
            observations: Input observations
            actions: Actions to evaluate

        Returns:
            Tuple of (log_probs, values, entropy)

        """
        action_logits, values = self.forward(observations)

        # Calculate action probabilities and distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # Get log probabilities and entropy
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return log_probs, values, entropy

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get CNN features for analysis."""
        return self.feature_extractor(observations)


def create_ppo_network(config) -> ActorCriticNetwork:
    """Factory function to create PPO network from configuration.

    Args:
        config: Configuration object

    Returns:
        ActorCriticNetwork instance

    """
    return ActorCriticNetwork(
        input_channels=config.env.frame_stack,
        action_dim=9,  # Fixed for our discrete action space
        features_dim=512,  # Standard for CarRacing
        hidden_dims=config.network.hidden_dims,
        use_batch_norm=config.network.use_batch_norm,
        dropout_rate=config.network.dropout_rate,
    )
