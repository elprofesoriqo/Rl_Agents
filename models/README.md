# Adding New RL Algorithms

This guide explains how to easily add new reinforcement learning algorithms to the gradient_rl framework, using A2C as an example.

## Project Structure

The models directory follows a hierarchical structure:

```
models/
├── __init__.py          # Module exports and imports
├── model.py            # Base classes for all RL agents
├── base_trainer.py     # Common training patterns
├── network.py          # Neural network architectures
├── value_based/        # DQN family algorithms
│   ├── __init__.py
│   ├── base_value.py
│   ├── dqn.py
│   ├── ddqn.py
├── policy_based/       # Policy gradient algorithms
│   ├── __init__.py
│   ├── base_policy.py
│   └── vanilla_pg.py
├── actor_critic/       # Actor-critic algorithms
│   ├── __init__.py
│   ├── base_actor_critic.py
│   ├── a2c.py
│   └── adversarial_a2c.py
└── [your_algorithm].py # Your new algorithm
```

## Base Classes

The framework provides several base classes in `model.py` and specialized bases:

- **`BaseAgent`**: Abstract base for all RL agents
- **`BaseTrainer`**: Common training patterns and utilities
- **`ValueBasedModel`**: For Q-learning style algorithms (DQN, etc.)
- **`PolicyBasedModel`**: For policy gradient algorithms (REINFORCE, etc.)
- **`ActorCriticModel`**: For algorithms with both policy and value functions

## Step-by-Step Guide: Adding A2C

### 1. Create the Algorithm File

Create `models/actor_critic/a2c.py` and implement the A2C algorithm by inheriting from `ActorCriticModel`.

### 2. Update `models/__init__.py`

Add your new algorithm to the module exports:

```python
# Add to imports
from .actor_critic import A2CModel

# Add to __all__ list
__all__ = [
    # ... existing exports ...
    'A2CModel'
]
```

### 3. Create Configuration File

Create `configs/cartpole/a2c.json`:

```json
{
  "agent_type": "a2c",
  "env": {
    "id": "CartPole-v1"
  },
  "model": {
    "hidden_dim": 256
  },
  "train": {
    "num_episodes": 1000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "value_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "max_episode_length": 500,
    "save_every": 100,
    "gradient_clip": 10.0
  }
}
```

### 4. Update Training Script

In `train.py`, add your algorithm to the agent factory:

```python
def create_agent(config):
    agent_type = config["agent_type"]
    
    if agent_type == "dqn":
        return DQNModel(config)
    elif agent_type == "policy_gradient":
        return PolicyGradientAgent(config)
    elif agent_type == "a2c":  # Add this line
        return A2CModel(config)
```

## Key Implementation Tips

### 1. Choose the Right Base Class
- **ValueBasedModel**: For Q-learning algorithms (DQN, Rainbow, etc.)
- **PolicyBasedModel**: For pure policy methods (REINFORCE, PPO, etc.)
- **ActorCriticModel**: For algorithms with both policy and value functions (A2C, SAC, etc.)

### 2. Required Methods
Every agent must implement:
- `act(state) -> int`: Action selection
- `get_algorithm_name() -> str`: Algorithm name for logging
- `create_networks()`: Create required neural networks
- `compute_loss()` or equivalent: Algorithm-specific loss computation
- `save(path) -> None`: Save checkpoint
- `load(path) -> None`: Load checkpoint
- `evaluate(num_episodes) -> Tuple[float, float]`: Evaluation

### 3. Network Architecture
- Inherit from appropriate base network class in `network.py`
- Use `self._to_tensor()` for device-aware tensor conversion
- Use `self._initialize_weights()` for proper weight initialization

### 4. Experiment Logging
- Call `self._setup_experiment_logging(experiment_name)` in `__init__`
- Use `self.writer` for TensorBoard logging
- Use `self._log_episode_stats()` for structured logging

### 5. Configuration
- Access config via `self.train_cfg`, `self.env_cfg`, `self.model_cfg`
- Provide sensible defaults with `.get(key, default_value)`
- Follow existing naming conventions

## Testing Your Implementation

1. **Create config file**: `configs/cartpole/a2c.json`
2. **Run training**: `python train.py configs/cartpole/a2c.json`
3. **Check outputs**: Look for experiment directory in `experiments/`
4. **Monitor training**: Use TensorBoard to view training curves

## Common Patterns

### Network Sharing (Actor-Critic)
```python
class SharedNetwork(FCNetwork):
    def forward(self, x):
        shared = self.shared_layers(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        return policy, value
```

### Multi-Step Returns
```python
def _compute_n_step_returns(self, rewards, values, n_steps=5):
    returns = []
    for i in range(len(rewards)):
        g = 0
        for j in range(min(n_steps, len(rewards) - i)):
            g += (self.gamma ** j) * rewards[i + j]
        if i + n_steps < len(values):
            g += (self.gamma ** n_steps) * values[i + n_steps]
        returns.append(g)
    return torch.FloatTensor(returns)
```

### Experience Replay
```python
# For off-policy algorithms, use the replay buffer
from games.replay_buffer import ReplayBuffer

self.replay_buffer = ReplayBuffer(
    capacity=self.train_cfg.get("memory_size", 100000)
)
```
