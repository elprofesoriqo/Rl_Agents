# Rl Agents

A clean, extensible reinforcement learning framework built with PyTorch. Implements classical RL algorithms with proper experiment tracking and visualization tools.

## Features

- **Multiple RL Algorithms**: DQN variants, Policy Gradient (REINFORCE), and extensible base classes for new algorithms
- **Experiment Tracking**: Structured logging with TensorBoard integration
- **Modular Design**: Clean separation between environments, models, and training logic
- **Easy Extension**: Add new algorithms with minimal boilerplate code

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run training**:
```bash
python train.py configs/pong/dqn.json
python train.py configs/cartpole/pg.json
```

3. **Monitor training**:
```bash
tensorboard --logdir experiments/
```

## Project Structure

```
├── models/              # RL algorithms and neural networks
│   ├── model.py        # Base classes (BaseAgent, etc.)
│   ├── base_trainer.py # Common training patterns
│   ├── network.py      # Neural network architectures
│   ├── value_based/    # DQN family algorithms
│   │   ├── base_value.py
│   │   ├── dqn.py
│   │   ├── ddqn.py
│   │   └── rainbow.py
│   ├── policy_based/   # Policy gradient algorithms
│   │   ├── base_policy.py
│   │   └── vanilla_pg.py
│   ├── actor_critic/   # Actor-critic algorithms
│   │   ├── base_actor_critic.py
│   │   ├── a2c.py
│   │   ├── a3c.py
│   │   └── adversarial_a2c.py
│   └── README.md      # Guide for adding new algorithms
├── games/              # Environment wrappers and utilities
│   ├── atari_env.py   # Atari environment preprocessing
│   ├── replay_buffer.py # Experience replay
│   ├── preprocessor.py # Frame preprocessing
│   └── experiment_logger.py # Structured experiment logging
├── configs/            # Algorithm configurations
│   ├── pong/
│   ├── breakout/
│   ├── skiing/
│   ├── asterix/
│   └── cartpole/
├── experiments/        # Training results and checkpoints
├── analysis/           # Experiment analysis and visualization tools
│   ├── analysis_utils.py # Analysis functions and ExperimentAnalyzer class
│   └── experiment_analysis.ipynb # Jupyter notebook for experiment comparison
└── train.py           # Main training script
```

## Algorithms

### Value-Based
- **DQN**: Deep Q-Network with experience replay
- **Double DQN**: Reduces overestimation bias
- **Rainbow DQN**: Multi-step learning, prioritized replay, noisy networks

### Policy-Based
- **REINFORCE**: Vanilla policy gradient with optional baseline

### Actor-Critic
- **A2C**: Advantage Actor-Critic
- **A3C**: Asynchronous Actor-Critic
- **Adversarial A2C**: A2C with adversarial training

### Easy to Extend
The framework provides base classes that make adding new algorithms straightforward. See `models/README.md` for a complete guide with examples.

## Configuration

Each algorithm uses JSON configuration files:

```json
{
  "agent_type": "dqn",
  "env": {
    "id": "ALE/Pong-v5"
  },
  "model": {
    "input_shape": [4, 84, 84]
  },
  "train": {
    "num_episodes": 10000,
    "learning_rate": 0.00025,
    "gamma": 0.99
  }
}
```

## Experiment Analysis

Use the provided Jupyter notebook to analyze and compare experiment results:

```bash
jupyter notebook analysis/experiment_analysis.ipynb
```

The notebook provides:
- Training curve visualization
- Algorithm comparison
- Performance metrics analysis
- Hyperparameter sensitivity analysis

## Supported Environments

- **Classic Control**: CartPole, MountainCar, Acrobot
- **Atari**: All Atari games via Gymnasium
- **Custom**: Easy to add new environments through gym interface

## Results

Experiments are automatically logged with:
- Training metrics (rewards, losses, episode lengths)
- TensorBoard visualizations
- Model checkpoints
- Configuration snapshots