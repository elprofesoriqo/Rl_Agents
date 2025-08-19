# Gradient RL

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
python train.py --config configs/dqn_pong.json
python train.py --config configs/pg_cartpole.json
```

3. **Monitor training**:
```bash
tensorboard --logdir experiments/
```

## Project Structure

```
├── models/              # RL algorithms and neural networks
│   ├── model.py        # Base classes (BaseAgent, ValueBasedAgent, etc.)
│   ├── network.py      # Neural network architectures
│   ├── dqn.py         # DQN family algorithms
│   ├── policy_gradient.py # REINFORCE implementation
│   └── README.md      # Guide for adding new algorithms
├── games/              # Environment wrappers and utilities
│   ├── atari_env.py   # Atari environment preprocessing
│   ├── replay_buffer.py # Experience replay
│   └── experiment_logger.py # Structured experiment logging
├── configs/            # Algorithm configurations
├── experiments/        # Training results and checkpoints
├── analysis/           # Experiment analysis and visualization tools
│   ├── analysis_utils.py # Analysis functions and ExperimentAnalyzer class
│   └── experiment_analysis.ipynb # Jupyter notebook for experiment comparison
└── train.py           # Main training script
```

## Algorithms

### Value-Based
- **DQN**: Deep Q-Network with experience replay
- **Nature DQN**: Convolutional architecture from Nature 2015 paper
- **Dueling DQN**: Separate value and advantage streams
- **Rainbow DQN**: Multi-step learning, prioritized replay, noisy networks

### Policy-Based
- **REINFORCE**: Vanilla policy gradient with optional baseline

### Easy to Extend
The framework provides base classes that make adding new algorithms straightforward. See `models/README.md` for a complete guide with A2C example.

## Configuration

Each algorithm uses JSON configuration files:

```json
{
  "env": {
    "id": "CartPole-v1"
  },
  "model": {
    "hidden_dim": 128
  },
  "train": {
    "num_episodes": 1000,
    "learning_rate": 1e-3,
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

