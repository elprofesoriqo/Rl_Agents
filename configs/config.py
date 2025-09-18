import yaml
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: str = "CarRacing-v2"
    render_mode: Optional[str] = None
    frame_stack: int = 4
    frame_skip: int = 4
    img_size: tuple = (84, 84)
    grayscale: bool = True
    normalize: bool = True
    discrete_actions: bool = True
    reward_shaping: bool = True
    max_episode_steps: int = 1000


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    init_method: str = "orthogonal"


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    mini_batch_size: int = 32


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_timesteps: int = 1_000_000
    n_parallel_envs: int = 8
    rollout_steps: int = 2048
    save_interval: int = 10000
    eval_interval: int = 5000
    eval_episodes: int = 10
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


@dataclass
class HPCConfig:
    """HPC optimization configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    compile_model: bool = True
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"  # nccl, gloo, mpi


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration."""
    capacity: int = 100000
    use_per: bool = True  # Prioritized Experience Replay
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.001
    epsilon: float = 1e-6
    gpu_buffer: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    tensorboard: bool = True
    excel: bool = True
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    save_videos: bool = True
    video_interval: int = 10000