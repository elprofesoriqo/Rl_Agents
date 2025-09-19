from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    name: str = 'CarRacing-v2'
    render_mode: str | None = None
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

    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    activation: str = 'relu'
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    init_method: str = 'orthogonal'


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

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    compile_model: bool = True
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = 'nccl'  # nccl, gloo, mpi


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration."""

    # Buffer settings
    capacity: int = 100000
    batch_size: int = 32
    min_size: int = 1000
    buffer_type: str = 'prioritized'  # 'prioritized' or 'uniform'

    # Prioritized replay settings
    alpha: float = 0.6  # How much prioritization is used (0 = uniform, 1 = full prioritization)
    beta_start: float = 0.4  # Initial importance sampling correction factor
    beta_steps: int = 100000  # Number of steps to anneal beta to 1.0
    epsilon: float = 1e-6  # Small constant to prevent zero priorities

    # GPU optimization
    gpu_buffer: bool = True  # Store buffer on GPU for performance


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = 'logs'
    tensorboard: bool = True
    excel: bool = True
    console_level: str = 'INFO'
    file_level: str = 'DEBUG'
    save_videos: bool = True
    video_interval: int = 10000


class Config:
    """Main configuration class."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize configuration."""
        self.env = EnvironmentConfig()
        self.network = NetworkConfig()
        self.ppo = PPOConfig()
        self.training = TrainingConfig()
        self.hpc = HPCConfig()
        self.replay_buffer = ReplayBufferConfig()
        self.logging = LoggingConfig()

        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)

        self._validate_config()

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {}
        for attr_name in ['env', 'network', 'ppo', 'training', 'hpc', 'replay_buffer', 'logging']:
            if hasattr(self, attr_name):
                section = getattr(self, attr_name)
                config_dict[attr_name] = {}
                for field in section.__dataclass_fields__.values():
                    value = getattr(section, field.name)
                    if isinstance(value, tuple):
                        value = list(value)
                    config_dict[attr_name][field.name] = value

        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.hpc.device == 'cuda' and not torch.cuda.is_available():
            print('Warning: CUDA not available, falling back to CPU')
            self.hpc.device = 'cpu'

        if self.hpc.device == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 4e9:  # Less than 4GB
                self.training.n_parallel_envs = min(self.training.n_parallel_envs, 4)
                self.ppo.batch_size = min(self.ppo.batch_size, 32)

        if self.ppo.batch_size % self.ppo.mini_batch_size != 0:
            self.ppo.mini_batch_size = self.ppo.batch_size // 4

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for attr_name in ['env', 'network', 'ppo', 'training', 'hpc', 'replay_buffer', 'logging']:
            if hasattr(self, attr_name):
                section = getattr(self, attr_name)
                config_dict[attr_name] = {
                    field.name: getattr(section, field.name) for field in section.__dataclass_fields__.values()
                }
        return config_dict

    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
