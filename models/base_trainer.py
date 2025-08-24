import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from .model import BaseAgent

class BaseTrainer(BaseAgent):
    """
    Base trainer class that handles common training patterns.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.env_cfg = config["env"]
        self.train_cfg = config["train"]
        self.model_cfg = config.get("model", {})
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = self.train_cfg.get("gamma", 0.99)
        self.episode = 0
        
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name for logging."""
        pass
    
    @abstractmethod
    def run_episode(self, episode: int, render_every: int) -> Tuple[float, int, List[float]]:
        """Run single episode and return (reward, steps, losses)."""
        pass
    
    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override to add algorithm-specific metrics."""
        return episode_data

    def train(self) -> None:
        """Generic training loop for all algorithms."""
        num_episodes = self.train_cfg.get("num_episodes", 1000)
        save_every = self.train_cfg.get("save_every", 100)
        render_every = self.train_cfg.get("render_every", 0)
        stats_window = int(self.train_cfg.get("stats_window", 100))

        print(f"Training {self.get_algorithm_name()} on {self.env_cfg['id']}")
        print(f"Device: {self.device} | Episodes: {num_episodes}")
        if hasattr(self, 'writer'):
            print(f"Log dir: {self.writer.log_dir}")
        print("=" * 60)

        recent_rewards = []
        best_reward = float("-inf")
        worst_reward = float("inf")

        for ep in range(num_episodes):
            episode_reward, episode_steps, episode_losses = self.run_episode(ep, render_every)
            
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > stats_window:
                recent_rewards.pop(0)
            
            best_reward = max(best_reward, episode_reward)
            worst_reward = min(worst_reward, episode_reward)
            rolling_avg = float(np.mean(recent_rewards))
            rolling_std = float(np.std(recent_rewards))
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

            if hasattr(self, 'writer'):
                self.writer.add_scalar("episode/reward", episode_reward, ep)
                self.writer.add_scalar("episode/steps", episode_steps, ep)
                self.writer.add_scalar("episode/avg_loss", avg_loss, ep)
                if hasattr(self, 'rolling_avg'):
                    self.writer.add_scalar("episode/rolling_avg", rolling_avg, ep)
            
                episode_data = {
                'episode_steps': episode_steps,
                'reward': episode_reward,
                'loss': avg_loss,
                'best_reward': best_reward,
                'worst_reward': worst_reward,
                'rolling_avg': rolling_avg,
                'rolling_std': rolling_std,
                'algorithm': self.get_algorithm_name(),
            }
            
            episode_data = self.get_episode_metrics(episode_data)
            self._log_episode_stats(ep, **episode_data)
            
            self.print_episode_progress(ep, episode_reward, episode_steps, avg_loss, rolling_avg)

            if save_every and ep > 0 and ep % save_every == 0:
                checkpoint_path = os.path.join(
                    self.experiment_logger.checkpoints_dir, 
                    f"checkpoint_ep_{ep}.pth"
                )
                self.save(checkpoint_path)
                        
            self.episode += 1

        self.cleanup()

    def print_episode_progress(self, episode: int, reward: float, steps: int, loss: float, rolling_avg: float):
        """Print episode progress."""
        print(f"Ep {episode:4d} | R {reward:8.2f} | steps {steps:5d} | loss {loss:7.4f} | avg {rolling_avg:6.2f}")

    def save_networks_and_optimizers(self, networks: Dict[str, torch.nn.Module], 
                                   optimizers: Dict[str, torch.optim.Optimizer],
                                   additional_data: Optional[Dict[str, Any]] = None) -> None:
        self.save_agent(self.get_checkpoint_path(), networks, optimizers, additional_data)

    def load_networks_and_optimizers(self, network_names: List[str], optimizer_names: List[str]) -> None:
        self.load_agent(self.get_checkpoint_path(), network_names, optimizer_names)

    def get_checkpoint_path(self, episode: Optional[int] = None) -> str:
        if episode is None:
            episode = self.episode
        return os.path.join(
            self.experiment_logger.checkpoints_dir,
            f"checkpoint_ep_{episode}.pth"
        )

    def _setup_experiment_logging(self, experiment_name: str) -> None:
        """Setup experiment logging infrastructure."""
        from games.experiment_logger import ExperimentLogger
        
        self.experiment_logger = ExperimentLogger(
            experiment_name=experiment_name,
            base_dir=self.train_cfg.get("experiments_dir", "experiments")
        )
        
        logdir = os.path.join(self.experiment_logger.experiment_dir, "tensorboard")
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        
        self.experiment_logger.log_config(self.config)

    def _log_episode_stats(self, episode: int, **kwargs) -> None:
        base_episode_data = {
            'episode': episode,
            'environment': self.env_cfg['id'],
            'gamma': self.gamma,
        }
        base_episode_data.update(kwargs)
        self.experiment_logger.log_episode(base_episode_data)

    def _setup_optimizer(self, network: nn.Module, lr: Optional[float] = None) -> torch.optim.Optimizer:
        """Setup optimizer for the given network."""
        if lr is None:
            lr = self.train_cfg.get("learning_rate", 2.5e-4)
            
        if self.train_cfg.get("use_rmsprop", False):
            alpha = self.train_cfg.get("rmsprop_alpha", 0.95)
            momentum = self.train_cfg.get("rmsprop_momentum", 0.95)
            eps = self.train_cfg.get("rmsprop_eps", 0.01)
            
            return optim.RMSprop(
                network.parameters(),
                lr=lr,
                alpha=alpha,
                momentum=momentum,
                eps=eps
            )
        else:
            return optim.Adam(network.parameters(), lr=lr)

    def _apply_gradient_clipping(self, network: nn.Module) -> None:
        """Apply gradient clipping to prevent exploding gradients."""
        grad_clip = self.train_cfg.get("gradient_clip", 10.0)
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)

    def _to_tensor(self, data, dtype=None) -> torch.Tensor:
        """Convert numpy array or list to tensor on correct device."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
            
        if dtype is not None:
            tensor = tensor.to(dtype)
            
        return tensor.to(self.device)

    def save_agent(self, path: str, networks: Dict[str, nn.Module], 
                   optimizers: Dict[str, torch.optim.Optimizer], 
                   additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Common save pattern for all agents."""
        checkpoint_path = self._save_checkpoint(
            self.episode, networks, optimizers, additional_data
        )
        
        if path != checkpoint_path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy2(checkpoint_path, path)

    def _save_checkpoint(self, episode: int, networks: Dict[str, nn.Module], 
                        optimizers: Dict[str, torch.optim.Optimizer],
                        additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Save a checkpoint with networks and optimizers."""
        checkpoint_path = os.path.join(
            self.experiment_logger.checkpoints_dir, 
            f"checkpoint_ep_{episode}.pth"
        )
        
        checkpoint_data = {
            "episode": episode,
            "config": self.config,
        }
        
        for name, network in networks.items():
            checkpoint_data[f"{name}_state_dict"] = network.state_dict()
            
        for name, optimizer in optimizers.items():
            checkpoint_data[f"{name}_optimizer"] = optimizer.state_dict()
            
        if additional_data:
            checkpoint_data.update(additional_data)
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint_data, checkpoint_path)
        
        return checkpoint_path

    def load_agent(self, path: str, network_names: List[str], 
                   optimizer_names: List[str]) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        
        for name in network_names:
            if hasattr(self, name):
                network = getattr(self, name)
                state_dict_key = f"{name}_state_dict"
                if state_dict_key in checkpoint:
                    network.load_state_dict(checkpoint[state_dict_key])
                elif name in checkpoint:
                    network.load_state_dict(checkpoint[name])
        
        for name in optimizer_names:
            if hasattr(self, name):
                optimizer = getattr(self, name)
                optimizer_key = f"{name}_optimizer"
                if optimizer_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[optimizer_key])
                elif name in checkpoint:
                    optimizer.load_state_dict(checkpoint[name])
        
        self.episode = checkpoint.get("episode", 0)
        if "steps_done" in checkpoint:
            self.steps_done = checkpoint["steps_done"]
        
        return checkpoint

    def evaluate_agent(self, num_episodes: int, networks_to_eval: List[nn.Module]) -> Tuple[float, float]:
        """Common evaluation pattern for all agents."""
        training_modes = []
        for network in networks_to_eval:
            training_modes.append(network.training)
            network.eval()
        
        env_or_game = getattr(self, 'env', None) or getattr(self, 'game', None)
        max_steps = getattr(self, 'max_episode_length', None)
        
        result = self._evaluate_agent(num_episodes, env_or_game, max_steps)
        
        for network, training_mode in zip(networks_to_eval, training_modes):
            network.train(training_mode)
        
        return result

    def _evaluate_agent(self, num_episodes: int, env_or_game, max_steps: int = None) -> Tuple[float, float]:
        """Common evaluation pattern."""
        rewards = []
        
        for _ in range(num_episodes):
            if hasattr(env_or_game, 'reset'):
                obs = env_or_game.reset()
                if isinstance(obs, tuple):
                    state = obs[0]
                else:
                    state = obs
            else:
                state = env_or_game.reset()
                
            total_reward = 0.0
            steps = 0
            
            while True:
                action = self._get_eval_action(state)
                
                step_result = env_or_game.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    next_state, reward, done, _ = step_result
                else:
                    next_state, reward, done = step_result
                
                state = next_state
                total_reward += float(reward)
                steps += 1
                
                if done or (max_steps and steps >= max_steps):
                    break
                    
            rewards.append(total_reward)
            
        return float(np.mean(rewards)), float(np.std(rewards))

    def cleanup(self) -> None:
        if hasattr(self, 'writer'):
            self.writer.close()
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'game'):
            self.game.close()

    @abstractmethod
    def _get_eval_action(self, state) -> int:
        """Get action for evaluation."""
        pass