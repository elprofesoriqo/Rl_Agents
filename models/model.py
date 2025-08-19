import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
import os
from datetime import datetime

from games.experiment_logger import ExperimentLogger

class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
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
    def train(self) -> None:
        """Main training loop"""
        pass

    @abstractmethod
    def act(self, state) -> int:
        """Select action given current state"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint"""
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate agent performance"""
        pass

    def _setup_experiment_logging(self, experiment_name: str) -> None:
        """Setup experiment logging infrastructure"""
        from games.experiment_logger import ExperimentLogger
        
        self.experiment_logger = ExperimentLogger(
            experiment_name=experiment_name,
            base_dir=self.train_cfg.get("experiments_dir", "experiments")
        )
        
        # Setup tensorboard logging
        logdir = os.path.join(self.experiment_logger.experiment_dir, "tensorboard")
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        
        self.experiment_logger.log_config(self.config)

    def _setup_optimizer(self, network: nn.Module, lr: Optional[float] = None) -> torch.optim.Optimizer:
        """Setup optimizer for the given network"""
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
        """Apply gradient clipping to prevent exploding gradients"""
        grad_clip = self.train_cfg.get("gradient_clip", 10.0)
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)

    def _log_episode_stats(self, episode: int, **kwargs) -> None:
        """Log episode statistics to experiment logger"""
        base_episode_data = {
            'episode': episode,
            'environment': self.env_cfg['id'],
            'gamma': self.gamma,
        }
        base_episode_data.update(kwargs)
        self.experiment_logger.log_episode(base_episode_data)

    def _save_checkpoint(self, episode: int, networks: Dict[str, nn.Module], 
                        optimizers: Dict[str, torch.optim.Optimizer],
                        additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Save a checkpoint with networks and optimizers"""
        checkpoint_path = os.path.join(
            self.experiment_logger.checkpoints_dir, 
            f"checkpoint_ep_{episode}.pth"
        )
        
        checkpoint_data = {
            "episode": episode,
            "config": self.config,
        }
        
        # network states
        for name, network in networks.items():
            checkpoint_data[f"{name}_state_dict"] = network.state_dict()
            
        # optimizer states
        for name, optimizer in optimizers.items():
            checkpoint_data[f"{name}_optimizer"] = optimizer.state_dict()
            
        if additional_data:
            checkpoint_data.update(additional_data)
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint_data, checkpoint_path)
        
        return checkpoint_path

    def save_agent(self, path: str, networks: Dict[str, nn.Module], 
                   optimizers: Dict[str, torch.optim.Optimizer], 
                   additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Common save pattern for all agents"""
        checkpoint_path = self._save_checkpoint(
            self.episode, networks, optimizers, additional_data
        )
        
        if path != checkpoint_path:
            import shutil
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy2(checkpoint_path, path)

    def load_agent(self, path: str, network_names: List[str], 
                   optimizer_names: List[str]) -> Dict[str, Any]:
        """Common load pattern for all agents"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load networks
        for name in network_names:
            if hasattr(self, name):
                network = getattr(self, name)
                state_dict_key = f"{name}_state_dict"
                if state_dict_key in checkpoint:
                    network.load_state_dict(checkpoint[state_dict_key])
                elif name in checkpoint:  # Legacy format
                    network.load_state_dict(checkpoint[name])
        
        # Load optimizers
        for name in optimizer_names:
            if hasattr(self, name):
                optimizer = getattr(self, name)
                optimizer_key = f"{name}_optimizer"
                if optimizer_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[optimizer_key])
                elif name in checkpoint:  # Legacy format
                    optimizer.load_state_dict(checkpoint[name])
        
        # Load episode and additional data
        self.episode = checkpoint.get("episode", 0)
        if "steps_done" in checkpoint:
            self.steps_done = checkpoint["steps_done"]
        
        return checkpoint

    def evaluate_agent(self, num_episodes: int, networks_to_eval: List[nn.Module]) -> Tuple[float, float]:
        """Common evaluation pattern for all agents"""
        # Set networks to eval mode
        training_modes = []
        for network in networks_to_eval:
            training_modes.append(network.training)
            network.eval()
        
        # Get environment for evaluation
        env_or_game = getattr(self, 'env', None) or getattr(self, 'game', None)
        max_steps = getattr(self, 'max_episode_length', None)
        
        # Perform evaluation
        result = self._evaluate_agent(num_episodes, env_or_game, max_steps)
        
        # Restore original training modes
        for network, training_mode in zip(networks_to_eval, training_modes):
            network.train(training_mode)
        
        return result

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'writer'):
            self.writer.close()
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'game'):
            self.game.close()
    
    def _to_tensor(self, data, dtype=None) -> torch.Tensor:
        """Convert numpy array or list to tensor on correct device"""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
            
        if dtype is not None:
            tensor = tensor.to(dtype)
            
        return tensor.to(self.device)
    
    def _numpy_to_tensor_dict(self, numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert dict of numpy arrays to tensors"""
        return {key: self._to_tensor(value) for key, value in numpy_dict.items()}
    
    def _get_checkpoint_path(self, episode: int) -> str:
        """Get standard checkpoint path"""
        return os.path.join(
            self.experiment_logger.checkpoints_dir, 
            f"checkpoint_ep_{episode}.pth"
        )
    
    def _evaluate_agent(self, num_episodes: int, env_or_game, max_steps: int = None) -> Tuple[float, float]:
        """Common evaluation pattern"""
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
                if len(step_result) == 5:  # Gymnasium format
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:  # Legacy gym format  
                    next_state, reward, done, _ = step_result
                else:  # Simple format
                    next_state, reward, done = step_result
                
                state = next_state
                total_reward += float(reward)
                steps += 1
                
                if done or (max_steps and steps >= max_steps):
                    break
                    
            rewards.append(total_reward)
            
        return float(np.mean(rewards)), float(np.std(rewards))
    
    @abstractmethod
    def _get_eval_action(self, state) -> int:
        """Get action for evaluation"""
        pass


class ValueBasedAgent(BaseAgent):
    """
    Base class for value-based RL agents (like DQN variants).
    Provides common functionality for Q-learning agents.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.batch_size = self.train_cfg.get("batch_size", 32)
        self.target_update = self.train_cfg.get("target_update", 10_000)
        self.epsilon_start = self.train_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = self.train_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = self.train_cfg.get("epsilon_decay", 1_000_000)
        self.warmup_steps = self.train_cfg.get("warmup_steps", 1000)
        
        self.steps_done = 0

    def epsilon(self) -> float:
        """Current epsilon value for epsilon-greedy exploration"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    @abstractmethod
    def _optimize(self) -> Optional[float]:
        """Perform one optimization step"""
        pass

    def _update_target_network(self, main_network: nn.Module, 
                              target_network: nn.Module) -> None:
        """Update target network with main network weights"""
        if self.steps_done % self.target_update == 0:
            target_network.load_state_dict(main_network.state_dict())


class PolicyBasedAgent(BaseAgent):
    """
    Base class for policy-based RL agents (like REINFORCE, A2C, PPO).
    Provides common functionality for policy gradient agents.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.max_episode_length = self.train_cfg.get("max_episode_length", 1000)
        self.use_baseline = self.train_cfg.get("use_baseline", True)

    def _compute_returns(self, rewards: List[float], 
                        normalize: bool = True) -> torch.Tensor:
        """Compute discounted returns for an episode"""
        returns = []
        G = 0.0
        
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for numerical stability
        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

    @abstractmethod
    def _collect_episode(self) -> Tuple[List[np.ndarray], List[int], List[float], float]:
        """Collect a full episode of experience"""
        pass

    def _episode_step(self, env, state):
        """Common episode step pattern"""
        action = self.act(state)
        step_result = env.step(action)
        if len(step_result) == 5:  # Gymnasium format
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        elif len(step_result) == 4:  # Legacy gym format  
            next_state, reward, done, _ = step_result
        else:  # Simple format
            next_state, reward, done = step_result
        return next_state, action, reward, done
    
    def _collect_episode_generic(self, env) -> Tuple[List[np.ndarray], List[int], List[float], float]:
        """Generic episode collection that can be overridden"""
        states, actions, rewards = [], [], []
        obs = env.reset()
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        total_reward = 0.0
        
        for step in range(self.max_episode_length):
            states.append(state)
            next_state, action, reward, done = self._episode_step(env, state)
            
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        return states, actions, rewards, total_reward
        
    def _get_eval_action(self, state) -> int:
        """Get greedy action for evaluation"""
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            if hasattr(self, 'policy_net'):
                probs = self.policy_net(state_tensor)
                return int(probs.argmax().item())
        return 0

    @abstractmethod
    def _collect_episode(self) -> Tuple[List[np.ndarray], List[int], List[float], float]:
        """Collect a full episode of experience"""
        pass

    @abstractmethod
    def _update_policy(self, states: List[np.ndarray], actions: List[int], 
                      returns: torch.Tensor) -> float:
        """Update policy using collected experience"""
        pass


class ActorCriticAgent(PolicyBasedAgent):
    """
    Base class for actor-critic agents (like A2C, PPO).
    Combines policy and value function learning.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.value_loss_coeff = self.train_cfg.get("value_loss_coeff", 0.5)
        self.entropy_coeff = self.train_cfg.get("entropy_coeff", 0.01)

    @abstractmethod
    def _compute_advantage(self, rewards: List[float], values: List[float], 
                          next_values: List[float], dones: List[bool]) -> torch.Tensor:
        """Compute advantage estimates"""
        pass