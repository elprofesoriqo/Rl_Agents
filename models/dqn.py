import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Tuple
import os
from datetime import datetime

from models.network import ConvNetwork, NoisyLinear

class NatureDQN(ConvNetwork):
    """
    DQN Network Architecture from Nature 2015 paper:
    "Human-level control through deep reinforcement learning"
    
    Input: 84x84x4 (4 consecutive grayscale frames)
    Conv1: 32 filters, 8x8, stride 4, ReLU
    Conv2: 64 filters, 4x4, stride 2, ReLU  
    Conv3: 64 filters, 3x3, stride 1, ReLU
    FC1: 512 units, ReLU
    FC2: n_actions units, linear
    """
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super().__init__(input_shape)
        self.n_actions = n_actions
        
        # Create FC head
        self.fc_head = self._create_fc_head(512, n_actions)
        
        # Initialize FC layers
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_to_flat(x)
        
        # Forward through FC head
        return self.fc_head(x)

class DuelingDQN(ConvNetwork):
    """
    Dueling DQN Architecture from 2015 paper:
    "Dueling Network Architectures for Deep Reinforcement Learning"
    https://arxiv.org/abs/1511.06581
    
    Separates Q(s,a) into V(s) + A(s,a) where:
    - V(s): State value function
    - A(s,a): Action advantage function 
    - Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super().__init__(input_shape)
        self.n_actions = n_actions
        
        # Create dueling
        self.shared, self.value_head, self.advantage_head = self._create_dueling_heads(512, n_actions)
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_to_flat(x)
        
        shared_features = self.shared(x)
        
        # Value and advantage streams
        value = self.value_head(shared_features)
        advantage = self.advantage_head(shared_features)
        
        return self._dueling_aggregation(value, advantage)

class RainbowDQN(ConvNetwork):
    """
    Rainbow DQN combining key components from:
    "Rainbow: Combining Improvements in Deep Reinforcement Learning" (AAAI, 2018)
    https://arxiv.org/abs/1710.02298
    
    Components:
    - Dueling architecture
    - Noisy networks for exploration  
    - Distributional learning (Categorical DQN)
    """
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int, 
                 n_atoms: int = 51, v_min: float = -10, v_max: float = 10, noisy: bool = True):
        super().__init__(input_shape)
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.noisy = noisy
        
        # Create distributional dueling heads (noisy or regular)
        LayerType = NoisyLinear if noisy else nn.Linear
        self.fc_shared = LayerType(self.conv_out_size, 512)
        self.value_stream = LayerType(512, n_atoms)
        self.advantage_stream = LayerType(512, n_actions * n_atoms)
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_to_flat(x)
        shared_features = F.relu(self.fc_shared(x))
        
        # Value stream: V(s) distribution
        value = self.value_stream(shared_features).view(-1, 1, self.n_atoms)
        
        # Advantage stream: A(s,a) distribution  
        advantage = self.advantage_stream(shared_features).view(-1, self.n_actions, self.n_atoms)
        
        q_dist = self._dueling_aggregation(value, advantage)

        # Apply softmax to get valid probability distributions
        return F.softmax(q_dist, dim=-1)

    def reset_noise(self):
        """Reset noise parameters for noisy networks"""
        if self.noisy:
            self.fc_shared.reset_noise()
            self.value_stream.reset_noise()
            self.advantage_stream.reset_noise()

from models.model import ValueBasedAgent
from games.replay_buffer import ReplayBuffer  
from games.experiment_logger import ExperimentLogger
from games.atari_env import AtariGame

class DQNAgent(ValueBasedAgent):
    """
    Deep Q-Network Agent implementation supporting multiple variants:
    - Nature DQN: "Human-level control through deep reinforcement learning" (Nature, 2015)
      https://doi.org/10.1038/nature14236
    
    - Double DQN: "Deep Reinforcement Learning with Double Q-learning" (AAAI, 2016)  
      https://arxiv.org/abs/1509.06461
    
    - Dueling DQN: "Dueling Network Architectures for Deep Reinforcement Learning" (ICML, 2016)
      https://arxiv.org/abs/1511.06581
      
    - Rainbow DQN: "Rainbow: Combining Improvements in Deep Reinforcement Learning" (AAAI, 2018)
      https://arxiv.org/abs/1710.02298
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.game = AtariGame(
            self.env_cfg["id"], 
            self.env_cfg.get("kwargs", {}), 
            self.env_cfg.get("seed")
        )

        self.n_actions = self.game.action_space_n
        self.input_shape = tuple(self.model_cfg.get("input_shape", (4, 84, 84)))
        self._init_networks()
        self.optimizer = self._setup_optimizer(self.q_network)
        self.memory = ReplayBuffer(self.train_cfg.get("memory_size", 1_000_000))
        self.reward_clip = self.train_cfg.get("reward_clip", None)

        experiment_name = f"{self.network_type}_{self.env_cfg['id'].split('/')[-1].replace('-v5', '')}"
        self._setup_experiment_logging(experiment_name)

    def _init_networks(self):
        """Initialize Q-networks based on network type"""
        network_type = self.model_cfg.get("network_type", "nature")
        
        if network_type == "dueling":
            self.q_network = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
            self.target_network = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
            self.is_distributional = False
        elif network_type == "rainbow":
            n_atoms = self.model_cfg.get("n_atoms", 51)
            v_min = self.model_cfg.get("v_min", -10)
            v_max = self.model_cfg.get("v_max", 10)
            noisy = self.model_cfg.get("noisy", True)
            self.q_network = RainbowDQN(self.input_shape, self.n_actions, n_atoms, v_min, v_max, noisy).to(self.device)
            self.target_network = RainbowDQN(self.input_shape, self.n_actions, n_atoms, v_min, v_max, noisy).to(self.device)
            self.n_atoms = n_atoms
            self.v_min = v_min
            self.v_max = v_max
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
            self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
            self.is_distributional = True
        else:  # Default Nature DQN
            self.q_network = NatureDQN(self.input_shape, self.n_actions).to(self.device)
            self.target_network = NatureDQN(self.input_shape, self.n_actions).to(self.device)
            self.is_distributional = False
            
        self.network_type = network_type
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()



    def act(self, state) -> int:
        """Select action using current policy"""
        # Rainbow with noisy networks doesn't use epsilon-greedy
        if self.network_type == "rainbow" and hasattr(self.q_network, 'noisy') and self.q_network.noisy:
            with torch.no_grad():
                state_tensor = self._to_tensor(state).unsqueeze(0)
                q_dist = self.q_network(state_tensor)
                q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=2)
                return int(q_values.argmax(dim=1).item())
        else: # Epsilon-greedy for Nature DQN and Dueling DQN
            if np.random.rand() < self.epsilon():
                return np.random.randint(self.n_actions)
            with torch.no_grad():
                state_tensor = self._to_tensor(state).unsqueeze(0)
                if self.is_distributional:
                    q_dist = self.q_network(state_tensor)
                    q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=2)
                else:
                    q_values = self.q_network(state_tensor)
                return int(q_values.argmax(dim=1).item())

    def _optimize(self):
        """Perform one optimization step"""
        if len(self.memory) < self.batch_size or self.steps_done < self.warmup_steps:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Use base class tensor conversion methods
        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.long)
        rewards_t = self._to_tensor(rewards, dtype=torch.float)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones.astype(np.uint8), dtype=torch.bool)

        if self.is_distributional:
            loss = self._rainbow_loss(states_t, actions_t, rewards_t, next_states_t, dones_t)
        else: # Standard Double DQN loss
            current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            next_actions = self.q_network(next_states_t).argmax(1).detach()
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            target_q = rewards_t + self.gamma * next_q * (~dones_t)
            loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        self._apply_gradient_clipping(self.q_network)
        
        self.optimizer.step()

        # Reset noisy networks for Rainbow
        if self.network_type == "rainbow" and hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        return float(loss.item())

    def _rainbow_loss(self, states, actions, rewards, next_states, dones):
        """Categorical DQN loss for Rainbow"""
        batch_size = states.size(0)
        
        # Current Q distribution
        current_q_dist = self.q_network(states)
        current_q_dist = current_q_dist.gather(2, actions.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.n_atoms)).squeeze(1)
        
        # Next Q distribution (Double DQN action selection)
        next_q_dist = self.q_network(next_states)
        next_q_values = (next_q_dist * self.support.view(1, 1, -1)).sum(dim=2)
        next_actions = next_q_values.argmax(1)
        
        # Target Q distribution
        target_q_dist = self.target_network(next_states)
        target_q_dist = target_q_dist.gather(2, next_actions.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.n_atoms)).squeeze(1)
        
        # Categorical projection
        target_support = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
        target_support = target_support.clamp(self.v_min, self.v_max)
        
        # Compute categorical indices and weights
        b = (target_support - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix disappearing probability mass
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.n_atoms - 1)) * (l == u)] += 1
        
        # Project onto support
        m = torch.zeros_like(target_q_dist)
        offset = torch.linspace(0, ((batch_size - 1) * self.n_atoms), batch_size).long().unsqueeze(1).expand(batch_size, self.n_atoms).to(self.device)
        
        m.view(-1).index_add_(0, (l + offset).view(-1), (target_q_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (target_q_dist * (b - l.float())).view(-1))
        
        loss = -torch.sum(m * current_q_dist.log(), dim=1).mean()
        return loss


    def _log_episode_stats(self, episode: int, episode_steps: int, reward: float, 
                          avg_loss: float, best_reward: float, worst_reward: float,
                          rolling_avg: float, rolling_std: float):
        """Log episode statistics to experiment directory"""
        episode_data = {
            'total_steps': self.steps_done,
            'episode_steps': episode_steps,
            'reward': reward,
            'loss': avg_loss,
            'epsilon': self.epsilon(),
            'best_reward': best_reward,
            'worst_reward': worst_reward,
            'rolling_avg': rolling_avg,
            'rolling_std': rolling_std,
            'network_type': self.network_type,
            'batch_size': self.batch_size,
            'learning_rate': self.train_cfg.get('learning_rate', 0.00025),
            'memory_size': len(self.memory),
        }
        
        super()._log_episode_stats(episode, **episode_data)

    def train(self) -> None:
        """Main training loop"""
        num_episodes = self.train_cfg.get("num_episodes", 1000)
        save_every = self.train_cfg.get("save_every", 100)
        render_every = self.train_cfg.get("render_every", 0)
        stats_window = int(self.train_cfg.get("stats_window", 100))
        max_steps_per_episode = self.train_cfg.get("max_steps_per_episode", None)
        reward_log_every = self.train_cfg.get("reward_log_every", None)

        print(f"Training {self.network_type.upper()} DQN on {self.env_cfg['id']}")
        print(f"Device: {self.device} | Episodes: {num_episodes}")
        print(f"Log dir: {self.writer.log_dir}")
        print("=" * 60)

        recent_rewards = []
        best_reward = float("-inf")
        worst_reward = float("inf")

        for ep in range(num_episodes):
            obs, _ = self.game.reset()
            state = obs
            render_this = render_every and (ep % render_every == 0)
            
            if render_this:
                try:
                    self.game.reset_render()
                    print(f"[Episode {ep}] Rendering enabled")
                except Exception:
                    render_this = False

            total_reward = 0.0
            steps = 0
            episode_losses = []
            
            while True:
                action = self.act(state)
                next_obs, reward, terminated, truncated, _ = self.game.step(action)
                
                if render_this:
                    try:
                        self.game.step_render(action)
                    except Exception:
                        render_this = False
                        
                next_state = next_obs
                done = terminated or truncated
                
                if max_steps_per_episode is not None and steps + 1 >= max_steps_per_episode:
                    done = True
                
                # Apply reward clipping if specified
                clipped_reward = reward
                if self.reward_clip is not None:
                    clipped_reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                
                self.memory.push(state, action, clipped_reward, next_state, done)

                state = next_state
                total_reward += float(reward)
                steps += 1
                self.steps_done += 1

                loss_val = self._optimize()
                if loss_val is not None:
                    episode_losses.append(loss_val)

                # Per-step logging
                if reward_log_every and self.steps_done % reward_log_every == 0:
                    print(f"[ep {ep:4d} | t {steps:5d}] R {float(reward):8.3f} | cumR {total_reward:8.1f} | eps {self.epsilon():.3f} | act {action}")

                # Update target network
                self._update_target_network(self.q_network, self.target_network)

                if done:
                    break

            # Episode statistics
            self.episode += 1
            self.writer.add_scalar("episode/reward", total_reward, ep)
            self.writer.add_scalar("episode/steps", steps, ep)
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            self.writer.add_scalar("episode/avg_loss", avg_loss, ep)

            # Update rolling statistics
            recent_rewards.append(total_reward)
            if len(recent_rewards) > stats_window:
                recent_rewards.pop(0)
            best_reward = max(best_reward, total_reward)
            worst_reward = min(worst_reward, total_reward)
            rolling_avg = float(np.mean(recent_rewards))
            rolling_std = float(np.std(recent_rewards))

            self._log_episode_stats(ep, steps, total_reward, avg_loss, best_reward, worst_reward, rolling_avg, rolling_std)
            print(f"Ep {ep:4d} | R {total_reward:8.2f} | steps {steps:5d} | loss {avg_loss:7.4f} | eps {self.epsilon():.3f}")

            if save_every and ep > 0 and ep % save_every == 0:
                checkpoint_path = os.path.join(self.experiment_logger.checkpoints_dir, f"checkpoint_ep_{ep}.pth")
                self.save(checkpoint_path)
                checkpoint_metrics = {
                    'episode': ep,
                    'reward': total_reward,
                    'rolling_avg': rolling_avg,
                    'steps': self.steps_done,
                    'loss': avg_loss
                }
                self.experiment_logger.log_checkpoint(checkpoint_path, ep, checkpoint_metrics)
                        
        self.cleanup()

    def save(self, path: str) -> None:
        """Save model checkpoint"""
        networks = {
            "q_network": self.q_network,
            "target_network": self.target_network
        }
        optimizers = {
            "optimizer": self.optimizer
        }
        additional_data = {
            "steps_done": self.steps_done
        }
        
        self.save_agent(path, networks, optimizers, additional_data)

    def load(self, path: str) -> None:
        """Load model checkpoint"""
        network_names = ["q_network", "target_network"]
        optimizer_names = ["optimizer"]
        
        self.load_agent(path, network_names, optimizer_names)

    def _get_eval_action(self, state) -> int:
        """Get greedy action for evaluation"""
        with torch.no_grad():
            state_t = self._to_tensor(state).unsqueeze(0)
            if self.is_distributional:
                q_dist = self.q_network(state_t)
                q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=2)
            else:
                q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())
    
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Evaluate the agent"""
        networks_to_eval = [self.q_network]
        return self.evaluate_agent(num_episodes, networks_to_eval)