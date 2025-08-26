import numpy as np
import torch
from typing import Any, Dict, Tuple, List
from abc import abstractmethod

from ..base_trainer import BaseTrainer
from games.replay_buffer import ReplayBuffer  
from games.atari_env import AtariGame

class ValueBasedModel(BaseTrainer):
    """
    Base class for all value-based RL models (DQN variants).
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
        
        self.game = AtariGame(
            self.env_cfg["id"], 
            self.env_cfg.get("kwargs", {}), 
            self.env_cfg.get("seed")
        )
        self.n_actions = self.game.action_space_n
        self.input_shape = tuple(self.model_cfg.get("input_shape", (4, 84, 84)))
        self.memory = ReplayBuffer(self.train_cfg.get("memory_size", 1_000_000))
        self.reward_clip = self.train_cfg.get("reward_clip", None)
        
        self.create_networks()
        self.optimizer = self._setup_optimizer(self.q_network)
        
        experiment_name = f"{self.get_algorithm_name()}_{self.env_cfg['id'].split('/')[-1].replace('-v5', '')}"
        self._setup_experiment_logging(experiment_name)

    def epsilon(self) -> float:
        """Current epsilon value for epsilon-greedy exploration"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def _update_target_network(self, main_network: torch.nn.Module, 
                              target_network: torch.nn.Module) -> None:
        """Update target network with main network weights"""
        if self.steps_done % self.target_update == 0:
            target_network.load_state_dict(main_network.state_dict())

    @abstractmethod
    def create_networks(self):
        """Create main and target Q-networks"""
        pass

    @abstractmethod
    def compute_loss(self, batch):
        """Compute loss for the specific DQN variant"""
        pass

    def act(self, state) -> int:
        """Select action using epsilon-greedy (or variant-specific method)"""
        if hasattr(self, '_custom_act'):
            return self._custom_act(state)
        
        # Standard epsilon-greedy
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            q_values = self._get_q_values(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def _get_q_values(self, state_tensor):
        """Get Q-values from network (handles distributional vs standard)"""
        if hasattr(self, 'is_distributional') and self.is_distributional:
            q_dist = self.q_network(state_tensor)
            return (q_dist * self.support.view(1, 1, -1)).sum(dim=2)
        return self.q_network(state_tensor)

    def run_episode(self, episode: int, render_every: int) -> Tuple[float, int, List[float]]:
        """Run single episode for DQN variants"""
        obs, _ = self.game.reset()
        state = obs
        render_this = render_every and (episode % render_every == 0)
        
        if render_this:
            try:
                self.game.reset_render()
            except Exception:
                render_this = False

        total_reward = 0.0
        steps = 0
        episode_losses = []
        max_steps_per_episode = self.train_cfg.get("max_steps_per_episode", None)
        
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
            
            self._update_target_network(self.q_network, self.target_network)

            if done:
                break
                
        return total_reward, steps, episode_losses

    def _optimize(self):
        """Perform one optimization step"""
        if len(self.memory) < self.batch_size or self.steps_done < self.warmup_steps:
            return None
        
        batch = self.memory.sample(self.batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self._apply_gradient_clipping(self.q_network)
        self.optimizer.step()

        if hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        return float(loss.item())

    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add DQN-specific metrics"""
        episode_data.update({
            'total_steps': self.steps_done,
            'batch_size': self.batch_size,
            'learning_rate': self.train_cfg.get('learning_rate', 0.00025),
            'memory_size': len(self.memory),
            'epsilon': self.epsilon(),
        })
        return episode_data

    def print_episode_progress(self, episode: int, reward: float, steps: int, loss: float, rolling_avg: float):
        """DQN-specific episode progress format"""
        eps_str = f" | eps {self.epsilon():.3f}"
        print(f"Ep {episode:4d} | R {reward:8.2f} | steps {steps:5d} | loss {loss:7.4f}{eps_str}")

    def save(self, path: str) -> None:
        networks = {"q_network": self.q_network, "target_network": self.target_network}
        optimizers = {"optimizer": self.optimizer}
        additional_data = {"steps_done": self.steps_done}
        self.save_agent(path, networks, optimizers, additional_data)

    def load(self, path: str) -> None:
        network_names = ["q_network", "target_network"]
        optimizer_names = ["optimizer"]
        self.load_agent(path, network_names, optimizer_names)

    def _get_eval_action(self, state) -> int:
        """Get greedy action for evaluation"""
        with torch.no_grad():
            state_t = self._to_tensor(state).unsqueeze(0)
            q_values = self._get_q_values(state_t)
            return int(q_values.argmax(dim=1).item())
    
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        networks_to_eval = [self.q_network]
        return self.evaluate_agent(num_episodes, networks_to_eval)