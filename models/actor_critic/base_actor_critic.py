import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from abc import abstractmethod

from ..base_trainer import BaseTrainer


class ActorCriticModel(BaseTrainer):
    """
    Base class for actor-critic RL models.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        self.max_episode_length = self.train_cfg.get("max_episode_length", 1000)
        self.n_steps = self.train_cfg.get("n_steps", 5)
        self.value_coef = self.train_cfg.get("value_coef", 0.5)
        self.entropy_coef = self.train_cfg.get("entropy_coef", 0.01)
        

    def _compute_n_step_returns(self, rewards: List[float], next_value: float = 0.0) -> torch.Tensor:
        """Compute n-step returns for actor-critic methods"""
        returns = []
        R = next_value
        
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns).to(self.device)

    def _compute_advantages(self, rewards: List[float], values: torch.Tensor, 
                          next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        returns = self._compute_n_step_returns(rewards, next_value)
        advantages = returns - values.detach()
        return returns, advantages

    def _collect_trajectories(self, num_steps: int, render: bool = False, initial_state=None) -> Dict[str, List]:
        """Collect trajectories for n-step learning"""
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        if initial_state is not None:
            state = initial_state
        else:
            state, _ = self.env.reset()
        
        done = False
        step_count = 0
        
        while step_count < num_steps and not done:
            states.append(state)
            
            action, log_prob, value = self.act_and_value(state)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            if render:
                if hasattr(self, 'game'):
                    try:
                        self.game.step_render(action)
                    except Exception:
                        render = False
            
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            
            state = next_state
            step_count += 1
            
            if self.max_episode_length is not None and step_count >= self.max_episode_length:
                done = True
        
        next_value = 0.0
        if not done:
            with torch.no_grad():
                _, _, next_value = self.act_and_value(state)
                next_value = next_value.item()
        
        return {
            'states': states,
            'actions': actions, 
            'rewards': rewards,
            'values': torch.stack(values).squeeze(),
            'log_probs': torch.stack(log_probs).squeeze(),
            'next_value': next_value,
            'next_state': state,
            'done': done
        }

    def run_episode(self, episode: int, render_every: int) -> Tuple[float, int, List[float]]:
        """Run episode using n-step actor-critic learning"""
        render_this = render_every and (episode % render_every == 0)
        
        if render_this:
            if hasattr(self, 'game'):
                try:
                    self.game.reset_render()
                except Exception:
                    render_this = False
        
        total_reward = 0.0
        total_steps = 0
        episode_losses = []
        current_state = None 
        
        while True:
            trajectory = self._collect_trajectories(self.n_steps, render=render_this, initial_state=current_state)
            
            loss_dict = self._update_networks(trajectory)
            episode_losses.append(sum(loss_dict.values()))
            
            total_reward += sum(trajectory['rewards'])
            total_steps += len(trajectory['rewards'])
            
            if trajectory['done']:
                break
            
            current_state = trajectory['next_state']
                
        return total_reward, total_steps, episode_losses

    def act(self, state) -> int:
        """Select action using current policy"""
        with torch.no_grad():
            action, _, _ = self.act_and_value(state)
            return action

    @abstractmethod
    def act_and_value(self, state) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and compute value simultaneously"""
        pass

    @abstractmethod 
    def _update_networks(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """Update actor and critic networks using trajectory data"""
        pass

    def get_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add actor-critic specific metrics"""
        episode_data.update({
            'n_steps': self.n_steps,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
        })
        return episode_data

    def print_episode_progress(self, episode: int, reward: float, steps: int, loss: float, rolling_avg: float):
        """Actor-critic specific episode progress format"""
        print(f"Ep {episode:4d} | R {reward:8.2f} | len {steps:3d} | loss {loss:7.4f} | avg {rolling_avg:6.2f}")

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def _get_eval_action(self, state) -> int:
        """Get deterministic action for evaluation"""
        with torch.no_grad():
            action, _, _ = self.act_and_value(state)
            return action

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        pass