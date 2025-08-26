import numpy as np
import torch
from typing import Any, Dict, List, Tuple
from abc import abstractmethod

from ..base_trainer import BaseTrainer


class PolicyBasedModel(BaseTrainer):
    """
    Base class for policy-based RL models.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        self.max_episode_length = self.train_cfg.get("max_episode_length", 1000)
        self.use_baseline = self.train_cfg.get("use_baseline", True)

    def _compute_returns(self, rewards: List[float], normalize: bool = True) -> torch.Tensor:
        """Compute discounted returns for an episode"""
        returns = []
        G = 0.0
        
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

    def _collect_episode(self, render: bool = False) -> Tuple[List[np.ndarray], List[int], List[float], float]:
        """Collect a full episode of experience"""
        states, actions, rewards = [], [], []
        obs = self.env.reset()
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        total_reward = 0.0
        
        for step in range(self.max_episode_length):
            states.append(state)
            
            action = self.act(state)
            step_result = self.env.step(action)
            
            if len(step_result) == 5: 
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_state, reward, done, _ = step_result
            else:
                next_state, reward, done = step_result
            
            if render:
                self.env.render()
            
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        return states, actions, rewards, total_reward

    def run_episode(self, episode: int, render_every: int) -> Tuple[float, int, List[float]]:
        """Run single episode"""
        render_this = render_every and (episode % render_every == 0)
        states, actions, rewards, total_reward = self._collect_episode(render=render_this)
        
        returns = self._compute_returns(rewards)
        policy_loss = self._update_policy(states, actions, returns)
        
        return total_reward, len(rewards), [policy_loss]

    @abstractmethod
    def _update_policy(self, states: List[np.ndarray], actions: List[int], returns: torch.Tensor) -> float:
        """Update policy using collected experience"""
        pass