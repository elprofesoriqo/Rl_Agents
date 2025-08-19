import numpy as np
import gymnasium as gym
import cv2
from collections import deque


class AtariPreprocessor(gym.ObservationWrapper):
    """
    Preprocessing exactly as described in Nature DQN paper:
    - Convert to grayscale
    - Resize to 84x84  
    - Stack 4 consecutive frames
    - Handle frame skipping and max pooling
    """
    
    def __init__(self, env, frame_stack=4):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        
        # Update observation space to grayscale 84x84x4
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8
        )
        
    def observation(self, obs):
        """Preprocess observation"""
        # Convert to grayscale and resize to 84x84
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add to frame stack
        self.frames.append(resized)
        
        # Return stacked frames
        return np.array(list(self.frames), dtype=np.uint8)
    
    def reset(self, **kwargs):
        """Reset and initialize frame stack"""
        obs, info = self.env.reset(**kwargs)
        
        # Fill frame stack with first frame
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        for _ in range(self.frame_stack):
            self.frames.append(resized)
            
        return np.array(list(self.frames), dtype=np.uint8), info