import argparse
import os
from pathlib import Path

from configs.config import Config
from utils.logger import Logger
from models.ppo_agent import PPOAgent
from utils.environment import CarRacingEnvironment


def main():
    """Main training loop for PPO agent."""
    parser = argparse.ArgumentParser(description='PPO CarRacing Agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'demo'],
                        default='train', help='Run mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file for resuming training')
    
    args = parser.parse_args()
    
    config = Config(args.config)
    
    logger = Logger(config)
    logger.info("Starting PPO CarRacing Agent")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    
    env = CarRacingEnvironment(config)
    
    agent = PPOAgent(config, env.observation_space, env.action_space)
    
    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    
    if args.mode == 'train':
        agent.train(env, logger)

if __name__ == "__main__":
    main()
