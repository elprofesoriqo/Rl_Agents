import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=UserWarning)

from configs.config import Config
from models.trainer import create_trainer
from utils.logger import Logger


def print_system_info() -> None:
    print(f'  CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA Device: {torch.cuda.get_device_name()}')
        print(f'  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print()


def main() -> None:
    """Main entry point for PPO CarRacing Agent."""
    parser = argparse.ArgumentParser(description='PPO CarRacing Agent')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file for resuming')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Configuration file path')

    args = parser.parse_args()
    print_system_info()
    config = Config(args.config)
    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f'Error: Checkpoint file not found: {args.checkpoint}')
        sys.exit(1)

    with Logger(config) as logger:
        try:
            trainer = create_trainer(config, logger)
            if args.checkpoint:
                trainer.load_checkpoint(args.checkpoint)
                logger.info(f'Resumed from checkpoint: {args.checkpoint}')
            trainer.train()
        except Exception as e:
            logger.error(f'❌ Error: {str(e)}')
            sys.exit(1)


if __name__ == '__main__':
    main()
