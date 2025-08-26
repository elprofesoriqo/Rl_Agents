import json
import os
from typing import Dict, Any


def create_config(algorithm: str, env_id: str, output_path: str) -> None:
    templates = {
        "dqn": {
            "agent_type": "dqn",
            "train": {
                "learning_rate": 2.5e-4,
                "use_rmsprop": True,
                "rmsprop_alpha": 0.95,
                "rmsprop_momentum": 0.95,
                "rmsprop_eps": 0.01,
                "gradient_clip": 10.0,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 1000000,
                "memory_size": 1000000,
                "batch_size": 32,
                "target_update": 10000,
                "num_episodes": 1000,
                "render_every": 0,
                "max_steps_per_episode": None,
                "save_every": 100,
                "stats_window": 100,
                "reward_clip": 1.0,
                "warmup_steps": 1000,
                "experiments_dir": "experiments"
            },
            "model": {
                "input_shape": [4, 84, 84]
            }
        },
        "ddqn": {
            "agent_type": "ddqn",
            "train": {
                "learning_rate": 6.25e-5,
                "use_rmsprop": True,
                "rmsprop_alpha": 0.95,
                "rmsprop_momentum": 0.95,
                "rmsprop_eps": 0.01,
                "gradient_clip": 10.0,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 1000000,
                "memory_size": 1000000,
                "batch_size": 32,
                "target_update": 8000,
                "num_episodes": 1000,
                "render_every": 0,
                "max_steps_per_episode": None,
                "save_every": 100,
                "stats_window": 100,
                "reward_clip": 1.0,
                "warmup_steps": 50000,
                "experiments_dir": "experiments"
            },
            "model": {
                "input_shape": [4, 84, 84]
            }
        },
        "rainbow": {
            "agent_type": "rainbow",
            "train": {
                "learning_rate": 1e-4,
                "use_rmsprop": True,
                "rmsprop_alpha": 0.95,
                "rmsprop_momentum": 0.95,
                "rmsprop_eps": 0.01,
                "gradient_clip": 10.0,
                "gamma": 0.99,
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "epsilon_decay": 1000000,
                "memory_size": 1000000,
                "batch_size": 32,
                "target_update": 8000,
                "num_episodes": 1000,
                "render_every": 0,
                "max_steps_per_episode": None,
                "save_every": 100,
                "stats_window": 100,
                "reward_clip": 1.0,
                "warmup_steps": 50000,
                "experiments_dir": "experiments"
            },
            "model": {
                "input_shape": [4, 84, 84],
                "n_atoms": 51,
                "v_min": -10,
                "v_max": 10,
                "noisy": True
            }
        },
        "a2c": {
            "agent_type": "a2c",
            "train": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "num_episodes": 1000,
                "max_episode_length": 500,
                "value_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "gradient_clip": 10.0,
                "render_every": 0,
                "save_every": 100,
                "stats_window": 100,
                "experiments_dir": "experiments"
            },
            "model": {
                "hidden_dim": 128,
                "shared_network": True
            }
        },
        "a3c": {
            "agent_type": "a3c",
            "train": {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "num_episodes": 1000,
                "max_episode_length": 500,
                "value_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "gradient_clip": 40.0,
                "num_workers": 8,
                "render_every": 0,
                "save_every": 100,
                "stats_window": 100,
                "experiments_dir": "experiments"
            },
            "model": {
                "hidden_dim": 256,
                "shared_network": True
            }
        },
        "adversarial_a2c": {
            "agent_type": "adversarial_a2c",
            "train": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "num_episodes": 1000,
                "max_episode_length": 500,
                "value_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "adversarial_coeff": 0.1,
                "gradient_clip": 10.0,
                "render_every": 0,
                "save_every": 100,
                "stats_window": 100,
                "experiments_dir": "experiments"
            },
            "model": {
                "hidden_dim": 128,
                "shared_network": True
            }
        }
    }
    
    config = templates[algorithm].copy()
    config["env"] = {
        "id": env_id,
        "kwargs": {},
        "seed": 42
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    import sys

    algorithm, env_id, output_path = sys.argv[1:4]
    
    available_algorithms = ["dqn", "ddqn", "rainbow", "a2c", "a3c", "adversarial_a2c"]
    if algorithm not in available_algorithms:
        print(f"Error: Unknown algorithm '{algorithm}'. Available: {', '.join(available_algorithms)}")
        sys.exit(1)
    
    create_config(algorithm, env_id, output_path)
    print(f"Created config: {output_path}")