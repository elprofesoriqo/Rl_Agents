import os
import sys
import json
from models.dqn import DQNAgent
from models.policy_gradient import PolicyGradientAgent


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_file):
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, Exception):
        sys.exit(1)
    
    # Validate required sections
    required_sections = ['env', 'train', 'model']
    for section in required_sections:
        if section not in config:
            sys.exit(1)
    
    # Validate required environment parameters
    if 'id' not in config['env']:
        sys.exit(1)
    
    # Validate required agent type
    if 'agent_type' not in config:
        sys.exit(1)
    
    return config




def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration
    config = load_config(config_file)
    
    # Ensure experiments directory exists
    experiments_dir = config["train"].get("experiments_dir", "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    
    # Initialize agent based on type
    agent_type = config["agent_type"]
    if agent_type == "dqn":
        agent = DQNAgent(config)
    elif agent_type == "policy_gradient":
        agent = PolicyGradientAgent(config)
    else:
        print(f"Error: Unknown agent type '{agent_type}'")
        sys.exit(1)
    
    # Start training
    agent.train()


if __name__ == "__main__":
    main()