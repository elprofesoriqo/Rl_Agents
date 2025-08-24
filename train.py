import os
import sys
import json
from models.value_based import DQNModel, DoubleDQNModel, RainbowModel
from models.policy_based import PolicyGradientAgent

def create_agent(config: dict):
    """Create agent based on config."""
    agent_type = config["agent_type"]
    
    if agent_type == "dqn":
        return DQNModel(config)
    elif agent_type == "ddqn" or agent_type == "double_dqn":
        return DoubleDQNModel(config)
    elif agent_type == "rainbow":
        return RainbowModel(config)
    elif agent_type == "policy_gradient" or agent_type == "reinforce":
        return PolicyGradientAgent(config)
    else:
        available_types = ["dqn", "ddqn", "rainbow", "policy_gradient"]
        print(f"Error: Unknown agent type '{agent_type}'")
        print(f"Available types: {available_types}")
        sys.exit(1)

def main():
    config_file = sys.argv[1]
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    experiments_dir = config["train"].get("experiments_dir", "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    print(f"Loading config: {config_file}")
    print(f"Agent type: {config['agent_type']}")
    print(f"Environment: {config['env']['id']}")
    
    agent = create_agent(config)
    agent.train()

if __name__ == "__main__":
    main()