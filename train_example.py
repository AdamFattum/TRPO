"""
Example training script for TRPO on various environments.

This script demonstrates how to use the TRPO implementation
on different environments with proper logging and visualization.
"""

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from trpo_agent import TRPOAgent, test_agent
import argparse
import os


def plot_training_progress(returns, save_path=None):
    """Plot training progress."""
    plt.figure(figsize=(10, 6))
    plt.plot(returns)
    plt.title('TRPO Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Average Return')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def train_trpo(env_name, num_epochs=500, steps_per_epoch=2000, save_model=True):
    """
    Train TRPO agent on specified environment.
    
    Args:
        env_name: Name of the environment
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        save_model: Whether to save the trained model
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Training TRPO on {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent with environment-specific hyperparameters
    if 'Pendulum' in env_name:
        agent = TRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            max_kl=0.01,
            gamma=0.99,
            lam=0.95
        )
    else:
        # Default hyperparameters for other environments
        agent = TRPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            max_kl=0.01,
            gamma=0.99,
            lam=0.97
        )
    
    # Training loop with logging
    returns = []
    
    for epoch in range(num_epochs):
        # Collect trajectories
        trajectories = agent.collect_trajectories(env, steps_per_epoch)
        
        # Compute advantages
        trajectories = agent.compute_advantages(trajectories)
        
        # Update policy using TRPO
        agent.update_policy(trajectories)
        
        # Update value function
        agent.update_value_function(trajectories)
        
        # Log progress
        avg_return = np.mean([traj['returns'].sum().item() for traj in trajectories])
        returns.append(avg_return)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}, Average Return: {avg_return:8.2f}")
        
        # Early stopping if solved
        if len(returns) >= 100 and np.mean(returns[-100:]) > env.spec.reward_threshold:
            print(f"Environment solved in {epoch} epochs!")
            break
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, env, num_episodes=10)
    
    # Plot training progress
    plot_training_progress(returns, f"{env_name}_training_progress.png")
    
    # Save model
    if save_model:
        model_path = f"trpo_{env_name.lower()}_model.pth"
        torch.save({
            'policy_state_dict': agent.policy_net.state_dict(),
            'value_state_dict': agent.value_net.state_dict(),
            'hyperparameters': {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'max_kl': agent.max_kl,
                'gamma': agent.gamma,
                'lam': agent.lam
            }
        }, model_path)
        print(f"Model saved to {model_path}")
    
    env.close()
    return agent, returns


def load_and_test_model(model_path, env_name, num_episodes=10):
    """Load a trained model and test it."""
    # Load model
    checkpoint = torch.load(model_path)
    hyperparams = checkpoint['hyperparameters']
    
    # Create environment and agent
    env = gym.make(env_name)
    agent = TRPOAgent(
        state_dim=hyperparams['state_dim'],
        action_dim=hyperparams['action_dim'],
        max_kl=hyperparams['max_kl'],
        gamma=hyperparams['gamma'],
        lam=hyperparams['lam']
    )
    
    # Load trained weights
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_state_dict'])
    
    # Test the agent
    print(f"Testing loaded model on {env_name}")
    test_agent(agent, env, num_episodes)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train TRPO agent')
    parser.add_argument('--env', type=str, default='Pendulum-v1',
                       help='Environment name (default: Pendulum-v1)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (default: 500)')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Steps per epoch (default: 2000)')
    parser.add_argument('--test-only', type=str, default=None,
                       help='Path to model file for testing only')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    
    args = parser.parse_args()
    
    if args.test_only:
        load_and_test_model(args.test_only, args.env)
    else:
        train_trpo(
            env_name=args.env,
            num_epochs=args.epochs,
            steps_per_epoch=args.steps,
            save_model=not args.no_save
        )


if __name__ == "__main__":
    main()

