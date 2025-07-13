
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import gym
from typing import List, Dict, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs parameters of a Gaussian distribution for continuous actions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        
        # Log standard deviation as a parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, states: torch.Tensor) -> Normal:
        """Forward pass returning a Normal distribution."""
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions under current policy."""
        dist = self.forward(states)
        return dist.log_prob(actions).sum(dim=-1)
    
    def get_kl_divergence(self, states: torch.Tensor, old_policy: 'PolicyNetwork') -> torch.Tensor:
        """Compute KL divergence between current and old policy."""
        current_dist = self.forward(states)
        old_dist = old_policy.forward(states)
        return torch.distributions.kl_divergence(old_dist, current_dist).mean()


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state values."""
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        return self.value_head(x).squeeze(-1)


class TRPOAgent:
    """
    Trust Region Policy Optimization Agent.
    
    Implements the TRPO algorithm with conjugate gradient optimization
    and line search for constrained policy updates.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        max_kl: float = 0.01,
        damping: float = 0.1,
        cg_iters: int = 10,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        gamma: float = 0.99,
        lam: float = 0.95
    ):
        """
        Initialize TRPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension for networks
            lr: Learning rate for value function
            max_kl: Maximum KL divergence for trust region
            damping: Damping coefficient for conjugate gradient
            cg_iters: Number of conjugate gradient iterations
            backtrack_iters: Maximum backtracking iterations
            backtrack_coeff: Backtracking coefficient
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.gamma = gamma
        self.lam = lam
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
    def collect_trajectories(self, env: gym.Env, total_steps: int) -> List[Dict]:
        """
        Collect trajectories using current policy.
        
        Args:
            env: Environment to collect trajectories from
            total_steps: Total number of steps to collect
            
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        steps_collected = 0
        
        while steps_collected < total_steps:
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': []
            }
            
            state = env.reset()
            done = False
            
            while not done and steps_collected < total_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action from policy
                with torch.no_grad():
                    dist = self.policy_net(state_tensor)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()
                    value = self.value_net(state_tensor)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action.numpy().flatten())
                
                # Store transition
                trajectory['states'].append(state)
                trajectory['actions'].append(action.numpy().flatten())
                trajectory['rewards'].append(reward)
                trajectory['log_probs'].append(log_prob.item())
                trajectory['values'].append(value.item())
                
                state = next_state
                steps_collected += 1
            
            # Convert lists to tensors
            for key in trajectory:
                if key in ['states', 'actions', 'rewards']:
                    trajectory[key] = torch.FloatTensor(trajectory[key])
                else:
                    trajectory[key] = torch.FloatTensor(trajectory[key])
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def compute_advantages(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Trajectories with computed advantages and returns
        """
        for trajectory in trajectories:
            rewards = trajectory['rewards']
            values = trajectory['values']
            
            # Compute returns and advantages using GAE
            advantages = []
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0  # Terminal state
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.lam * gae
                advantages.insert(0, gae)
            
            trajectory['advantages'] = torch.FloatTensor(advantages)
            
            # Compute returns for value function training
            returns = []
            discounted_return = 0
            for reward in reversed(rewards):
                discounted_return = reward + self.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            trajectory['returns'] = torch.FloatTensor(returns)
        
        return trajectories
    
    def conjugate_gradient(self, Ax_function, b: torch.Tensor, max_iters: int = 10, tol: float = 1e-8) -> torch.Tensor:
        """
        Solve Ax = b using conjugate gradient method.
        
        Args:
            Ax_function: Function that computes Ax for given x
            b: Right-hand side vector
            max_iters: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Solution vector x
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)
        
        for i in range(max_iters):
            Ap = Ax_function(p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            
            if torch.sqrt(rsnew) < tol:
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    def hessian_vector_product(self, vector: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product for KL divergence.
        
        Args:
            vector: Vector to multiply with Hessian
            states: States for computing KL divergence
            
        Returns:
            Hessian-vector product
        """
        # Create a copy of the policy for computing KL divergence
        old_policy = PolicyNetwork(self.state_dim, self.action_dim)
        old_policy.load_state_dict(self.policy_net.state_dict())
        
        # Compute KL divergence
        kl = self.policy_net.get_kl_divergence(states, old_policy)
        
        # Compute gradient of KL divergence
        grad = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad])
        
        # Compute gradient-vector product
        grad_vector_product = torch.dot(flat_grad, vector)
        
        # Compute Hessian-vector product
        hvp = torch.autograd.grad(grad_vector_product, self.policy_net.parameters())
        flat_hvp = torch.cat([g.view(-1) for g in hvp])
        
        return flat_hvp + self.damping * vector
    
    def get_flat_params(self) -> torch.Tensor:
        """Get flattened parameters from policy network."""
        return torch.cat([param.view(-1) for param in self.policy_net.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Set policy network parameters from flattened vector."""
        idx = 0
        for param in self.policy_net.parameters():
            param_length = param.numel()
            param.data = flat_params[idx:idx + param_length].view(param.shape)
            idx += param_length
    
    def line_search(
        self,
        old_params: torch.Tensor,
        search_direction: torch.Tensor,
        step_size: float,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> None:
        """
        Perform backtracking line search.
        
        Args:
            old_params: Original policy parameters
            search_direction: Search direction from conjugate gradient
            step_size: Initial step size
            states: Batch of states
            actions: Batch of actions
            advantages: Batch of advantages
            old_log_probs: Log probabilities under old policy
        """
        for i in range(self.backtrack_iters):
            new_params = old_params + (self.backtrack_coeff ** i) * step_size * search_direction
            self.set_flat_params(new_params)
            
            # Check KL constraint
            old_policy = PolicyNetwork(self.state_dim, self.action_dim)
            old_policy.load_state_dict(self.policy_net.state_dict())
            self.set_flat_params(old_params)
            
            kl = self.policy_net.get_kl_divergence(states, old_policy)
            
            if kl <= self.max_kl:
                # Check surrogate improvement
                current_log_probs = self.policy_net.get_log_prob(states, actions)
                ratio = torch.exp(current_log_probs - old_log_probs)
                surrogate_improvement = (ratio * advantages).mean()
                
                if surrogate_improvement > 0:
                    return  # Accept the update
            
            # Restore old parameters for next iteration
            self.set_flat_params(old_params)
        
        # If no improvement found, keep old parameters
        print("Line search failed, keeping old parameters")
    
    def update_policy(self, trajectories: List[Dict]) -> None:
        """
        Update policy using TRPO.
        
        Args:
            trajectories: List of trajectory dictionaries with advantages
        """
        # Flatten all trajectory data
        all_states = torch.cat([traj['states'] for traj in trajectories])
        all_actions = torch.cat([traj['actions'] for traj in trajectories])
        all_advantages = torch.cat([traj['advantages'] for traj in trajectories])
        all_old_log_probs = torch.cat([traj['log_probs'] for traj in trajectories])
        
        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # Compute surrogate objective gradient
        current_log_probs = self.policy_net.get_log_prob(all_states, all_actions)
        ratio = torch.exp(current_log_probs - all_old_log_probs)
        surrogate_obj = (ratio * all_advantages).mean()
        
        # Compute policy gradient
        policy_grad = torch.autograd.grad(surrogate_obj, self.policy_net.parameters())
        flat_policy_grad = torch.cat([g.view(-1) for g in policy_grad])
        
        # Solve for natural gradient using conjugate gradient
        def hvp_function(v):
            return self.hessian_vector_product(v, all_states)
        
        natural_grad = self.conjugate_gradient(hvp_function, flat_policy_grad, self.cg_iters)
        
        # Compute step size
        quad_form = torch.dot(natural_grad, hvp_function(natural_grad))
        step_size = torch.sqrt(2 * self.max_kl / quad_form)
        
        # Perform line search
        old_params = self.get_flat_params()
        self.line_search(old_params, natural_grad, step_size, all_states, 
                        all_actions, all_advantages, all_old_log_probs)
    
    def update_value_function(self, trajectories: List[Dict], epochs: int = 5) -> None:
        """
        Update value function using supervised learning.
        
        Args:
            trajectories: List of trajectory dictionaries with returns
            epochs: Number of training epochs
        """
        all_states = torch.cat([traj['states'] for traj in trajectories])
        all_returns = torch.cat([traj['returns'] for traj in trajectories])
        
        for _ in range(epochs):
            predicted_values = self.value_net(all_states)
            value_loss = nn.MSELoss()(predicted_values, all_returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def train(self, env: gym.Env, num_epochs: int = 1000, steps_per_epoch: int = 4000) -> None:
        """
        Train the TRPO agent.
        
        Args:
            env: Environment to train on
            num_epochs: Number of training epochs
            steps_per_epoch: Number of environment steps per epoch
        """
        for epoch in range(num_epochs):
            # Collect trajectories
            trajectories = self.collect_trajectories(env, steps_per_epoch)
            
            # Compute advantages
            trajectories = self.compute_advantages(trajectories)
            
            # Update policy using TRPO
            self.update_policy(trajectories)
            
            # Update value function
            self.update_value_function(trajectories)
            
            # Log progress
            if epoch % 10 == 0:
                avg_return = np.mean([traj['returns'].sum().item() for traj in trajectories])
                print(f"Epoch {epoch}, Average Return: {avg_return:.2f}")


def test_agent(agent: TRPOAgent, env: gym.Env, num_episodes: int = 10) -> None:
    """
    Test the trained agent.
    
    Args:
        agent: Trained TRPO agent
        env: Environment to test on
        num_episodes: Number of test episodes
    """
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist = agent.policy_net(state_tensor)
                action = dist.mean  # Use mean action for testing
            
            state, reward, done, _ = env.step(action.numpy().flatten())
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    print(f"Average test reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")


if __name__ == "__main__":
    # Example usage
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = TRPOAgent(state_dim, action_dim)
    
    # Train the agent
    agent.train(env, num_epochs=500, steps_per_epoch=2000)
    
    # Test the trained agent
    test_agent(agent, env)

