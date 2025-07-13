"""
Utility functions for TRPO implementation.

This module contains helper functions for logging, visualization,
and other common tasks in reinforcement learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import json
import os
from datetime import datetime


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize logger with specified directory."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
        
        self.metrics = {
            'epochs': [],
            'returns': [],
            'policy_losses': [],
            'value_losses': [],
            'kl_divergences': [],
            'step_sizes': []
        }
    
    def log(self, epoch: int, **kwargs):
        """Log metrics for current epoch."""
        self.metrics['epochs'].append(epoch)
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Returns
        if 'returns' in self.metrics and self.metrics['returns']:
            axes[0, 0].plot(self.metrics['epochs'], self.metrics['returns'])
            axes[0, 0].set_title('Average Returns')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].grid(True)
        
        # Policy Loss
        if 'policy_losses' in self.metrics and self.metrics['policy_losses']:
            axes[0, 1].plot(self.metrics['epochs'], self.metrics['policy_losses'])
            axes[0, 1].set_title('Policy Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Value Loss
        if 'value_losses' in self.metrics and self.metrics['value_losses']:
            axes[1, 0].plot(self.metrics['epochs'], self.metrics['value_losses'])
            axes[1, 0].set_title('Value Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # KL Divergence
        if 'kl_divergences' in self.metrics and self.metrics['kl_divergences']:
            axes[1, 1].plot(self.metrics['epochs'], self.metrics['kl_divergences'])
            axes[1, 1].axhline(y=0.01, color='r', linestyle='--', label='KL Limit')
            axes[1, 1].set_title('KL Divergence')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('KL Divergence')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute explained variance for value function evaluation.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Explained variance ratio
    """
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()
    
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / var_y if var_y > 0 else 0


def discount_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted cumulative sum of rewards.
    
    Args:
        rewards: Array of rewards
        gamma: Discount factor
        
    Returns:
        Discounted cumulative rewards
    """
    discounted = np.zeros_like(rewards)
    discounted[-1] = rewards[-1]
    
    for t in reversed(range(len(rewards) - 1)):
        discounted[t] = rewards[t] + gamma * discounted[t + 1]
    
    return discounted


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Advantage estimates
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def flat_grad(grads: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten gradients into a single vector.
    
    Args:
        grads: List of gradient tensors
        
    Returns:
        Flattened gradient vector
    """
    return torch.cat([grad.view(-1) for grad in grads])


def set_flat_params_to(model: torch.nn.Module, flat_params: torch.Tensor):
    """
    Set model parameters from flattened parameter vector.
    
    Args:
        model: PyTorch model
        flat_params: Flattened parameter vector
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """
    Get flattened parameters from model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Flattened parameter vector
    """
    return torch.cat([param.view(-1) for param in model.parameters()])


def conjugate_gradients(Avp_f, b: torch.Tensor, nsteps: int = 10, 
                       residual_tol: float = 1e-10) -> torch.Tensor:
    """
    Conjugate gradient algorithm for solving Ax = b.
    
    Args:
        Avp_f: Function that computes A*v for vector v
        b: Right-hand side vector
        nsteps: Maximum number of steps
        residual_tol: Tolerance for residual
        
    Returns:
        Solution vector x
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        
        if new_rdotr < residual_tol:
            break
            
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        
    return x


def line_search(f, x: torch.Tensor, fullstep: torch.Tensor, 
               expected_improve_rate: float, max_backtracks: int = 10,
               accept_ratio: float = 0.1) -> torch.Tensor:
    """
    Backtracking line search.
    
    Args:
        f: Function to optimize
        x: Current point
        fullstep: Full step direction
        expected_improve_rate: Expected improvement rate
        max_backtracks: Maximum number of backtracking steps
        accept_ratio: Acceptance ratio for improvement
        
    Returns:
        New point after line search
    """
    fval = f(x)
    
    for (_n_backtracks, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
            
    return x


def explained_variance_1d(ypred: np.ndarray, y: np.ndarray) -> float:
    """
    Compute explained variance for 1D arrays.
    
    Args:
        ypred: Predicted values
        y: True values
        
    Returns:
        Explained variance
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def safe_mean(arr: List[float]) -> float:
    """
    Compute mean safely, handling empty lists.
    
    Args:
        arr: List of values
        
    Returns:
        Mean value or 0 if empty
    """
    return np.mean(arr) if len(arr) > 0 else 0.0


class RunningMeanStd:
    """Running mean and standard deviation computation."""
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Initialize running statistics.
        
        Args:
            epsilon: Small constant for numerical stability
            shape: Shape of the statistics
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, 
                           batch_var: np.ndarray, batch_count: int):
        """Update from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

