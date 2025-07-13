# Trust Region Policy Optimization (TRPO) Implementation

A clean, educational implementation of Trust Region Policy Optimization (TRPO) as described in the paper "Trust Region Policy Optimization" by Schulman et al. (2015).

## Overview

TRPO is a policy gradient method that constrains policy updates to ensure monotonic improvement. It uses KL divergence to measure the distance between policies and employs conjugate gradient optimization to efficiently solve the constrained optimization problem.

## Key Features

- **Monotonic Improvement**: Theoretical guarantees for policy improvement
- **Stable Training**: Trust region constraint prevents destructive updates
- **Efficient Implementation**: Uses conjugate gradient method for scalability
- **Comprehensive**: Includes GAE for advantage estimation and value function learning

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trpo-implementation.git
cd trpo-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
import gym
from trpo_agent import TRPOAgent

# Create environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Create and train agent
agent = TRPOAgent(state_dim, action_dim)
agent.train(env, num_epochs=500, steps_per_epoch=2000)
```

### Custom Configuration

```python
agent = TRPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=64,
    max_kl=0.01,          # Trust region size
    gamma=0.99,           # Discount factor
    lam=0.95,             # GAE lambda
    cg_iters=10,          # Conjugate gradient iterations
    backtrack_iters=10    # Line search iterations
)
```

## Algorithm Components

### 1. Policy Network
- Neural network outputting Gaussian distribution parameters
- Separate mean and log standard deviation outputs
- Supports continuous action spaces

### 2. Value Network
- Estimates state values for advantage computation
- Used in Generalized Advantage Estimation (GAE)
- Trained via supervised learning on returns

### 3. Trust Region Optimization
- Constrains policy updates using KL divergence
- Conjugate gradient method for efficient optimization
- Backtracking line search for constraint satisfaction

### 4. Advantage Estimation
- Generalized Advantage Estimation (GAE)
- Balances bias and variance in advantage estimates
- Configurable λ parameter for trade-off control

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_kl` | 0.01 | Maximum KL divergence for trust region |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda parameter |
| `cg_iters` | 10 | Conjugate gradient iterations |
| `backtrack_iters` | 10 | Maximum line search steps |
| `damping` | 0.1 | Damping coefficient for numerical stability |

## Supported Environments

- **Continuous Control**: MuJoCo environments (Pendulum, HalfCheetah, etc.)
- **Discrete Actions**: Atari games (with categorical policy)
- **Custom Environments**: Any OpenAI Gym compatible environment

## Results

The implementation reproduces the results from the original paper:

- **Continuous Control**: Achieves state-of-the-art performance on MuJoCo tasks
- **Stability**: Consistent improvement without catastrophic failures
- **Sample Efficiency**: Better than vanilla policy gradients and derivative-free methods

## File Structure

```
trpo-implementation/
├── trpo_agent.py          # Main TRPO implementation
├── examples/              # Example scripts
│   ├── train_pendulum.py  # Pendulum training example
│   ├── train_mujoco.py    # MuJoCo training example
│   └── visualize.py       # Visualization utilities
├── utils/                 # Utility functions
│   ├── logger.py          # Training logger
│   └── plotting.py        # Plotting utilities
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Theory

TRPO solves the constrained optimization problem:

```
maximize L^π_old(π)
subject to D_KL(π_old, π) ≤ δ
```

Where:
- `L^π_old(π)` is the surrogate objective function
- `D_KL(π_old, π)` is the KL divergence between policies
- `δ` is the trust region size

The algorithm provides a monotonic improvement guarantee:

```
η(π_new) ≥ L^π_old(π_new) - C·α²
```

Where `α` is the total variation distance and `C` is a problem-dependent constant.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{schulman2015trust,
  title={Trust region policy optimization},
  author={Schulman, John and Levine, Sergey and Moritz, Philipp and Jordan, Michael I and Abbeel, Pieter},
  booktitle={International conference on machine learning},
  pages={1889--1897},
  year={2015},
  organization={PMLR}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

1. Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). Trust region policy optimization.
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation.
3. Kakade, S., & Langford, J. (2002). Approximately optimal approximate reinforcement learning.

