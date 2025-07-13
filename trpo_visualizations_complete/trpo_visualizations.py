"""
TRPO Algorithm Visualizations

This script creates comprehensive visualizations to demonstrate how TRPO operates,
using simulated data to show the key concepts and behaviors of the algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Ellipse
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_simulated_training_data(num_epochs=100):
    """Generate realistic simulated training data for TRPO."""
    np.random.seed(42)
    
    # Simulate training progress with realistic patterns
    epochs = np.arange(num_epochs)
    
    # Returns: Start low, improve with some noise, plateau
    base_return = -1000 + 800 * (1 - np.exp(-epochs / 30))
    returns = base_return + 50 * np.random.normal(0, 1, num_epochs)
    
    # KL divergence: Should stay below threshold with occasional spikes
    kl_base = 0.005 + 0.003 * np.exp(-epochs / 20)
    kl_divergences = np.abs(kl_base + 0.002 * np.random.normal(0, 1, num_epochs))
    kl_divergences = np.clip(kl_divergences, 0.001, 0.015)
    
    # Step sizes: Adaptive based on KL constraint
    step_sizes = 0.01 / (1 + kl_divergences * 100)
    step_sizes += 0.001 * np.random.normal(0, 1, num_epochs)
    step_sizes = np.clip(step_sizes, 0.001, 0.02)
    
    # Surrogate improvements: Generally positive, decreasing over time
    surrogate_improvements = 0.1 * np.exp(-epochs / 40) + 0.02 * np.random.normal(0, 1, num_epochs)
    
    # Value losses: Decreasing over time
    value_losses = 100 * np.exp(-epochs / 25) + 5 * np.random.normal(0, 1, num_epochs)
    value_losses = np.clip(value_losses, 1, 200)
    
    # Policy entropy: Decreasing as policy becomes more deterministic
    policy_entropies = 2.0 * np.exp(-epochs / 50) + 0.1 * np.random.normal(0, 1, num_epochs)
    policy_entropies = np.clip(policy_entropies, 0.1, 2.5)
    
    # Advantage statistics
    advantage_means = 0.1 * np.sin(epochs / 10) + 0.05 * np.random.normal(0, 1, num_epochs)
    advantage_stds = 1.0 + 0.5 * np.exp(-epochs / 30) + 0.1 * np.random.normal(0, 1, num_epochs)
    
    # Line search steps: Usually 1-3, occasionally higher
    line_search_steps = np.random.choice([1, 2, 3, 4, 5], num_epochs, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    
    # CG iterations: Usually converge quickly
    cg_iterations = np.random.choice([3, 4, 5, 6, 7, 8, 9, 10], num_epochs, 
                                   p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02])
    
    # Explained variance: Improving over time
    explained_variances = 0.8 * (1 - np.exp(-epochs / 20)) + 0.1 * np.random.normal(0, 1, num_epochs)
    explained_variances = np.clip(explained_variances, 0.1, 0.95)
    
    return {
        'epochs': epochs,
        'returns': returns,
        'kl_divergences': kl_divergences,
        'step_sizes': step_sizes,
        'surrogate_improvements': surrogate_improvements,
        'value_losses': value_losses,
        'policy_entropies': policy_entropies,
        'advantage_means': advantage_means,
        'advantage_stds': advantage_stds,
        'line_search_steps': line_search_steps,
        'cg_iterations': cg_iterations,
        'explained_variances': explained_variances
    }

def create_comprehensive_visualizations():
    """Create comprehensive visualizations of TRPO training."""
    metrics = generate_simulated_training_data()
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training Progress
    ax1 = plt.subplot(3, 4, 1)
    plt.plot(metrics['epochs'], metrics['returns'], linewidth=2, color='blue')
    plt.title('Training Progress: Average Returns', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Return')
    plt.grid(True, alpha=0.3)
    
    # 2. KL Divergence Tracking
    ax2 = plt.subplot(3, 4, 2)
    plt.plot(metrics['epochs'], metrics['kl_divergences'], linewidth=2, color='red')
    plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='KL Limit (δ=0.01)')
    plt.title('Trust Region Constraint: KL Divergence', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 3. Step Size Adaptation
    ax3 = plt.subplot(3, 4, 3)
    plt.plot(metrics['epochs'], metrics['step_sizes'], linewidth=2, color='green')
    plt.title('Adaptive Step Sizes', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Step Size')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 4. Surrogate Objective Improvement
    ax4 = plt.subplot(3, 4, 4)
    plt.plot(metrics['epochs'], metrics['surrogate_improvements'], linewidth=2, color='purple')
    plt.title('Surrogate Objective Improvement', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Surrogate Improvement')
    plt.grid(True, alpha=0.3)
    
    # 5. Value Function Learning
    ax5 = plt.subplot(3, 4, 5)
    plt.plot(metrics['epochs'], metrics['value_losses'], linewidth=2, color='orange')
    plt.title('Value Function Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 6. Policy Entropy
    ax6 = plt.subplot(3, 4, 6)
    plt.plot(metrics['epochs'], metrics['policy_entropies'], linewidth=2, color='brown')
    plt.title('Policy Entropy (Exploration)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)
    
    # 7. Advantage Statistics
    ax7 = plt.subplot(3, 4, 7)
    plt.plot(metrics['epochs'], metrics['advantage_means'], linewidth=2, color='cyan', label='Mean')
    plt.plot(metrics['epochs'], metrics['advantage_stds'], linewidth=2, color='magenta', label='Std')
    plt.title('Advantage Function Statistics', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Advantage Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Line Search Performance
    ax8 = plt.subplot(3, 4, 8)
    plt.plot(metrics['epochs'], metrics['line_search_steps'], linewidth=2, color='red', marker='o', markersize=3)
    plt.title('Line Search Steps', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Steps')
    plt.grid(True, alpha=0.3)
    
    # 9. Conjugate Gradient Iterations
    ax9 = plt.subplot(3, 4, 9)
    plt.plot(metrics['epochs'], metrics['cg_iterations'], linewidth=2, color='blue', marker='s', markersize=3)
    plt.title('Conjugate Gradient Iterations', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('CG Iterations')
    plt.grid(True, alpha=0.3)
    
    # 10. Explained Variance
    ax10 = plt.subplot(3, 4, 10)
    plt.plot(metrics['epochs'], metrics['explained_variances'], linewidth=2, color='green')
    plt.title('Value Function Explained Variance', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Explained Variance')
    plt.grid(True, alpha=0.3)
    
    # 11. Trust Region Visualization
    ax11 = plt.subplot(3, 4, 11)
    scatter = plt.scatter(metrics['kl_divergences'], metrics['step_sizes'], 
                         c=metrics['epochs'], cmap='viridis', alpha=0.6)
    plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.7, label='KL Limit')
    plt.title('Trust Region Behavior', fontsize=12, fontweight='bold')
    plt.xlabel('KL Divergence')
    plt.ylabel('Step Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Epoch')
    
    # 12. Algorithm Efficiency
    ax12 = plt.subplot(3, 4, 12)
    efficiency = metrics['surrogate_improvements'] / metrics['line_search_steps']
    plt.plot(metrics['epochs'], efficiency, linewidth=2, color='purple')
    plt.title('Algorithm Efficiency\n(Improvement/Line Search Steps)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Efficiency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_trust_region_demonstration():
    """Create a visual demonstration of the trust region concept."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Trust Region Concept
    ax = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Current policy (center)
    ax.plot(0, 0, 'ro', markersize=10, label='Current Policy π_old')
    
    # Trust region boundary
    trust_region_x = 0.5 * np.cos(theta)
    trust_region_y = 0.5 * np.sin(theta)
    ax.plot(trust_region_x, trust_region_y, 'r--', linewidth=2, label='Trust Region (KL ≤ δ)')
    ax.fill(trust_region_x, trust_region_y, alpha=0.2, color='red')
    
    # Candidate new policies
    np.random.seed(42)
    for i in range(8):
        angle = i * np.pi / 4
        radius = 0.3 + 0.2 * np.random.random()
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        if radius <= 0.5:
            ax.plot(x, y, 'go', markersize=8, alpha=0.7)
        else:
            ax.plot(x, y, 'rx', markersize=8, alpha=0.7)
    
    ax.plot([], [], 'go', markersize=8, label='Valid Updates')
    ax.plot([], [], 'rx', markersize=8, label='Invalid Updates')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title('Trust Region Concept', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KL Divergence vs Performance
    ax = axes[0, 1]
    kl_values = np.linspace(0, 0.05, 100)
    # Simulated performance improvement (decreases as KL increases beyond optimal point)
    performance = np.exp(-50 * (kl_values - 0.01)**2) * (1 - kl_values/0.05)
    
    ax.plot(kl_values, performance, 'b-', linewidth=3, label='Performance Improvement')
    ax.axvline(x=0.01, color='red', linestyle='--', linewidth=2, label='TRPO KL Limit')
    ax.fill_between(kl_values[kl_values <= 0.01], performance[kl_values <= 0.01], 
                    alpha=0.3, color='green', label='Safe Region')
    ax.fill_between(kl_values[kl_values > 0.01], performance[kl_values > 0.01], 
                    alpha=0.3, color='red', label='Risky Region')
    
    ax.set_xlabel('KL Divergence')
    ax.set_ylabel('Performance Improvement')
    ax.set_title('KL Divergence vs Performance Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Conjugate Gradient Convergence
    ax = axes[1, 0]
    iterations = np.arange(1, 11)
    # Simulated CG convergence (exponential decay)
    residual = np.exp(-0.8 * iterations) + 0.01 * np.random.random(10)
    
    ax.semilogy(iterations, residual, 'bo-', linewidth=2, markersize=6)
    ax.axhline(y=1e-8, color='red', linestyle='--', label='Convergence Threshold')
    ax.set_xlabel('CG Iteration')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Conjugate Gradient Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Line Search Process
    ax = axes[1, 1]
    step_fractions = [1.0, 0.8, 0.64, 0.512, 0.41]
    kl_values = [0.025, 0.018, 0.012, 0.008, 0.006]
    improvements = [0.15, 0.12, 0.08, 0.05, 0.03]
    
    colors = ['red' if kl > 0.01 else 'green' for kl in kl_values]
    bars = ax.bar(range(len(step_fractions)), improvements, color=colors, alpha=0.7)
    
    # Add KL values as text on bars
    for i, (bar, kl) in enumerate(zip(bars, kl_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'KL={kl:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Line Search Step')
    ax.set_ylabel('Surrogate Improvement')
    ax.set_title('Backtracking Line Search Process', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(step_fractions)))
    ax.set_xticklabels([f'α^{i}' for i in range(len(step_fractions))])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='KL > δ (rejected)'),
                      Patch(facecolor='green', alpha=0.7, label='KL ≤ δ (accepted)')]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_trust_region_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_policy_evolution_visualization():
    """Visualize how the policy evolves during training."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs_to_show = [0, 20, 40, 60, 80, 99]
    
    np.random.seed(42)
    
    for idx, epoch in enumerate(epochs_to_show):
        ax = axes[idx // 3, idx % 3]
        
        # Simulate policy evolution - actions become more concentrated over time
        concentration = 1 + epoch / 20  # Increasing concentration
        
        # Generate action samples
        actions = np.random.normal(0, 1/concentration, 1000)
        
        # Create action distribution plot
        ax.hist(actions, bins=30, alpha=0.7, density=True, color=f'C{idx}')
        ax.set_title(f'Action Distribution at Epoch {epoch}', fontweight='bold')
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.axvline(np.mean(actions), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(actions):.2f}')
        ax.axvline(np.mean(actions) + np.std(actions), color='orange', linestyle=':', alpha=0.8, label=f'Std: {np.std(actions):.2f}')
        ax.axvline(np.mean(actions) - np.std(actions), color='orange', linestyle=':', alpha=0.8)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_policy_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_algorithm_comparison():
    """Create a comparison of TRPO with other policy gradient methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = np.arange(100)
    
    # Simulate different algorithms
    np.random.seed(42)
    
    # TRPO: Stable, monotonic improvement
    trpo_returns = -1000 + 800 * (1 - np.exp(-epochs / 30)) + 30 * np.random.normal(0, 1, 100)
    
    # Vanilla PG: More volatile, slower convergence
    vpg_returns = -1000 + 600 * (1 - np.exp(-epochs / 50)) + 80 * np.random.normal(0, 1, 100)
    
    # PPO: Similar to TRPO but slightly different trajectory
    ppo_returns = -1000 + 750 * (1 - np.exp(-epochs / 35)) + 40 * np.random.normal(0, 1, 100)
    
    # 1. Learning Curves Comparison
    ax = axes[0, 0]
    ax.plot(epochs, trpo_returns, label='TRPO', linewidth=2, color='blue')
    ax.plot(epochs, vpg_returns, label='Vanilla PG', linewidth=2, color='red')
    ax.plot(epochs, ppo_returns, label='PPO', linewidth=2, color='green')
    ax.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sample Efficiency
    ax = axes[0, 1]
    sample_steps = epochs * 2000  # 2000 steps per epoch
    ax.plot(sample_steps, trpo_returns, label='TRPO', linewidth=2, color='blue')
    ax.plot(sample_steps, vpg_returns, label='Vanilla PG', linewidth=2, color='red')
    ax.plot(sample_steps, ppo_returns, label='PPO', linewidth=2, color='green')
    ax.set_title('Sample Efficiency', fontsize=14, fontweight='bold')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Average Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Stability Analysis (Rolling Standard Deviation)
    window = 10
    trpo_std = np.array([np.std(trpo_returns[max(0, i-window):i+1]) for i in range(len(trpo_returns))])
    vpg_std = np.array([np.std(vpg_returns[max(0, i-window):i+1]) for i in range(len(vpg_returns))])
    ppo_std = np.array([np.std(ppo_returns[max(0, i-window):i+1]) for i in range(len(ppo_returns))])
    
    ax = axes[1, 0]
    ax.plot(epochs, trpo_std, label='TRPO', linewidth=2, color='blue')
    ax.plot(epochs, vpg_std, label='Vanilla PG', linewidth=2, color='red')
    ax.plot(epochs, ppo_std, label='PPO', linewidth=2, color='green')
    ax.set_title('Training Stability (Rolling Std)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Return Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final Performance Distribution
    ax = axes[1, 1]
    final_trpo = trpo_returns[-20:]  # Last 20 epochs
    final_vpg = vpg_returns[-20:]
    final_ppo = ppo_returns[-20:]
    
    ax.hist(final_trpo, alpha=0.7, label='TRPO', bins=10, color='blue')
    ax.hist(final_vpg, alpha=0.7, label='Vanilla PG', bins=10, color='red')
    ax.hist(final_ppo, alpha=0.7, label='PPO', bins=10, color='green')
    ax.set_title('Final Performance Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_mathematical_visualization():
    """Create visualizations of the mathematical concepts in TRPO."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Surrogate Objective Function
    ax = axes[0, 0]
    theta_diff = np.linspace(-0.5, 0.5, 100)
    
    # L(θ) = E[π(a|s;θ)/π(a|s;θ_old) * A(s,a)]
    # Simplified visualization
    surrogate = theta_diff * (1 - 0.5 * theta_diff**2)  # Quadratic approximation
    
    ax.plot(theta_diff, surrogate, 'b-', linewidth=3, label='Surrogate Objective L(θ)')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Current Policy θ_old')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Policy Parameter Difference (θ - θ_old)')
    ax.set_ylabel('Surrogate Objective Value')
    ax.set_title('Surrogate Objective Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KL Divergence Constraint
    ax = axes[0, 1]
    theta_range = np.linspace(-1, 1, 100)
    kl_divergence = theta_range**2  # Quadratic in parameter difference
    
    ax.plot(theta_range, kl_divergence, 'r-', linewidth=3, label='KL Divergence')
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='KL Constraint δ=0.01')
    ax.fill_between(theta_range, 0, kl_divergence, where=(kl_divergence <= 0.01), 
                    alpha=0.3, color='green', label='Feasible Region')
    ax.set_xlabel('Policy Parameter Difference')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Constraint', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Natural Gradient vs Standard Gradient
    ax = axes[1, 0]
    
    # Create a simple 2D visualization
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Objective function (elliptical)
    Z = -(X**2 + 4*Y**2)
    
    ax.contour(X, Y, Z, levels=10, alpha=0.6)
    
    # Standard gradient (steepest descent)
    ax.arrow(0, 0, 0.5, -0.25, head_width=0.1, head_length=0.1, fc='red', ec='red', label='Standard Gradient')
    
    # Natural gradient (follows natural geometry)
    ax.arrow(0, 0, 1.0, -0.125, head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='Natural Gradient')
    
    ax.set_xlabel('Parameter θ₁')
    ax.set_ylabel('Parameter θ₂')
    ax.set_title('Natural vs Standard Gradient', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 4. Advantage Function Evolution
    ax = axes[1, 1]
    states = np.linspace(0, 10, 100)
    
    # Simulate advantage functions at different training stages
    early_advantage = np.sin(states) + 0.5 * np.random.normal(0, 1, 100)
    mid_advantage = 2 * np.sin(states + 1) + 0.3 * np.random.normal(0, 1, 100)
    late_advantage = 3 * np.sin(states + 2) + 0.1 * np.random.normal(0, 1, 100)
    
    ax.plot(states, early_advantage, label='Early Training', alpha=0.7, linewidth=2)
    ax.plot(states, mid_advantage, label='Mid Training', alpha=0.7, linewidth=2)
    ax.plot(states, late_advantage, label='Late Training', alpha=0.7, linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('State')
    ax.set_ylabel('Advantage A(s,a)')
    ax.set_title('Advantage Function Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_mathematical_concepts.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run all visualizations."""
    print("Creating TRPO Algorithm Visualizations")
    print("=" * 50)
    
    print("1. Generating comprehensive training analysis...")
    create_comprehensive_visualizations()
    
    print("2. Creating trust region demonstration...")
    create_trust_region_demonstration()
    
    print("3. Visualizing policy evolution...")
    create_policy_evolution_visualization()
    
    print("4. Comparing with other algorithms...")
    create_algorithm_comparison()
    
    print("5. Illustrating mathematical concepts...")
    create_mathematical_visualization()
    
    print("\nAll visualizations completed!")
    print("Generated files:")
    print("- trpo_comprehensive_analysis.png: Complete training metrics")
    print("- trpo_trust_region_demo.png: Trust region concept")
    print("- trpo_policy_evolution.png: Policy evolution over time")
    print("- trpo_algorithm_comparison.png: Comparison with other methods")
    print("- trpo_mathematical_concepts.png: Mathematical foundations")

if __name__ == "__main__":
    main()

