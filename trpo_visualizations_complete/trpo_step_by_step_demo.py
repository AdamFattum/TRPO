"""
TRPO Step-by-Step Algorithm Demonstration

This script demonstrates the TRPO algorithm step by step with detailed
explanations and visualizations of each component.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def demonstrate_conjugate_gradient():
    """Demonstrate the conjugate gradient method used in TRPO."""
    print("CONJUGATE GRADIENT METHOD DEMONSTRATION")
    print("=" * 50)
    
    # Create a simple quadratic problem: minimize 0.5 * x^T A x - b^T x
    A = np.array([[4, 1], [1, 3]])  # Positive definite matrix
    b = np.array([1, 2])
    
    # True solution
    x_true = np.linalg.solve(A, b)
    print(f"True solution: x* = [{x_true[0]:.3f}, {x_true[1]:.3f}]")
    
    # Conjugate gradient iterations
    x = np.array([0.0, 0.0])  # Initial guess
    r = b - A @ x  # Initial residual
    p = r.copy()   # Initial search direction
    
    # Store trajectory
    trajectory = [x.copy()]
    residuals = [np.linalg.norm(r)]
    
    print("\nConjugate Gradient Iterations:")
    print("Iter |    x₁    |    x₂    | Residual")
    print("-" * 40)
    print(f"  0  | {x[0]:7.3f} | {x[1]:7.3f} | {residuals[0]:7.3f}")
    
    for i in range(5):  # Maximum 5 iterations
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        trajectory.append(x.copy())
        residuals.append(np.linalg.norm(r_new))
        
        print(f"  {i+1}  | {x[0]:7.3f} | {x[1]:7.3f} | {residuals[-1]:7.3f}")
        
        if np.linalg.norm(r_new) < 1e-6:
            print(f"Converged in {i+1} iterations!")
            break
        
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    
    # Visualize the convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Contour plot showing trajectory
    x1_range = np.linspace(-0.5, 1.0, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Objective function values
    Z = 0.5 * (A[0,0]*X1**2 + 2*A[0,1]*X1*X2 + A[1,1]*X2**2) - b[0]*X1 - b[1]*X2
    
    ax1.contour(X1, X2, Z, levels=20, alpha=0.6)
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2, markersize=6, label='CG Trajectory')
    ax1.plot(x_true[0], x_true[1], 'g*', markersize=15, label='True Solution')
    ax1.plot(0, 0, 'bs', markersize=8, label='Starting Point')
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Conjugate Gradient Convergence Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual convergence
    ax2.semilogy(range(len(residuals)), residuals, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual Norm (log scale)')
    ax2.set_title('Residual Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_conjugate_gradient_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_line_search():
    """Demonstrate the backtracking line search used in TRPO."""
    print("\nLINE SEARCH DEMONSTRATION")
    print("=" * 50)
    
    # Simulate a line search scenario
    def surrogate_objective(step_size):
        """Simulated surrogate objective improvement."""
        return 0.1 * step_size * (1 - 2 * step_size)
    
    def kl_divergence(step_size):
        """Simulated KL divergence as function of step size."""
        return step_size**2
    
    max_kl = 0.01
    backtrack_coeff = 0.8
    initial_step = 0.2
    
    print(f"Initial step size: {initial_step}")
    print(f"KL constraint: {max_kl}")
    print(f"Backtrack coefficient: {backtrack_coeff}")
    print()
    
    step_sizes = []
    kl_values = []
    improvements = []
    accepted = []
    
    current_step = initial_step
    
    print("Step | Step Size | KL Div  | Improvement | Accept?")
    print("-" * 50)
    
    for i in range(6):
        kl_val = kl_divergence(current_step)
        improvement = surrogate_objective(current_step)
        
        step_sizes.append(current_step)
        kl_values.append(kl_val)
        improvements.append(improvement)
        
        # Check constraints
        kl_ok = kl_val <= max_kl
        improvement_ok = improvement > 0
        accept = kl_ok and improvement_ok
        accepted.append(accept)
        
        print(f"  {i}  | {current_step:8.4f} | {kl_val:6.4f} | {improvement:10.4f} | {'✓' if accept else '✗'}")
        
        if accept:
            print(f"\nAccepted step size: {current_step:.4f}")
            break
        
        current_step *= backtrack_coeff
    
    # Visualize line search
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Step size vs metrics
    x_pos = range(len(step_sizes))
    colors = ['green' if acc else 'red' for acc in accepted]
    
    ax1.bar([x - 0.2 for x in x_pos], improvements, width=0.4, 
            color=colors, alpha=0.7, label='Surrogate Improvement')
    ax1.bar([x + 0.2 for x in x_pos], [kl * 10 for kl in kl_values], width=0.4, 
            color='blue', alpha=0.7, label='KL Divergence (×10)')
    
    ax1.axhline(y=max_kl * 10, color='red', linestyle='--', alpha=0.7, label='KL Limit (×10)')
    ax1.set_xlabel('Line Search Step')
    ax1.set_ylabel('Value')
    ax1.set_title('Line Search Progress')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'α^{i}' for i in range(len(step_sizes))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Step size trajectory
    step_range = np.linspace(0, 0.25, 100)
    kl_curve = step_range**2
    improvement_curve = 0.1 * step_range * (1 - 2 * step_range)
    
    ax2.plot(step_range, improvement_curve, 'b-', linewidth=2, label='Surrogate Improvement')
    ax2.plot(step_range, kl_curve, 'r-', linewidth=2, label='KL Divergence')
    ax2.axhline(y=max_kl, color='red', linestyle='--', alpha=0.7, label='KL Constraint')
    
    # Mark tested points
    for i, (step, kl, imp, acc) in enumerate(zip(step_sizes, kl_values, improvements, accepted)):
        marker = 'go' if acc else 'rx'
        ax2.plot(step, imp, marker, markersize=8, label=f'Step {i}' if i < 3 else "")
        ax2.plot(step, kl, marker, markersize=8)
    
    ax2.set_xlabel('Step Size')
    ax2.set_ylabel('Value')
    ax2.set_title('Line Search Landscape')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_line_search_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_advantage_estimation():
    """Demonstrate Generalized Advantage Estimation (GAE)."""
    print("\nGENERALIZED ADVANTAGE ESTIMATION (GAE)")
    print("=" * 50)
    
    # Simulate a trajectory
    np.random.seed(42)
    T = 20  # Trajectory length
    gamma = 0.99  # Discount factor
    lam = 0.95    # GAE lambda
    
    # Simulated rewards and values
    rewards = np.random.normal(0, 1, T)
    values = np.random.normal(0, 0.5, T + 1)  # Include terminal value
    
    print(f"Trajectory length: {T}")
    print(f"Discount factor γ: {gamma}")
    print(f"GAE lambda λ: {lam}")
    print()
    
    # Compute TD errors
    td_errors = rewards + gamma * values[1:] - values[:-1]
    
    # Compute advantages using GAE
    advantages = np.zeros(T)
    gae = 0
    
    for t in reversed(range(T)):
        gae = td_errors[t] + gamma * lam * gae
        advantages[t] = gae
    
    # Compute returns
    returns = np.zeros(T)
    discounted_return = 0
    for t in reversed(range(T)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return
    
    print("Time | Reward | Value  | TD Error | Advantage | Return")
    print("-" * 60)
    for t in range(min(10, T)):  # Show first 10 steps
        print(f" {t:2d}  | {rewards[t]:6.3f} | {values[t]:6.3f} | {td_errors[t]:8.3f} | {advantages[t]:9.3f} | {returns[t]:6.3f}")
    
    if T > 10:
        print("...")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    time_steps = range(T)
    
    # Rewards and values
    ax = axes[0, 0]
    ax.plot(time_steps, rewards, 'bo-', label='Rewards', alpha=0.7)
    ax.plot(time_steps, values[:-1], 'ro-', label='State Values', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Rewards and State Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TD errors
    ax = axes[0, 1]
    ax.plot(time_steps, td_errors, 'go-', label='TD Errors', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('TD Error')
    ax.set_title('Temporal Difference Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Advantages
    ax = axes[1, 0]
    ax.plot(time_steps, advantages, 'mo-', label='GAE Advantages', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Advantage')
    ax.set_title('Generalized Advantage Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Returns
    ax = axes[1, 1]
    ax.plot(time_steps, returns, 'co-', label='Returns', alpha=0.7)
    ax.plot(time_steps, values[:-1], 'ro-', label='Value Estimates', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Returns vs Value Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_gae_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_trust_region_update():
    """Demonstrate the complete TRPO update step."""
    print("\nTRPO UPDATE STEP DEMONSTRATION")
    print("=" * 50)
    
    # Simulate policy parameters (2D for visualization)
    theta_old = np.array([0.0, 0.0])
    
    # Simulate gradient of surrogate objective
    g = np.array([0.5, 0.3])  # Policy gradient
    
    # Simulate Fisher Information Matrix (Hessian of KL divergence)
    F = np.array([[2.0, 0.5], [0.5, 1.5]])  # Positive definite
    
    # Natural gradient: F^(-1) * g
    natural_grad = np.linalg.solve(F, g)
    
    # Step size from trust region constraint
    max_kl = 0.01
    quad_form = natural_grad.T @ F @ natural_grad
    step_size = np.sqrt(2 * max_kl / quad_form)
    
    # Proposed update
    theta_new = theta_old + step_size * natural_grad
    
    print(f"Current policy parameters: θ_old = [{theta_old[0]:.3f}, {theta_old[1]:.3f}]")
    print(f"Policy gradient: g = [{g[0]:.3f}, {g[1]:.3f}]")
    print(f"Fisher Information Matrix:")
    print(f"F = [[{F[0,0]:.3f}, {F[0,1]:.3f}],")
    print(f"     [{F[1,0]:.3f}, {F[1,1]:.3f}]]")
    print(f"Natural gradient: F^(-1)g = [{natural_grad[0]:.3f}, {natural_grad[1]:.3f}]")
    print(f"Step size: α = {step_size:.6f}")
    print(f"New policy parameters: θ_new = [{theta_new[0]:.3f}, {theta_new[1]:.3f}]")
    
    # Verify KL constraint
    kl_divergence = 0.5 * (theta_new - theta_old).T @ F @ (theta_new - theta_old)
    print(f"KL divergence: {kl_divergence:.6f} (constraint: {max_kl:.6f})")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Trust region visualization
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Trust region boundary (ellipse defined by KL constraint)
    # (θ - θ_old)^T F (θ - θ_old) = 2 * max_kl
    eigenvals, eigenvecs = np.linalg.eig(F)
    a = np.sqrt(2 * max_kl / eigenvals[0])  # Semi-major axis
    b = np.sqrt(2 * max_kl / eigenvals[1])  # Semi-minor axis
    
    # Ellipse points
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)
    
    # Rotate ellipse
    rotation_matrix = eigenvecs
    ellipse_points = rotation_matrix @ np.array([ellipse_x, ellipse_y])
    ellipse_x_rot = ellipse_points[0] + theta_old[0]
    ellipse_y_rot = ellipse_points[1] + theta_old[1]
    
    ax1.plot(ellipse_x_rot, ellipse_y_rot, 'r--', linewidth=2, label='Trust Region Boundary')
    ax1.fill(ellipse_x_rot, ellipse_y_rot, alpha=0.2, color='red')
    
    # Current and new policy
    ax1.plot(theta_old[0], theta_old[1], 'bo', markersize=10, label='Current Policy θ_old')
    ax1.plot(theta_new[0], theta_new[1], 'go', markersize=10, label='New Policy θ_new')
    
    # Gradients
    ax1.arrow(theta_old[0], theta_old[1], 0.1*g[0], 0.1*g[1], 
              head_width=0.02, head_length=0.02, fc='blue', ec='blue', label='Policy Gradient')
    ax1.arrow(theta_old[0], theta_old[1], step_size*natural_grad[0], step_size*natural_grad[1], 
              head_width=0.02, head_length=0.02, fc='green', ec='green', label='Natural Gradient Step')
    
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_title('TRPO Policy Update')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Comparison of gradient directions
    ax2.arrow(0, 0, g[0], g[1], head_width=0.05, head_length=0.05, 
              fc='blue', ec='blue', label='Standard Gradient', width=0.02)
    ax2.arrow(0, 0, natural_grad[0], natural_grad[1], head_width=0.05, head_length=0.05, 
              fc='red', ec='red', label='Natural Gradient', width=0.02)
    
    ax2.set_xlabel('Direction Component 1')
    ax2.set_ylabel('Direction Component 2')
    ax2.set_title('Gradient Direction Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/trpo_update_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run all step-by-step demonstrations."""
    print("TRPO ALGORITHM STEP-BY-STEP DEMONSTRATION")
    print("=" * 60)
    print()
    
    demonstrate_conjugate_gradient()
    demonstrate_line_search()
    demonstrate_advantage_estimation()
    demonstrate_trust_region_update()
    
    print("\n" + "=" * 60)
    print("STEP-BY-STEP DEMONSTRATION COMPLETE")
    print("Generated files:")
    print("- trpo_conjugate_gradient_demo.png: Conjugate gradient method")
    print("- trpo_line_search_demo.png: Backtracking line search")
    print("- trpo_gae_demo.png: Generalized Advantage Estimation")
    print("- trpo_update_demo.png: Complete TRPO update step")

if __name__ == "__main__":
    main()

