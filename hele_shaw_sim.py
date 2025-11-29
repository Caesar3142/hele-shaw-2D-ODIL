import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------------------------------
# 1. Laplace solver for basis flows (Hele-Shaw) - simplified
# -------------------------------------------------------

def solve_laplace(N, bc_type='horizontal'):
    """
    Solve Laplace equation with different boundary conditions
    bc_type: 'horizontal', 'vertical', or 'radial'
    """
    p = np.zeros((N, N))
    
    # Set boundary conditions based on type (use larger pressure differences)
    if bc_type == 'horizontal':
        # Left wall high, right wall low -> horizontal flow
        p[:, 0] = 100.0   # Left boundary (high pressure)
        p[:, -1] = 0.0    # Right boundary (low pressure)
    elif bc_type == 'vertical':
        # Top wall high, bottom wall low -> vertical flow
        p[0, :] = 100.0   # Top boundary (high pressure)
        p[-1, :] = 0.0    # Bottom boundary (low pressure)
    elif bc_type == 'radial':
        # Diagonal gradient with larger amplitude
        for i in range(N):
            for j in range(N):
                p[i, j] = 50.0 + 30.0 * np.sin(np.pi * (i + j) / N)
    
    # Iterative solver (Jacobi method)
    for _ in range(2000):
        p_new = p.copy()
        p_new[1:-1, 1:-1] = 0.25 * (
            p[:-2, 1:-1] +
            p[2:, 1:-1] +
            p[1:-1, :-2] +
            p[1:-1, 2:]
        )
        # Maintain boundary conditions
        if bc_type == 'horizontal':
            p_new[:, 0] = 100.0
            p_new[:, -1] = 0.0
        elif bc_type == 'vertical':
            p_new[0, :] = 100.0
            p_new[-1, :] = 0.0
        p = p_new
    
    return p

def compute_velocity(p, h=0.1, mu=1.0, scale=1.0):
    """
    Compute velocity from pressure gradient
    scale: Additional scaling factor to make velocities more visible
    """
    px = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / 2
    py = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2
    u = -(h*h / (12*mu)) * px * scale
    v = -(h*h / (12*mu)) * py * scale
    return np.stack([u, v], axis=-1)

# -------------------------------------------------------
# 2. Basis flows for 3 wall motions (placeholder version)
# -------------------------------------------------------

def precompute_basis(N=64, velocity_scale=50.0):
    """
    Create three different basis flows with different boundary conditions
    velocity_scale: Scale factor to make velocities more visible (default 50.0)
    """
    # Create three different basis flows with different boundary conditions
    p1 = solve_laplace(N, bc_type='horizontal')  # Horizontal flow
    p2 = solve_laplace(N, bc_type='vertical')    # Vertical flow
    p3 = solve_laplace(N, bc_type='radial')      # Radial/diagonal flow

    U1 = compute_velocity(p1, scale=velocity_scale)
    U2 = compute_velocity(p2, scale=velocity_scale)
    U3 = compute_velocity(p3, scale=velocity_scale)

    return U1, U2, U3

# -------------------------------------------------------
# 3. Bilinear interpolation on the grid
# -------------------------------------------------------

def interp(U, x, y):
    N = U.shape[0]
    xg = x * (N - 1)
    yg = y * (N - 1)

    i = int(np.floor(yg))
    j = int(np.floor(xg))

    i = max(0, min(N-1, i))
    j = max(0, min(N-1, j))

    di = yg - i
    dj = xg - j

    i1 = min(i+1, N-1)
    j1 = min(j+1, N-1)

    v00 = U[i, j]
    v10 = U[i1, j]
    v01 = U[i, j1]
    v11 = U[i1, j1]

    return (1-di)*(1-dj)*v00 + di*(1-dj)*v10 + (1-di)*dj*v01 + di*dj*v11

# -------------------------------------------------------
# 4. Flow and particle dynamics
# -------------------------------------------------------

def flow(x, y, omega, U1, U2, U3):
    U = omega[0]*U1 + omega[1]*U2 + omega[2]*U3
    return interp(U, x, y)

def midpoint_step(pos, omega, dt, U1, U2, U3):
    xm, ym = pos
    um = flow(xm, ym, omega, U1, U2, U3)
    mid = pos + 0.5 * dt * um
    u_mid = flow(mid[0], mid[1], omega, U1, U2, U3)
    return pos + dt * u_mid

# -------------------------------------------------------
# 5. Simple "policy" (linear tanh) as a placeholder controller
# -------------------------------------------------------

def policy(pos, w):
    x, y = pos
    inp = np.array([x, y, 1.0])  # [x, y, bias]
    omega = np.tanh(w @ inp)    # 3 outputs in [-1,1]
    return omega

# -------------------------------------------------------
# 6. Very simple training loop (finite difference gradient)
# -------------------------------------------------------

def train(visualize=True):
    U1, U2, U3 = precompute_basis()

    # weights for policy: shape (3 actuators, 3 inputs)
    w = 0.01 * np.random.randn(3, 3)

    target = np.array([0.8, 0.8])
    dt = 0.1  # Increased time step to make trajectories more visible
    
    # Track training history
    loss_history = []
    trajectory_history = []

    for epoch in range(50):
        pos = np.array([0.2, 0.2])
        loss = 0.0
        trajectory = [pos.copy()]

        # simulate one trajectory (increased steps for better visibility)
        for _ in range(500):
            omega = policy(pos, w)
            pos = midpoint_step(pos, omega, dt, U1, U2, U3)
            trajectory.append(pos.copy())
            loss += np.sum((pos - target)**2)
        
        trajectory_history.append(np.array(trajectory))
        loss_history.append(loss)

        # gradient by finite difference (slow but simple)
        grad_w = np.zeros_like(w)
        eps = 1e-3

        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w2 = w.copy()
                w2[i, j] += eps

                pos2 = np.array([0.2, 0.2])
                loss2 = 0.0
                for _ in range(500):
                    omega2 = policy(pos2, w2)
                    pos2 = midpoint_step(pos2, omega2, dt, U1, U2, U3)
                    loss2 += np.sum((pos2 - target)**2)

                grad_w[i, j] = (loss2 - loss) / eps

        # update weights
        w -= 0.01 * grad_w

        print(f"Epoch {epoch}, Loss {loss:.4f}")

    return w, U1, U2, U3, loss_history, trajectory_history

# -------------------------------------------------------
# 7. Visualization functions
# -------------------------------------------------------

def visualize_flow_field(U1, U2, U3, omega, ax=None):
    """Visualize the flow field with velocity vectors and streamlines"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    N = U1.shape[0]
    U = omega[0]*U1 + omega[1]*U2 + omega[2]*U3
    
    # Create grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # Extract velocity components
    u = U[:, :, 0]
    v = U[:, :, 1]
    
    # Calculate speed for coloring
    speed = np.sqrt(u**2 + v**2)
    max_speed = np.max(speed) if np.max(speed) > 0 else 1.0
    
    # Plot speed as background colormap
    im = ax.imshow(speed, extent=[0, 1, 0, 1], origin='lower', 
                   cmap='Blues', alpha=0.5, interpolation='bilinear',
                   vmin=0, vmax=max_speed)
    plt.colorbar(im, ax=ax, label='Velocity magnitude')
    
    # Only plot streamlines if there's meaningful flow
    if max_speed > 1e-6:
        try:
            ax.streamplot(X, Y, u, v, density=1.5, color='darkblue', 
                         linewidth=1.5, arrowsize=1.2)
        except:
            pass  # Skip if streamlines fail
    
    # Plot velocity field as quiver (subsampled for clarity)
    skip = max(1, N // 12)
    if max_speed > 1e-6:
        scale_factor = 20.0 / max_speed if max_speed > 0 else 20.0
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  u[::skip, ::skip], v[::skip, ::skip],
                  scale=scale_factor, alpha=0.7, color='darkred', width=0.004)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Flow Field (ω = [{omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f}])')
    
    return ax

def visualize_trajectory(trajectory, target, ax=None, title="Particle Trajectory", color='red'):
    """Visualize particle trajectory"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    traj = np.array(trajectory)
    
    # Check if trajectory actually moved
    movement = np.linalg.norm(traj[-1] - traj[0])
    
    # Plot trajectory with arrow markers to show direction
    ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2.5, alpha=0.8, 
            label=f'Trajectory (distance: {movement:.3f})', zorder=3)
    
    # Add arrows to show direction (every Nth point)
    if len(traj) > 20:
        step = max(1, len(traj) // 15)
        for i in range(step, len(traj)-step, step*2):
            dx = traj[i, 0] - traj[i-step, 0]
            dy = traj[i, 1] - traj[i-step, 1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                ax.arrow(traj[i-step, 0], traj[i-step, 1], dx, dy,
                        head_width=0.01, head_length=0.01, fc=color, ec=color, 
                        alpha=0.5, length_includes_head=True)
    
    # Mark trajectory points at key locations
    if len(traj) > 10:
        step = len(traj) // 10
        ax.plot(traj[::step, 0], traj[::step, 1], 'o', color=color, 
                markersize=5, alpha=0.7, zorder=4, markeredgecolor='black', markeredgewidth=0.5)
    
    # Mark start, end, and target with larger, more visible markers
    ax.scatter(traj[0, 0], traj[0, 1], color='green', s=200, marker='o', 
               label='Start', zorder=6, edgecolors='black', linewidths=2)
    ax.scatter(traj[-1, 0], traj[-1, 1], color='orange', s=200, marker='s', 
               label='End', zorder=6, edgecolors='black', linewidths=2)
    ax.scatter(target[0], target[1], color='red', s=250, marker='*', 
               label='Target', zorder=6, edgecolors='black', linewidths=2)
    
    # Set limits to show full domain
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    return ax

def plot_training_loss(loss_history, ax=None):
    """Plot training loss over epochs"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = np.arange(len(loss_history))
    ax.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # Log scale for better visualization
    
    return ax

def simulate_trajectory(w, U1, U2, U3, start_pos, target, dt=0.01, n_steps=200):
    """Simulate a trajectory with the trained policy"""
    pos = start_pos.copy()
    trajectory = [pos.copy()]
    
    for _ in range(n_steps):
        omega = policy(pos, w)
        pos = midpoint_step(pos, omega, dt, U1, U2, U3)
        trajectory.append(pos.copy())
    
    return np.array(trajectory)

# -------------------------------------------------------
# 8. Main entry
# -------------------------------------------------------

if __name__ == "__main__":
    print("Starting training...")
    print("Precomputing basis flows...")
    w, U1, U2, U3, loss_history, trajectory_history = train()
    print("Training complete.")
    
    # Debug: Check if flows are non-zero
    print(f"\nDebug info:")
    print(f"  U1 velocity range: [{U1.min():.4f}, {U1.max():.4f}]")
    print(f"  U2 velocity range: [{U2.min():.4f}, {U2.max():.4f}]")
    print(f"  U3 velocity range: [{U3.min():.4f}, {U3.max():.4f}]")
    print(f"  First trajectory length: {len(trajectory_history[0])} points")
    print(f"  First trajectory start: {trajectory_history[0][0]}")
    print(f"  First trajectory end: {trajectory_history[0][-1]}")
    print(f"  Trajectory moved: {np.linalg.norm(trajectory_history[0][-1] - trajectory_history[0][0]):.4f}")
    
    print("\nGenerating visualizations...")
    
    # 1. Plot training loss
    fig1 = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plot_training_loss(loss_history)
    
    # 2. Show first trajectory (before training)
    plt.subplot(2, 2, 2)
    visualize_trajectory(trajectory_history[0], np.array([0.8, 0.8]), 
                        title="Initial Trajectory (Epoch 0)")
    
    # 3. Show last trajectory (after training)
    plt.subplot(2, 2, 3)
    visualize_trajectory(trajectory_history[-1], np.array([0.8, 0.8]), 
                        title="Final Trajectory (After Training)", color='purple')
    
    # 4. Show flow field at final state
    plt.subplot(2, 2, 4)
    final_pos = trajectory_history[-1][-1]
    final_omega = policy(final_pos, w)
    print(f"  Flow field omega: {final_omega}")
    visualize_flow_field(U1, U2, U3, final_omega)
    
    plt.tight_layout()
    plt.savefig('hele_shaw_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'hele_shaw_results.png'")
    
    # 5. Create a detailed trajectory comparison
    fig2 = plt.figure(figsize=(15, 5))
    
    # Show trajectories at different epochs
    epochs_to_show = [0, len(trajectory_history)//4, len(trajectory_history)//2, -1]
    for idx, epoch in enumerate(epochs_to_show):
        plt.subplot(1, 4, idx+1)
        visualize_trajectory(trajectory_history[epoch], np.array([0.8, 0.8]),
                            title=f"Epoch {epoch} (Loss: {loss_history[epoch]:.2f})",
                            color=plt.cm.viridis(epoch / len(trajectory_history)))
    
    plt.tight_layout()
    plt.savefig('hele_shaw_trajectory_evolution.png', dpi=150, bbox_inches='tight')
    print("Trajectory evolution saved to 'hele_shaw_trajectory_evolution.png'")
    
    print("\nFinal distance to target:", 
          np.linalg.norm(trajectory_history[-1][-1] - np.array([0.8, 0.8])))
    
    # Show all plots
    print("\nDisplaying plots...")
    plt.show()
