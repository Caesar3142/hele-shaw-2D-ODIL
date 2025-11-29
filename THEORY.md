# Theory and Implementation

This document describes the physics, equations, and numerical methods used in the Hele-Shaw 2D ODIL (Optimal Design using Implicit Layers) simulation.

## Overview

The code implements an optimal control problem for particle navigation in a 2D Hele-Shaw flow. The goal is to learn a control policy that guides a particle from a starting position to a target location by dynamically adjusting the flow field through a linear combination of three precomputed basis flows.

## Physical Model: Hele-Shaw Flow

Hele-Shaw flow describes the motion of a viscous fluid between two parallel plates separated by a small gap. The key assumptions are:

1. The gap height `h` is much smaller than the characteristic length scale
2. The flow is dominated by viscous forces (low Reynolds number)
3. The flow is approximately 2D, with velocity components `u(x,y)` and `v(x,y)`

### Governing Equations

#### 1. Laplace Equation for Pressure

In Hele-Shaw flow, the pressure field satisfies Laplace's equation:

```
∇²p = ∂²p/∂x² + ∂²p/∂y² = 0
```

This is solved numerically using the Jacobi iterative method on a discrete grid of size `N×N`.

**Numerical discretization (Jacobi method):**
```
p[i,j]^(n+1) = 0.25 × (p[i-1,j]^(n) + p[i+1,j]^(n) + p[i,j-1]^(n) + p[i,j+1]^(n))
```

Boundary conditions are set according to the flow type:
- **Horizontal flow**: `p(x=0) = 100`, `p(x=1) = 0`
- **Vertical flow**: `p(y=0) = 0`, `p(y=1) = 100`
- **Radial flow**: Sinusoidal pressure distribution

#### 2. Velocity-Pressure Relationship

The velocity components are computed from the pressure gradient using the Hele-Shaw formula:

```
u = - (h²/(12μ)) × ∂p/∂x
v = - (h²/(12μ)) × ∂p/∂y
```

Where:
- `h` is the gap height (default: 0.1)
- `μ` is the dynamic viscosity (default: 1.0)
- The negative sign indicates flow from high to low pressure

**Numerical implementation:**
```
px = (p[:, j+1] - p[:, j-1]) / (2×Δx)    [central difference]
py = (p[i+1, :] - p[i-1, :]) / (2×Δy)    [central difference]

u = -(h²/(12μ)) × px × scale
v = -(h²/(12μ)) × py × scale
```

The `scale` parameter is used to make velocities more visible for visualization.

## Basis Flow Decomposition

The code precomputes three basis flow fields `U₁`, `U₂`, `U₃`, each corresponding to different boundary conditions:

1. **U₁**: Horizontal flow (left-to-right pressure gradient)
2. **U₂**: Vertical flow (top-to-bottom pressure gradient)
3. **U₃**: Radial/diagonal flow (sinusoidal pressure distribution)

Any flow field can be represented as a linear combination:

```
U(x,y) = ω₁·U₁(x,y) + ω₂·U₂(x,y) + ω₃·U₃(x,y)
```

Where `ω = [ω₁, ω₂, ω₃]` is the control vector (actuator strengths).

## Particle Dynamics

### Advection Equation

Particles are advected by the flow field according to:

```
dx/dt = u(x, y)
dy/dt = v(x, y)
```

Where `[u, v]` is the velocity vector at position `[x, y]`.

### Numerical Integration: Midpoint Method

The code uses the midpoint (Runge-Kutta 2nd order) method for time integration:

```
k₁ = u(xₙ, yₙ, ω)                    [velocity at current position]
x_mid = xₙ + 0.5×dt×k₁                [midpoint position]
k₂ = u(x_mid, y_mid, ω)               [velocity at midpoint]
xₙ₊₁ = xₙ + dt×k₂                     [updated position]
```

This is more accurate than Euler's method and provides better stability.

### Bilinear Interpolation

Since the flow field is defined on a discrete grid, velocity values at arbitrary positions `(x, y)` are obtained via bilinear interpolation:

```
v(x, y) = (1-δᵢ)(1-δⱼ)·v[i,j] + δᵢ(1-δⱼ)·v[i+1,j] 
        + (1-δᵢ)δⱼ·v[i,j+1] + δᵢδⱼ·v[i+1,j+1]
```

Where:
- `i, j` are the grid indices below `(x, y)`
- `δᵢ, δⱼ` are the fractional parts

## Control Policy

### Policy Function

The control policy maps the particle's current position to actuator strengths:

```
ω = tanh(W · [x, y, 1]ᵀ)
```

Where:
- `W` is a `3×3` weight matrix (3 actuators, 3 inputs: x, y, bias)
- `tanh` ensures the output is bounded in `[-1, 1]`
- The bias term allows for position-independent control

### Flow Composition

Given the control vector `ω`, the actual flow field is:

```
U(x, y, ω) = ω₁·U₁(x, y) + ω₂·U₂(x, y) + ω₃·U₃(x, y)
```

The velocity at position `(x, y)` is then:
```
u(x, y) = interpolate(U(x, y, ω), x, y)
```

## Optimization Problem

### Objective Function

The goal is to minimize the distance between the particle's final position and a target location:

```
L = Σᵢ ||xᵢ - x_target||²
```

Where the sum is over all time steps (or accumulated loss over trajectory).

### Training Algorithm: Finite Difference Gradient Descent

The code uses finite differences to compute gradients:

```
∂L/∂W[i,j] ≈ (L(W + ε·eᵢⱼ) - L(W)) / ε
```

Where:
- `ε` is a small perturbation (default: 1e-3)
- `eᵢⱼ` is a unit vector with 1 at position `(i, j)`

### Weight Update

```
W ← W - α · ∇W L
```

Where `α` is the learning rate (default: 0.01).

### Training Loop

For each epoch:
1. Simulate a trajectory from the initial position `[0.2, 0.2]` using current policy
2. Compute the loss (sum of squared distances to target `[0.8, 0.8]`)
3. Compute gradients using finite differences
4. Update the weight matrix using gradient descent
5. Repeat for 50 epochs

## Code Structure

1. **`solve_laplace(N, bc_type)`**: Solves Laplace equation using Jacobi iteration
2. **`compute_velocity(p, h, mu, scale)`**: Computes velocity from pressure gradient
3. **`precompute_basis(N, velocity_scale)`**: Generates three basis flow fields
4. **`interp(U, x, y)`**: Bilinear interpolation for velocity lookup
5. **`flow(x, y, omega, U1, U2, U3)`**: Computes flow field from basis flows
6. **`midpoint_step(pos, omega, dt, U1, U2, U3)`**: Single time step using midpoint method
7. **`policy(pos, w)`**: Control policy mapping position to actuator strengths
8. **`train()`**: Main training loop with finite difference gradient computation
9. **Visualization functions**: Plot flow fields, trajectories, and training loss

## Numerical Parameters

- Grid size: `N = 64`
- Gap height: `h = 0.1`
- Viscosity: `μ = 1.0`
- Time step: `dt = 0.1`
- Integration steps: `500` per trajectory
- Training epochs: `50`
- Learning rate: `α = 0.01`
- Gradient perturbation: `ε = 1e-3`
- Initial position: `[0.2, 0.2]`
- Target position: `[0.8, 0.8]`

## Key Assumptions and Simplifications

1. **Quasi-2D flow**: The gap height is assumed small enough for 2D approximation
2. **Incompressible flow**: Density is constant
3. **Steady-state pressure**: Pressure field is computed once per basis flow
4. **Linear control**: Flow is a linear combination of basis flows
5. **Simple policy**: Linear transformation with tanh activation
6. **Finite difference gradients**: Simple but computationally expensive gradient computation

## Future Enhancements

Potential improvements:
- Use automatic differentiation (JAX) for efficient gradient computation
- Implement more sophisticated control policies (neural networks)
- Add time-dependent basis flows
- Include obstacles or boundaries in the domain
- Implement more advanced optimization algorithms (Adam, L-BFGS)

