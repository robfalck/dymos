"""
Runnable example demonstrating the Radau brachistochrone phase function.

This script shows how to use the radau_brachistochrone_phase() function to:
1. Set up a brachistochrone optimal control problem
2. Compute residuals (defects, boundary conditions, objective)
3. Use JAX automatic differentiation to compute gradients

This is a demonstration of the phase function's capabilities, not a full
optimization solver. For solving the problem to optimality, an optimizer
like scipy.optimize.minimize would be used with these phase functions.

Usage:
    cd /c/Users/robfa/Codes/dymos.git
    pixi run -e dev python -m dymos.jax.examples.run_radau_brachistochrone_example
"""

import jax
import jax.numpy as jnp
import numpy as np

from dymos.jax.examples.radau_brachistochrone_phase import (
    radau_brachistochrone_phase,
    create_radau_grid_data
)


def create_reasonable_guess(grid_data, x0=0.0, xf=10.0, y0=0.0, yf=-10.0, t_duration=2.0):
    """
    Create a reasonable initial guess for the brachistochrone problem.

    Uses linear interpolation for states between boundary values and
    constant control values.

    Parameters
    ----------
    grid_data : dict
        Grid data structure from create_radau_grid_data()
    x0, xf : float
        Initial and final x positions
    y0, yf : float
        Initial and final y positions
    t_duration : float
        Guess for phase duration

    Returns
    -------
    design_vars : dict
        Design variables with reasonable initial guess
    """
    num_disc = grid_data['num_disc_nodes']
    num_all = grid_data['num_all_nodes']

    # Linear interpolation for states
    tau = np.linspace(0, 1, num_disc)
    x_guess = x0 + tau * (xf - x0)
    y_guess = y0 + tau * (yf - y0)

    # Estimate velocity from distance and time
    distance = np.sqrt((xf - x0)**2 + (yf - y0)**2)
    v_avg = distance / t_duration
    v_guess = np.linspace(0.1, v_avg * 1.5, num_disc)

    # Constant control (45 degrees as initial guess)
    theta_guess = np.ones(num_all) * np.pi / 4

    design_vars = {
        'states:x': jnp.array(x_guess),
        'states:y': jnp.array(y_guess),
        'states:v': jnp.array(v_guess),
        'controls:theta': jnp.array(theta_guess),
        't_initial': 0.0,
        't_duration': t_duration,
        'x_initial': x0,
        'y_initial': y0,
        'v_initial': 0.0,
        'x_final': xf,
        'y_final': yf,
    }

    return design_vars


def print_separator(title):
    """Print a formatted section separator."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    """
    Main demonstration function.

    Sets up the brachistochrone problem, evaluates the phase function,
    and demonstrates gradient computation with JAX.
    """
    print_separator("Radau Brachistochrone Phase - Runnable Example")

    # ========================================================================
    # 1. Problem Setup
    # ========================================================================
    print("\n1. Creating grid data...")
    num_segments = 3
    order = 3
    grid_data = create_radau_grid_data(num_segments=num_segments, order=order)

    print(f"   Number of segments: {num_segments}")
    print(f"   Polynomial order: {order}")
    print(f"   Discretization nodes: {grid_data['num_disc_nodes']}")
    print(f"   Collocation nodes: {grid_data['num_col_nodes']}")
    print(f"   Total nodes: {grid_data['num_all_nodes']}")

    # ========================================================================
    # 2. Initial Guess
    # ========================================================================
    print("\n2. Creating initial guess...")
    design_vars = create_reasonable_guess(
        grid_data,
        x0=0.0, xf=10.0,
        y0=10.0, yf=5.0,
        t_duration=1.8
    )

    print(f"   Initial position: ({design_vars['x_initial']:.1f}, {design_vars['y_initial']:.1f})")
    print(f"   Final position: ({design_vars['x_final']:.1f}, {design_vars['y_final']:.1f})")
    print(f"   Initial velocity: {design_vars['v_initial']:.1f}")
    print(f"   Phase duration guess: {design_vars['t_duration']:.2f} s")

    # ========================================================================
    # 3. Problem Options
    # ========================================================================
    print("\n3. Configuring problem options...")
    options = {
        'g': 9.80665,  # Gravitational acceleration (m/s^2)
        'enforce_initial': {'x': True, 'y': True, 'v': True},
        'enforce_final': {'x': True, 'y': True},
    }

    print(f"   Gravity: {options['g']:.5f} m/s^2")
    print(f"   Enforcing initial: {list(options['enforce_initial'].keys())}")
    print(f"   Enforcing final: {list(options['enforce_final'].keys())}")

    # ========================================================================
    # 4. Evaluate Phase Function
    # ========================================================================
    print_separator("Phase Function Evaluation")

    print("\nEvaluating radau_brachistochrone_phase()...")
    residuals = radau_brachistochrone_phase(design_vars, grid_data, options)

    print("\nResiduals computed successfully! [OK]")
    print("\nResidual Summary:")
    print(f"  Objective (time to minimize): {residuals['objective']:.6f} s")
    print(f"\n  Defect norms:")
    print(f"    ||defect:x|| = {jnp.linalg.norm(residuals['defect:x']):.6e}")
    print(f"    ||defect:y|| = {jnp.linalg.norm(residuals['defect:y']):.6e}")
    print(f"    ||defect:v|| = {jnp.linalg.norm(residuals['defect:v']):.6e}")

    # Check for continuity defects
    if residuals['continuity:x'].size > 0:
        print(f"\n  Continuity defect norms:")
        print(f"    ||continuity:x|| = {jnp.linalg.norm(residuals['continuity:x']):.6e}")
        print(f"    ||continuity:y|| = {jnp.linalg.norm(residuals['continuity:y']):.6e}")
        print(f"    ||continuity:v|| = {jnp.linalg.norm(residuals['continuity:v']):.6e}")

    # Boundary constraint residuals
    print(f"\n  Boundary constraint residuals:")
    if 'initial:x' in residuals:
        print(f"    initial:x = {residuals['initial:x']:.6e}")
    if 'initial:y' in residuals:
        print(f"    initial:y = {residuals['initial:y']:.6e}")
    if 'initial:v' in residuals:
        print(f"    initial:v = {residuals['initial:v']:.6e}")
    if 'final:x' in residuals:
        print(f"    final:x = {residuals['final:x']:.6e}")
    if 'final:y' in residuals:
        print(f"    final:y = {residuals['final:y']:.6e}")

    # ========================================================================
    # 5. Demonstrate JAX Automatic Differentiation
    # ========================================================================
    print_separator("JAX Automatic Differentiation Demo")

    print("\nComputing gradients using JAX autodiff...")

    # Gradient of objective with respect to t_duration
    def objective_fn(t_dur):
        """Extract objective as function of duration."""
        dv = {**design_vars, 't_duration': t_dur}
        res = radau_brachistochrone_phase(dv, grid_data, options)
        return res['objective']

    grad_obj_t = jax.grad(objective_fn)(design_vars['t_duration'])
    print(f"\n  d(objective)/d(t_duration) = {grad_obj_t:.6f}")
    print(f"    (Should be ~1.0 since objective = t_duration)")

    # Gradient of x defect norm with respect to initial x position
    def defect_x_norm_fn(x_init):
        """Extract defect:x norm as function of initial x."""
        dv = design_vars.copy()
        # Update initial x state value
        x_vals = dv['states:x']
        x_vals_new = x_vals.at[0].set(x_init)
        dv['states:x'] = x_vals_new
        dv['x_initial'] = x_init
        res = radau_brachistochrone_phase(dv, grid_data, options)
        return jnp.linalg.norm(res['defect:x'])

    grad_defect_x = jax.grad(defect_x_norm_fn)(design_vars['states:x'][0])
    print(f"\n  d(||defect:x||)/d(x[0]) = {grad_defect_x:.6e}")
    print(f"    (Shows sensitivity of dynamics to initial state)")

    # ========================================================================
    # 6. Note on JIT Compilation
    # ========================================================================
    print_separator("Note on JIT Compilation")

    print("""
The phase function can be JIT-compiled for improved performance, but requires
fixed options (which residuals to enforce) since conditional logic must be
static during compilation.

For production use, you would:
  1. Fix the 'options' dict structure
  2. JIT-compile the phase function once
  3. Reuse the compiled function for many evaluations

This approach is ideal for optimization loops where the same constraints
are evaluated thousands of times with different design variable values.
""")

    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Summary")

    print("""
[SUCCESS] Successfully demonstrated:
  - Grid data creation for Radau pseudospectral method
  - Initial guess generation for design variables
  - Phase function evaluation (computes residuals)
  - JAX automatic differentiation (gradient computation)

[INFO] What the residuals mean:
  - defect:x, defect:y, defect:v: Violations of ODE dynamics at collocation nodes
    (Should be ~0 for a valid solution)
  - continuity:*: State continuity across segment boundaries
    (Should be ~0 for smooth trajectory)
  - initial:*, final:*: Boundary condition violations
    (Should be ~0 to satisfy constraints)
  - objective: The quantity to minimize (time in this case)

[NEXT STEPS] To solve the optimization problem:
  - Use scipy.optimize.minimize or JAXopt to solve the optimization problem
  - Minimize objective subject to residuals = 0
  - Extract optimal trajectory from solution
  - Plot results to visualize the brachistochrone curve

This phase function is ready to be used with an optimizer!
""")


if __name__ == '__main__':
    main()
