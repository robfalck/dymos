"""Spec-based brachistochrone example using Pydantic specs and JAX backend.

This module demonstrates the composable architecture for optimal control:
1. Define problem structure with PhaseSpec
2. Define physics with a pure JAX ODE function
3. Use the JAX backend to create a fast, JIT-compilable phase function
4. Differentiate and JIT compile using standard JAX tools

Run this module directly to see the example output:
    python -m dymos.jax.examples.spec_based_brachistochrone
"""
import json

import jax
import jax.numpy as jnp

from dymos.jax.specs import (
    PhaseSpec, StateSpec, ControlSpec, ParameterSpec, TimeSpec, GridSpec
)
from dymos.jax.backend import create_jax_radau_phase
from dymos.jax.examples.brachistochrone_ode_dict import brachistochrone_ode_dict
from dymos.jax.examples.radau_brachistochrone_phase import (
    radau_brachistochrone_phase, create_radau_grid_data
)


def build_brachistochrone_spec(num_segments=3, order=3):
    """Build PhaseSpec for the brachistochrone problem.

    Parameters
    ----------
    num_segments : int
        Number of grid segments.
    order : int
        Polynomial order per segment.

    Returns
    -------
    spec : PhaseSpec
        Complete brachistochrone problem specification.
    """
    spec = PhaseSpec(
        states=[
            StateSpec(name='x', fix_initial=True, fix_final=True),
            StateSpec(name='y', fix_initial=True, fix_final=True),
            StateSpec(name='v', fix_initial=True),
        ],
        controls=[
            ControlSpec(name='theta'),
        ],
        parameters=[
            ParameterSpec(name='g', value=9.80665),
        ],
        time=TimeSpec(
            fix_initial=True,
            fix_duration=False,
            initial_value=0.0,
            duration_bounds=(0.5, 10.0),
        ),
        grid=GridSpec(num_segments=num_segments, order=order, transcription='radau'),
    )
    return spec


def build_design_vars(grid_data, t_initial=0.0, t_duration=1.8):
    """Build design variables dict for the spec-based phase function.

    Parameters
    ----------
    grid_data : dict
        Grid data from create_radau_grid_data.
    t_initial : float
        Initial time.
    t_duration : float
        Phase duration.

    Returns
    -------
    design_vars : dict
        Dict with 'states', 'controls', 'time', and boundary targets.
    """
    num_disc_nodes = grid_data['num_disc_nodes']
    num_all_nodes = grid_data['num_all_nodes']

    design_vars = {
        'states': {
            'x': jnp.linspace(0.0, 10.0, num_disc_nodes),
            'y': jnp.linspace(0.0, -10.0, num_disc_nodes),
            'v': jnp.linspace(0.0, 14.0, num_disc_nodes),
        },
        'controls': {
            'theta': jnp.ones(num_all_nodes) * jnp.pi / 4,
        },
        'time': {
            'initial': t_initial,
            'duration': t_duration,
        },
        # Boundary targets (used when fix_initial/fix_final=True)
        'x_initial': 0.0,
        'y_initial': 0.0,
        'v_initial': 0.0,
        'x_final': 10.0,
        'y_final': -10.0,
    }
    return design_vars


def build_legacy_design_vars(grid_data, t_initial=0.0, t_duration=1.8):
    """Build design variables in the legacy (flat) format for comparison.

    Parameters
    ----------
    grid_data : dict
        Grid data from create_radau_grid_data.
    t_initial : float
        Initial time.
    t_duration : float
        Phase duration.

    Returns
    -------
    design_vars : dict
        Flat dict in the format used by radau_brachistochrone_phase.
    """
    num_disc_nodes = grid_data['num_disc_nodes']
    num_all_nodes = grid_data['num_all_nodes']

    return {
        'states:x': jnp.linspace(0.0, 10.0, num_disc_nodes),
        'states:y': jnp.linspace(0.0, -10.0, num_disc_nodes),
        'states:v': jnp.linspace(0.0, 14.0, num_disc_nodes),
        'controls:theta': jnp.ones(num_all_nodes) * jnp.pi / 4,
        't_initial': t_initial,
        't_duration': t_duration,
        'x_initial': 0.0,
        'y_initial': 0.0,
        'v_initial': 0.0,
        'x_final': 10.0,
        'y_final': -10.0,
    }


def main():
    """Run the spec-based brachistochrone example."""
    print("=" * 60)
    print("Spec-Based Brachistochrone Example")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 1. Create PhaseSpec (static problem configuration)
    # -------------------------------------------------------------------
    print("\n1. Creating PhaseSpec...")
    spec = build_brachistochrone_spec(num_segments=2, order=3)
    print(f"   States: {spec.get_state_names()}")
    print(f"   Controls: {spec.get_control_names()}")
    print(f"   Parameters: {spec.get_parameter_names()}")
    print(f"   Grid: {spec.grid.num_segments} segments, order {spec.grid.order}")

    # -------------------------------------------------------------------
    # 2. Create phase function from spec (JAX backend)
    # -------------------------------------------------------------------
    print("\n2. Creating phase function from spec (JAX backend)...")
    phase_fn = create_jax_radau_phase(spec, ode_function=brachistochrone_ode_dict)
    print("   Phase function created successfully.")

    # -------------------------------------------------------------------
    # 3. Evaluate residuals
    # -------------------------------------------------------------------
    print("\n3. Evaluating residuals...")
    grid_data = create_radau_grid_data(spec.grid.num_segments, spec.grid.order)
    design_vars = build_design_vars(grid_data)

    residuals = phase_fn(design_vars)
    print(f"   Residual keys: {list(residuals.keys())}")
    print(f"   defect:x shape: {residuals['defect:x'].shape}")
    print(f"   objective (t_duration): {residuals['objective']:.4f}")

    # -------------------------------------------------------------------
    # 4. Compare with legacy specialized version
    # -------------------------------------------------------------------
    print("\n4. Comparing with legacy radau_brachistochrone_phase...")
    legacy_dv = build_legacy_design_vars(grid_data)
    legacy_options = {
        'g': 9.80665,
        'enforce_initial': {'x': True, 'y': True, 'v': True},
        'enforce_final': {'x': True, 'y': True},
    }
    legacy_residuals = radau_brachistochrone_phase(legacy_dv, grid_data, legacy_options)

    for state_name in ['x', 'y', 'v']:
        key = f'defect:{state_name}'
        max_diff = float(jnp.max(jnp.abs(residuals[key] - legacy_residuals[key])))
        print(f"   Max diff in {key}: {max_diff:.2e}")

    # -------------------------------------------------------------------
    # 5. JIT compilation
    # -------------------------------------------------------------------
    print("\n5. JIT compilation...")
    phase_jitted = jax.jit(phase_fn)
    residuals_jit = phase_jitted(design_vars)
    print(f"   JIT compiled successfully. objective = {residuals_jit['objective']:.4f}")

    # -------------------------------------------------------------------
    # 6. Gradient computation
    # -------------------------------------------------------------------
    print("\n6. Gradient computation (dict-based)...")

    def scalar_objective(dv):
        return phase_fn(dv)['objective']

    grad_fn = jax.grad(scalar_objective)
    grads = grad_fn(design_vars)
    print(f"   Gradient of objective wrt t_duration: "
          f"{grads['time']['duration']:.4f}  (expected: 1.0)")
    print(f"   Gradient structure matches design_vars: "
          f"{set(grads.keys()) == set(design_vars.keys())}")

    # -------------------------------------------------------------------
    # 7. Spec serialization
    # -------------------------------------------------------------------
    print("\n7. Spec serialization (JSON round-trip)...")
    spec_dict = spec.model_dump()
    spec_json = json.dumps(spec_dict, indent=2)
    spec_loaded = PhaseSpec(**json.loads(spec_json))
    phase_fn_loaded = create_jax_radau_phase(spec_loaded, brachistochrone_ode_dict)
    residuals_loaded = phase_fn_loaded(design_vars)
    match = jnp.allclose(residuals['objective'], residuals_loaded['objective'])
    print(f"   Loaded spec gives same objective: {bool(match)}")

    print("\n" + "=" * 60)
    print("Example completed successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()
