# Dymos JAX Functions

This directory contains **pure JAX function** implementations of Dymos component computations. These are standalone mathematical functions that can be used directly with JAX's autodiff, JIT compilation, and other transformations.

## Architecture: Pure Functions, Not Classes

This implementation uses **pure functions** rather than class-based components:
- Standalone `def function_name(...)` functions
- No classes, no state, no OpenMDAO dependencies
- Pure mathematical operations that return outputs
- Maximum composability and flexibility

## Benefits

- **Automatic Differentiation:** Use `jax.grad`, `jax.jacobian` directly on functions
- **Performance:** Apply `jax.jit` and `jax.vmap` at the appropriate level for your use case
- **GPU Support:** Functions run on GPU without modification
- **Composability:** Pure functions easily combined, transformed, and optimized together
- **Flexibility:** Users control where to apply JIT compilation (not forced by library)

## Current Limitations

- **No units:** This version ignores physical units
- **Explicit components only:** Implicit components not yet supported
- **No Groups:** Only leaf computational components converted

## Usage

Import and use as pure JAX functions:

```python
import jax
import jax.numpy as jnp
from dymos.jax.common.time import time
from dymos.jax.pseudospectral.components.collocation import collocation_defect

# Direct usage
node_ptau = jnp.linspace(-1, 1, 10)
node_dptau_dstau = jnp.ones(10)
t, t_phase, dt_dstau = time(0.0, 10.0, node_ptau, node_dptau_dstau)

# Apply JIT at your chosen level (individual function or composite)
@jax.jit
def my_phase_computation(t_initial, t_duration):
    t, t_phase, dt_dstau = time(t_initial, t_duration, node_ptau, node_dptau_dstau)
    # JAX optimizes across function boundaries
    return jnp.sum(t_phase)

# Automatic differentiation
time_grad = jax.grad(lambda t_dur: jnp.sum(time(0.0, t_dur, node_ptau, node_dptau_dstau)[0]))
gradient = time_grad(10.0)

# Vectorization with vmap
batched_time = jax.vmap(
    lambda ti, td: time(ti, td, node_ptau, node_dptau_dstau),
    in_axes=(0, 0)
)
```

## Completed Functions

### Common Components
- **common/time.py**: `time(t_initial, t_duration, node_ptau, node_dptau_dstau)`
  Returns: `(t, t_phase, dt_dstau)`

### Explicit Shooting Components
- **explicit_shooting/tau.py**: `tau(t, t_initial, t_duration, ptau0_seg, ptauf_seg)`
  Returns: `(ptau, stau, dstau_dt, t_phase)`

### Pseudospectral Components
- **pseudospectral/components/collocation.py**: `collocation_defect(f_approx, f_computed, dt_dstau)`
  Returns: `defect`

- **pseudospectral/components/state_interp.py**:
  - `state_interp_radau(xd, dt_dstau, Ad)` → `xdotc`
  - `state_interp_gauss_lobatto(xd, fd, dt_dstau, Ai, Bi, Ad, Bd)` → `(xc, xdotc)`

- **pseudospectral/components/control_endpoint_defect.py**:
  `control_endpoint_defect(u_all, L, col_indices, num_disc_end_segment)`
  Returns: `defect`

- **pseudospectral/components/gauss_lobatto_interleave.py**:
  `gauss_lobatto_interleave(disc_values, col_values, disc_indices, col_indices, num_nodes)`
  Returns: `all_values`

- **pseudospectral/components/radau_defect.py**:
  `radau_defect(x, f_ode, dt_dstau, D, x_initial, x_final, segment_end_indices)`
  Returns: `(rate_defect, initial_defect, final_defect, continuity_defect)`

- **pseudospectral/components/birkhoff_defect.py**:
  `birkhoff_defect(X, V, f_computed, dt_dstau, A, C, xv_indices, x_initial, x_final, num_segments)`
  Returns: `(state_defect, state_rate_defect, continuity_defect)`

### Explicit Shooting Control Interpolation Components

- **explicit_shooting/vandermonde_control_interp.py**:
  `vandermonde_control_interp(u_input, stau, dstau_dt, input_to_disc_map, disc_node_indices, V_hat_inv)`
  Returns: `(u, u_dot, u_ddot)`

- **explicit_shooting/cubic_spline_control_interp.py**:
  `cubic_spline_control_interp(u_input, ptau, ptau_grid, input_node_indices, t_duration)`
  Returns: `(u, u_dot, u_ddot)`

- **explicit_shooting/barycentric_control_interp.py**:
  `barycentric_control_interp(u_input, stau, dstau_dt, input_to_disc_map, disc_node_indices, taus_seg, w_b)`
  Returns: `(u, u_dot, u_ddot)`

## Component Status

See [COMPONENT_INVENTORY.md](COMPONENT_INVENTORY.md) for full list and conversion status.

## Testing

Each function has tests in `tests/` comparing outputs and JAX-computed derivatives against the original OpenMDAO component.

Run tests with the pixi environment:
```bash
cd /path/to/dymos.git
pixi run -e dev testflo dymos/jax/tests/ -v
```

## Pure Function Pattern

All JAX functions follow this pattern:

```python
"""JAX function for [computation name]."""
import jax.numpy as jnp


def function_name(input1, input2, static_param):
    """
    Compute [description].

    Parameters
    ----------
    input1 : ArrayLike
        Description. Shape: (num_nodes, *shape)
    input2 : ArrayLike
        Description. Shape: (num_nodes, *shape)
    static_param : type
        Description (constants, matrices, etc.)

    Returns
    -------
    output1 : ArrayLike
        Description. Shape: (num_nodes, *shape)
    output2 : ArrayLike
        Description. Shape: (num_nodes, *shape)

    Notes
    -----
    Pure function suitable for jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - inputs/outputs in consistent units.
    """
    # Pure mathematical computation using jnp
    output1 = jnp.some_operation(input1, static_param)
    output2 = jnp.another_operation(input2)

    return output1, output2
```

**Key principles:**
1. **Pure functions only:** No classes, no state
2. **No decorators:** Keep functions pure, let users apply JIT
3. **JAX NumPy:** Use `jax.numpy` for array operations
4. **Clear signatures:** All inputs as arguments, return outputs as tuple
5. **Ignore units:** Assume consistent units chosen by caller

## Contributing

When adding new JAX functions:
1. Create pure function following the pattern above
2. Import `jax.numpy as jnp` (use sparingly - only when needed)
3. Add comprehensive tests (outputs, gradients, JIT, vmap)
4. Update COMPONENT_INVENTORY.md with ✅ status
5. Document parameter shapes and return values clearly
