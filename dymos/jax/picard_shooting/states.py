"""JAX function for Picard states pass-through."""
import jax.numpy as jnp


def states_passthrough(states):
    """
    Pass-through function for states during Picard iteration.

    This function serves as an identity operation, accepting state values and
    returning them unchanged. Used in Picard iteration with NonlinearBlockGS
    to provide the current state values to the ODE evaluation.

    Parameters
    ----------
    states : ArrayLike
        State values at all nodes.
        Shape: (num_nodes, *state_shape)

    Returns
    -------
    state_val : ArrayLike
        Same as input states (identity operation).
        Shape: (num_nodes, *state_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    In the OpenMDAO implementation, this component accepts 'states:{name}' and
    outputs 'state_val:{name}' to enable NonlinearBlockGS convergence. In JAX,
    this is simply an identity function and the iteration logic is handled
    externally (e.g., with jax.lax.custom_root).
    """
    # Identity operation - just return the input
    return states
