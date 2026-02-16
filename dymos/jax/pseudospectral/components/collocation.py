"""JAX function for collocation defect computation."""
import jax.numpy as jnp


def collocation_defect(f_approx, f_computed, dt_dstau):
    """
    Compute the collocation defect for a state.

    The defect is the difference between the interpolated state derivative
    and the computed state derivative, scaled by dt/dstau.

    Parameters
    ----------
    f_approx : ArrayLike
        Approximated (interpolated) state derivative at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    f_computed : ArrayLike
        Computed state derivative from ODE at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at each node.
        Shape: (num_col_nodes,)

    Returns
    -------
    defect : ArrayLike
        Collocation defect at each node.
        Shape: (num_col_nodes, *state_shape)
    """
    # Compute difference between approximated and computed derivatives
    diff = f_approx - f_computed

    # Scale by dt_dstau, broadcasting across state dimensions
    # Reshape dt_dstau to (num_col_nodes, 1, 1, ...) for proper broadcasting
    dt_dstau_expanded = jnp.reshape(dt_dstau, (dt_dstau.shape[0],) + (1,) * (diff.ndim - 1))

    defect = diff * dt_dstau_expanded

    return defect
