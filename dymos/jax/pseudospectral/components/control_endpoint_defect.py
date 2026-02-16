"""JAX function for control endpoint defect computation in Radau pseudospectral."""
import jax.numpy as jnp


def control_endpoint_defect(u_all, L, col_indices, num_disc_end_segment):
    """
    Compute the control endpoint defect for Radau pseudospectral transcription.

    In Radau pseudospectral methods, dynamic control values are specified at collocation
    nodes. This function computes the defect between the actual control value at the
    phase endpoint and the value obtained by Lagrange interpolation from the collocation
    nodes in the final segment. The defect should be constrained to zero.

    Parameters
    ----------
    u_all : ArrayLike
        Control values at all nodes.
        Shape: (num_nodes, *control_shape)
    L : ArrayLike
        Last row of the Lagrange interpolation matrix (for the final segment).
        This is used to interpolate the endpoint value from collocation nodes.
        Shape: (num_disc_end_segment,) or (1, num_disc_end_segment)
    col_indices : ArrayLike
        Indices of collocation nodes within the full node array.
        Shape: (num_col_nodes,)
    num_disc_end_segment : int
        Number of discretization nodes in the final segment.

    Returns
    -------
    defect : ArrayLike
        Control endpoint defect (actual - interpolated).
        Shape: (1, *control_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The mathematical operation is:
        defect = u_endpoint - L @ u_col_end_segment
    where u_endpoint is the actual control at the last node and u_col_end_segment
    are the control values at the collocation nodes in the final segment.
    """
    # Extract control values at collocation nodes
    u_col = u_all[col_indices]

    # Get control values at the last num_disc_end_segment collocation nodes
    # (these are the collocation nodes in the final segment)
    u_col_end_segment = u_col[-num_disc_end_segment:]

    # Actual control value at the endpoint (last node in the phase)
    u_endpoint = u_all[-1:, ...]  # Keep first dimension for shape consistency

    # Ensure L is 2D for matrix multiplication
    L_2d = jnp.atleast_2d(L)

    # Compute interpolated endpoint value using Lagrange interpolation
    # L_2d has shape (1, num_disc_end_segment)
    # u_col_end_segment has shape (num_disc_end_segment, *control_shape)
    # Result should have shape (1, *control_shape)
    u_endpoint_interp = jnp.tensordot(L_2d, u_col_end_segment, axes=(1, 0))

    # Compute defect: actual - interpolated
    defect = u_endpoint - u_endpoint_interp

    return defect
