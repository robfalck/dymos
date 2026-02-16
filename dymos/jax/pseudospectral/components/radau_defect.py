"""JAX function for Radau pseudospectral defect computation."""
import jax.numpy as jnp


def radau_defect(x, f_ode, dt_dstau, D, x_initial, x_final, segment_end_indices=None):
    """
    Compute Radau pseudospectral defects for a state variable.

    Radau pseudospectral methods use Lagrange differentiation to approximate state
    derivatives at collocation nodes. This function computes four types of defects:

    1. Rate defect: difference between interpolated and computed derivatives
    2. Initial defect: difference between prescribed and interpolated initial values
    3. Final defect: difference between prescribed and interpolated final values
    4. Continuity defect: state value jumps at segment boundaries (multi-segment only)

    Parameters
    ----------
    x : ArrayLike
        State values at all discretization nodes.
        Shape: (num_disc_nodes, *state_shape)
    f_ode : ArrayLike
        Computed state derivative from ODE at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at collocation nodes.
        Shape: (num_col_nodes,)
    D : ArrayLike
        Lagrange differentiation matrix (dense or sparse converted to dense).
        Maps discretization node values to collocation node derivatives.
        Shape: (num_col_nodes, num_disc_nodes)
    x_initial : ArrayLike
        Prescribed initial state value.
        Shape: (1, *state_shape)
    x_final : ArrayLike
        Prescribed final state value.
        Shape: (1, *state_shape)
    segment_end_indices : ArrayLike, optional
        Indices of segment end nodes in the state array. Required for multi-segment
        problems to compute continuity defects. If None, no continuity defect computed.
        Shape: (num_segment_ends,)

    Returns
    -------
    rate_defect : ArrayLike
        Defect in state rate at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    initial_defect : ArrayLike
        Defect in initial state value.
        Shape: (1, *state_shape)
    final_defect : ArrayLike
        Defect in final state value.
        Shape: (1, *state_shape)
    continuity_defect : ArrayLike or None
        Defect in state continuity at segment boundaries.
        Shape: (num_segments - 1, *state_shape) if segment_end_indices provided, else None

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The rate defect is computed as:
        rate_defect = (D @ x) / dt_dstau - f_ode
    where the division by dt_dstau is broadcast across state dimensions.

    For sparse differentiation matrices, convert to dense before passing to this function:
        D_dense = D_sparse.toarray()
    """
    # Get dimensions
    num_disc_nodes = x.shape[0]
    state_shape = x.shape[1:]

    # Flatten state for matrix multiplication using -1 for automatic dimension inference
    x_flat = jnp.reshape(x, (num_disc_nodes, -1))

    # Compute approximated derivative using differentiation matrix
    # D @ x_flat gives derivatives at collocation nodes
    f_approx_flat = D @ x_flat

    # Reshape back to state shape
    num_col_nodes = D.shape[0]
    f_approx = jnp.reshape(f_approx_flat, (num_col_nodes,) + state_shape)

    # Compute rate defect: f_approx - f_ode * dt_dstau
    # Need to broadcast dt_dstau across state dimensions
    dt_dstau_expanded = jnp.reshape(dt_dstau, (num_col_nodes,) + (1,) * len(state_shape))
    rate_defect = f_approx - f_ode * dt_dstau_expanded

    # Compute initial value defect
    initial_defect = x_initial - x[0:1, ...]

    # Compute final value defect
    final_defect = x_final - x[-1:, ...]

    # Compute continuity defect (only for multi-segment problems)
    continuity_defect = None
    if segment_end_indices is not None:
        # Extract segment end indices (excluding first and last)
        # segment_end_indices[1:-1] gives interior segment boundaries
        # Even indices (2::2) are ends of segments
        # Odd indices (1:-1:2) are starts of next segments
        end_indices = segment_end_indices[2::2]
        start_indices = segment_end_indices[1:-1:2]

        # Continuity defect is difference between end of segment and start of next
        continuity_defect = x[start_indices, ...] - x[end_indices, ...]

    return rate_defect, initial_defect, final_defect, continuity_defect
