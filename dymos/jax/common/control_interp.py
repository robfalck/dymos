"""JAX functions for control interpolation."""
import jax.numpy as jnp


def control_interp_polynomial(u_input, L, D, D2, t_duration):
    """
    Interpolate polynomial control values and derivatives.

    Uses Lagrange interpolation to compute control values, first derivatives,
    and second derivatives at output nodes from control values at LGL nodes.

    Parameters
    ----------
    u_input : ArrayLike
        Control values at polynomial discretization nodes (LGL).
        Shape: (order + 1, *control_shape)
    L : ArrayLike
        Lagrange interpolation matrix from disc nodes to output nodes.
        Shape: (num_output_nodes, order + 1)
    D : ArrayLike
        Lagrange differentiation matrix (first derivative).
        Shape: (num_output_nodes, order + 1)
    D2 : ArrayLike
        Lagrange second differentiation matrix.
        Shape: (num_output_nodes, order + 1)
    t_duration : float
        Duration of the phase (for scaling derivatives).

    Returns
    -------
    val : ArrayLike
        Interpolated control values at output nodes.
        Shape: (num_output_nodes, *control_shape)
    rate : ArrayLike
        First time derivative of control at output nodes.
        Shape: (num_output_nodes, *control_shape)
    rate2 : ArrayLike
        Second time derivative of control at output nodes.
        Shape: (num_output_nodes, *control_shape)
    boundary_val : ArrayLike
        Control values at first and last nodes.
        Shape: (2, *control_shape)
    boundary_rate : ArrayLike
        First derivative at first and last nodes.
        Shape: (2, *control_shape)
    boundary_rate2 : ArrayLike
        Second derivative at first and last nodes.
        Shape: (2, *control_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    For polynomial controls, the phase is represented in normalized tau space [-1, 1],
    so dt_dtau = t_duration / 2 (constant across all nodes).
    """
    num_input_nodes = u_input.shape[0]
    control_shape = u_input.shape[1:]
    num_output_nodes = L.shape[0]

    # Flatten control dimensions for matrix multiplication
    u_flat = jnp.reshape(u_input, (num_input_nodes, -1))

    # dt_dtau = t_duration / 2 for polynomial controls (normalized tau space)
    dt_dtau = 0.5 * t_duration

    # Interpolate values
    val_flat = L @ u_flat
    val = jnp.reshape(val_flat, (num_output_nodes,) + control_shape)

    # Compute first derivative
    rate_flat = (D @ u_flat) / dt_dtau
    rate = jnp.reshape(rate_flat, (num_output_nodes,) + control_shape)

    # Compute second derivative
    rate2_flat = (D2 @ u_flat) / (dt_dtau ** 2)
    rate2 = jnp.reshape(rate2_flat, (num_output_nodes,) + control_shape)

    # Extract boundary values (first and last nodes)
    boundary_val = jnp.array([val[0], val[-1]])
    boundary_rate = jnp.array([rate[0], rate[-1]])
    boundary_rate2 = jnp.array([rate2[0], rate2[-1]])

    return val, rate, rate2, boundary_val, boundary_rate, boundary_rate2


def control_interp_full(u_input, L, D, D2, dt_dstau, S=None):
    """
    Interpolate full control values and derivatives.

    Uses Lagrange interpolation to compute control values, first derivatives,
    and second derivatives at output nodes from control values at input nodes.
    Optionally computes continuity defects at segment boundaries.

    Parameters
    ----------
    u_input : ArrayLike
        Control values at control input nodes.
        Shape: (num_input_nodes, *control_shape)
    L : ArrayLike
        Lagrange interpolation matrix from input nodes to output nodes.
        Shape: (num_output_nodes, num_input_nodes)
    D : ArrayLike
        Lagrange differentiation matrix (first derivative).
        Shape: (num_output_nodes, num_input_nodes)
    D2 : ArrayLike
        Lagrange second differentiation matrix.
        Shape: (num_output_nodes, num_input_nodes)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at each output node.
        Shape: (num_output_nodes,)
    S : ArrayLike, optional
        Selection matrix for computing continuity defects at segment boundaries.
        Shape: (num_segments - 1, num_output_nodes)
        If None, continuity defects are not computed.

    Returns
    -------
    val : ArrayLike
        Interpolated control values at output nodes.
        Shape: (num_output_nodes, *control_shape)
    rate : ArrayLike
        First time derivative of control at output nodes.
        Shape: (num_output_nodes, *control_shape)
    rate2 : ArrayLike
        Second time derivative of control at output nodes.
        Shape: (num_output_nodes, *control_shape)
    boundary_val : ArrayLike
        Control values at first and last nodes.
        Shape: (2, *control_shape)
    boundary_rate : ArrayLike
        First derivative at first and last nodes.
        Shape: (2, *control_shape)
    boundary_rate2 : ArrayLike
        Second derivative at first and last nodes.
        Shape: (2, *control_shape)
    val_cnty_defect : ArrayLike or None
        Continuity defects for values at segment boundaries.
        Shape: (num_segments - 1, *control_shape)
        None if S is None.
    rate_cnty_defect : ArrayLike or None
        Continuity defects for rates at segment boundaries.
        Shape: (num_segments - 1, *control_shape)
        None if S is None.
    rate2_cnty_defect : ArrayLike or None
        Continuity defects for second derivatives at segment boundaries.
        Shape: (num_segments - 1, *control_shape)
        None if S is None.

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The continuity defect at segment boundary i is computed as:
        defect[i] = val[seg_start[i+1]] - val[seg_end[i]]
    This is implemented via the selection matrix S which has +1 for segment starts
    and -1 for segment ends.
    """
    num_input_nodes = u_input.shape[0]
    control_shape = u_input.shape[1:]
    num_output_nodes = L.shape[0]

    # Flatten control dimensions for matrix multiplication
    u_flat = jnp.reshape(u_input, (num_input_nodes, -1))

    # Interpolate values
    val_flat = L @ u_flat
    val = jnp.reshape(val_flat, (num_output_nodes,) + control_shape)

    # Compute first derivative
    # D @ u gives derivative wrt segment tau, divide by dt_dstau to get d/dt
    dt_dstau_expanded = jnp.reshape(dt_dstau, (num_output_nodes,) + (1,) * len(control_shape))
    rate_flat = (D @ u_flat) / dt_dstau[:, jnp.newaxis]
    rate = jnp.reshape(rate_flat, (num_output_nodes,) + control_shape)

    # Compute second derivative
    rate2_flat = (D2 @ u_flat) / (dt_dstau[:, jnp.newaxis] ** 2)
    rate2 = jnp.reshape(rate2_flat, (num_output_nodes,) + control_shape)

    # Extract boundary values (first and last nodes)
    boundary_val = jnp.array([val[0], val[-1]])
    boundary_rate = jnp.array([rate[0], rate[-1]])
    boundary_rate2 = jnp.array([rate2[0], rate2[-1]])

    # Compute continuity defects if selection matrix provided
    if S is not None:
        val_cnty_defect_flat = S @ val_flat
        val_cnty_defect = jnp.reshape(val_cnty_defect_flat,
                                     (S.shape[0],) + control_shape)

        rate_cnty_defect_flat = S @ rate_flat
        rate_cnty_defect = jnp.reshape(rate_cnty_defect_flat,
                                      (S.shape[0],) + control_shape)

        rate2_cnty_defect_flat = S @ rate2_flat
        rate2_cnty_defect = jnp.reshape(rate2_cnty_defect_flat,
                                       (S.shape[0],) + control_shape)
    else:
        val_cnty_defect = None
        rate_cnty_defect = None
        rate2_cnty_defect = None

    return (val, rate, rate2,
            boundary_val, boundary_rate, boundary_rate2,
            val_cnty_defect, rate_cnty_defect, rate2_cnty_defect)
