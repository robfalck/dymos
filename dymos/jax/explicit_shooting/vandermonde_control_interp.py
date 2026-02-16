"""JAX function for Vandermonde control interpolation in explicit shooting."""
import jax.numpy as jnp


def vandermonde_control_interp(u_input, stau, dstau_dt, input_to_disc_map,
                               disc_node_indices, V_hat_inv):
    """
    Interpolate control values using Vandermonde (polynomial) interpolation.

    This function takes control values at input nodes, broadcasts them to discretization
    nodes via a mapping, and then uses polynomial interpolation to evaluate the control
    and its first two derivatives at arbitrary segment tau values.

    Parameters
    ----------
    u_input : ArrayLike
        Control values at input nodes.
        Shape: (num_input_nodes, *control_shape)
    stau : ArrayLike
        Segment tau values where control should be evaluated (in [-1, 1]).
        Shape: (vec_size,)
    dstau_dt : float or ArrayLike
        Derivative of segment tau with respect to time.
        Scalar or shape: (vec_size,)
    input_to_disc_map : ArrayLike
        Mapping from input nodes to discretization nodes (indices).
        Shape: (num_disc_nodes,)
    disc_node_indices : ArrayLike
        Indices of discretization nodes within the current segment.
        Shape: (num_disc_nodes_in_segment,)
    V_hat_inv : ArrayLike
        Inverse of the Vandermonde matrix at discretization nodes.
        Shape: (order, order) where order is the segment transcription order.

    Returns
    -------
    u : ArrayLike
        Interpolated control values at stau.
        Shape: (vec_size, *control_shape)
    u_dot : ArrayLike
        First derivative of control with respect to time at stau.
        Shape: (vec_size, *control_shape)
    u_ddot : ArrayLike
        Second derivative of control with respect to time at stau.
        Shape: (vec_size, *control_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The interpolation process:
    1. Map input node values to discretization nodes: u_disc = u_input[map]
    2. Extract values at segment nodes: u_seg = u_disc[segment_indices]
    3. Compute polynomial coefficients: a = V_hat_inv @ u_seg
    4. Evaluate polynomial at stau using Vandermonde matrix: u = V(stau) @ a
    5. Evaluate derivatives using derivative of Vandermonde matrix

    The Vandermonde matrix V(tau) for n points is:
        V[i, j] = tau[i]^j for j=0..n-1 (increasing powers)
    """
    vec_size = len(stau)
    control_shape = u_input.shape[1:]
    order = V_hat_inv.shape[0]  # Polynomial order (e.g., 3 for quadratic)

    # Step 1: Map input nodes to discretization nodes
    # Create identity-like mapping matrix
    num_disc_total = len(input_to_disc_map)
    num_input_total = u_input.shape[0]

    # Map: for each discretization node, which input node does it come from?
    # This is essentially u_disc[i] = u_input[input_to_disc_map[i]]
    u_disc = u_input[input_to_disc_map]

    # Step 2: Extract values at nodes in current segment
    u_seg = u_disc[disc_node_indices]

    # Flatten control for matrix operations
    u_seg_flat = jnp.reshape(u_seg, (order, -1))

    # Step 3: Compute polynomial coefficients
    # a = V_hat_inv @ u_seg
    a = V_hat_inv @ u_seg_flat

    # Step 4: Build Vandermonde matrices at evaluation points
    # V(stau) = [1, stau, stau^2, ..., stau^(n-1)]
    V_stau = jnp.vander(stau, N=order, increasing=True)

    # Step 5: Build derivative matrices
    # dV/dstau: shift and multiply by powers
    dV_stau = jnp.zeros_like(V_stau)
    dV_stau = dV_stau.at[:, 1:].set(V_stau[:, :-1])
    fac = jnp.arange(order, dtype=float)
    dV_stau = dV_stau * fac[jnp.newaxis, :]

    # d2V/dstau2: shift twice and multiply by powers
    dV2_stau = jnp.zeros_like(V_stau)
    dV2_stau = dV2_stau.at[:, 2:].set(V_stau[:, :-2])
    fac2 = fac[:-1]
    dV2_stau = dV2_stau.at[:, 1:].multiply(fac2[jnp.newaxis, :] * fac[jnp.newaxis, 1:])

    # Step 6: Evaluate polynomial and derivatives
    # u = V @ a
    u_flat = V_stau @ a
    u_dot_flat = dV_stau @ a
    u_ddot_flat = dV2_stau @ a

    # Scale derivatives by dstau_dt
    # Make dstau_dt broadcastable if it's a scalar
    if jnp.ndim(dstau_dt) == 0:
        dstau_dt_exp = dstau_dt
        dstau_dt2 = dstau_dt ** 2
    else:
        dstau_dt_exp = dstau_dt[:, jnp.newaxis]
        dstau_dt2 = dstau_dt_exp ** 2

    u_dot_flat = u_dot_flat * dstau_dt_exp
    u_ddot_flat = u_ddot_flat * dstau_dt2

    # Step 7: Reshape back to control shape
    u = jnp.reshape(u_flat, (vec_size,) + control_shape)
    u_dot = jnp.reshape(u_dot_flat, (vec_size,) + control_shape)
    u_ddot = jnp.reshape(u_ddot_flat, (vec_size,) + control_shape)

    return u, u_dot, u_ddot
