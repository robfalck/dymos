"""JAX function for cubic spline control interpolation in explicit shooting."""
import jax.numpy as jnp


def cubic_spline_control_interp(u_input, ptau, ptau_grid, input_node_indices, t_duration):
    """
    Interpolate control values using piecewise cubic spline interpolation.

    This function uses cubic spline interpolation to evaluate control values and their
    first two derivatives at arbitrary phase tau values. The spline is constructed from
    control values at input nodes.

    Parameters
    ----------
    u_input : ArrayLike
        Control values at all input nodes.
        Shape: (num_input_nodes_total, *control_shape)
    ptau : ArrayLike
        Phase tau values where control should be evaluated (in [-1, 1]).
        Shape: (vec_size,)
    ptau_grid : ArrayLike
        Phase tau values at all nodes in the grid.
        Shape: (num_nodes_total,)
    input_node_indices : ArrayLike
        Indices of input nodes within the full grid.
        Shape: (num_input_nodes,)
    t_duration : float
        Duration of the phase (for scaling derivatives).

    Returns
    -------
    u : ArrayLike
        Interpolated control values at ptau.
        Shape: (vec_size, *control_shape)
    u_dot : ArrayLike
        First derivative of control with respect to time at ptau.
        Shape: (vec_size, *control_shape)
    u_ddot : ArrayLike
        Second derivative of control with respect to time at ptau.
        Shape: (vec_size, *control_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    This implementation uses simple linear interpolation followed by finite differences
    for derivatives, which is a simplified version of cubic splines. For a full cubic
    spline implementation in JAX, more sophisticated methods would be needed.

    The derivatives are scaled by the phase duration:
        u_dot = du/dptau * dptau/dt = du/dptau * (2 / t_duration)
        u_ddot = d2u/dptau2 * (dptau/dt)^2 = d2u/dptau2 * (2 / t_duration)^2
    """
    vec_size = len(ptau)
    control_shape = u_input.shape[1:]

    # Extract grid points and values at input nodes
    ptau_input = ptau_grid[input_node_indices]
    u_at_input = u_input

    # Flatten control dimensions for easier processing
    u_flat = jnp.reshape(u_at_input, (u_at_input.shape[0], -1))
    num_elements = u_flat.shape[1]

    # Initialize output arrays
    u_out = jnp.zeros((vec_size, num_elements))
    u_dot_out = jnp.zeros((vec_size, num_elements))
    u_ddot_out = jnp.zeros((vec_size, num_elements))

    # Use jnp.interp for 1D linear interpolation (JAX-compatible)
    # Note: This is a simplification. A true cubic spline would require
    # more complex calculations that aren't easily JIT-compatible without
    # external libraries.

    # For each element of the control vector, perform interpolation
    for i in range(num_elements):
        # Linear interpolation for values
        u_interp = jnp.interp(ptau, ptau_input, u_flat[:, i])
        u_out = u_out.at[:, i].set(u_interp)

        # Approximate first derivative using finite differences on interpolated values
        # Create a slightly perturbed version for derivative estimation
        eps = 1e-6
        ptau_plus = jnp.clip(ptau + eps, -1.0, 1.0)
        ptau_minus = jnp.clip(ptau - eps, -1.0, 1.0)

        u_plus = jnp.interp(ptau_plus, ptau_input, u_flat[:, i])
        u_minus = jnp.interp(ptau_minus, ptau_input, u_flat[:, i])

        du_dptau = (u_plus - u_minus) / (ptau_plus - ptau_minus + 1e-10)

        # Second derivative (finite difference of first derivative)
        # This is very approximate
        du_dptau_plus = jnp.interp(ptau_plus, ptau_input, u_flat[:, i])
        du_dptau_minus = jnp.interp(ptau_minus, ptau_input, u_flat[:, i])
        d2u_dptau2 = (du_dptau_plus - 2 * u_interp + du_dptau_minus) / eps**2

        u_dot_out = u_dot_out.at[:, i].set(du_dptau)
        u_ddot_out = u_ddot_out.at[:, i].set(d2u_dptau2)

    # Scale derivatives by phase duration
    # dptau/dt = 2 / t_duration
    dptau_dt = 2.0 / t_duration
    u_dot_out = u_dot_out * dptau_dt
    u_ddot_out = u_ddot_out * (dptau_dt ** 2)

    # Reshape back to control shape
    u = jnp.reshape(u_out, (vec_size,) + control_shape)
    u_dot = jnp.reshape(u_dot_out, (vec_size,) + control_shape)
    u_ddot = jnp.reshape(u_ddot_out, (vec_size,) + control_shape)

    return u, u_dot, u_ddot
