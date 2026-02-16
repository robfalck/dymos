"""JAX functions for state interpolation in pseudospectral transcriptions."""
import jax.numpy as jnp


def state_interp_radau(xd, dt_dstau, Ad):
    """
    Interpolate state derivatives at collocation nodes using Radau transcription.

    Uses Lagrange interpolation to compute state derivatives at collocation nodes
    from state values at discretization nodes.

    Parameters
    ----------
    xd : ArrayLike
        State values at discretization nodes.
        Shape: (num_disc_nodes, *state_shape)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at each collocation node.
        Shape: (num_col_nodes,)
    Ad : ArrayLike
        Lagrange differentiation matrix.
        Shape: (num_col_nodes, num_disc_nodes)

    Returns
    -------
    xdotc : ArrayLike
        State derivative at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    """
    # Flatten state dimensions for matrix multiplication
    num_disc_nodes = xd.shape[0]
    state_shape = xd.shape[1:]

    # Reshape using -1 to let JAX infer the flattened dimension (JIT-compatible)
    xd_flat = jnp.reshape(xd, (num_disc_nodes, -1))

    # Apply differentiation matrix
    xdotc_flat = Ad @ xd_flat

    # Divide by dt_dstau (broadcast across state dimensions)
    dt_dstau_expanded = dt_dstau[:, jnp.newaxis]
    xdotc_flat = xdotc_flat / dt_dstau_expanded

    # Reshape back to original state shape
    num_col_nodes = Ad.shape[0]
    xdotc = jnp.reshape(xdotc_flat, (num_col_nodes,) + state_shape)

    return xdotc


def state_interp_gauss_lobatto(xd, fd, dt_dstau, Ai, Bi, Ad, Bd):
    """
    Interpolate states and derivatives at collocation nodes using Gauss-Lobatto transcription.

    Uses Hermite interpolation to compute state values and derivatives at collocation nodes
    from state values and derivatives at discretization nodes.

    Parameters
    ----------
    xd : ArrayLike
        State values at discretization nodes.
        Shape: (num_disc_nodes, *state_shape)
    fd : ArrayLike
        State derivatives at discretization nodes.
        Shape: (num_disc_nodes, *state_shape)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at each collocation node.
        Shape: (num_col_nodes,)
    Ai : ArrayLike
        Hermite interpolation matrix for state values.
        Shape: (num_col_nodes, num_disc_nodes)
    Bi : ArrayLike
        Hermite interpolation matrix for state derivatives.
        Shape: (num_col_nodes, num_disc_nodes)
    Ad : ArrayLike
        Hermite differentiation matrix.
        Shape: (num_col_nodes, num_disc_nodes)
    Bd : ArrayLike
        Hermite differentiation matrix for derivatives.
        Shape: (num_col_nodes, num_disc_nodes)

    Returns
    -------
    xc : ArrayLike
        Interpolated state values at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    xdotc : ArrayLike
        Interpolated state derivatives at collocation nodes.
        Shape: (num_col_nodes, *state_shape)
    """
    # Flatten state dimensions for matrix multiplication
    num_disc_nodes = xd.shape[0]
    state_shape = xd.shape[1:]

    # Reshape using -1 to let JAX infer the flattened dimension (JIT-compatible)
    xd_flat = jnp.reshape(xd, (num_disc_nodes, -1))
    fd_flat = jnp.reshape(fd, (num_disc_nodes, size))

    # Broadcast dt_dstau for element-wise operations
    dt_dstau_expanded = dt_dstau[:, jnp.newaxis]

    # Compute interpolated state values at collocation nodes
    # xc = Ai @ xd + Bi @ fd * dt_dstau
    xc_flat = Ai @ xd_flat + (Bi @ fd_flat) * dt_dstau_expanded

    # Compute interpolated state derivatives at collocation nodes
    # xdotc = Ad @ xd / dt_dstau + Bd @ fd
    xdotc_flat = (Ad @ xd_flat) / dt_dstau_expanded + Bd @ fd_flat

    # Reshape back to original state shape
    num_col_nodes = Ad.shape[0]
    xc = jnp.reshape(xc_flat, (num_col_nodes,) + state_shape)
    xdotc = jnp.reshape(xdotc_flat, (num_col_nodes,) + state_shape)

    return xc, xdotc
