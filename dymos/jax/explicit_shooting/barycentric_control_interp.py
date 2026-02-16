"""JAX function for barycentric Lagrange control interpolation in explicit shooting."""
import jax.numpy as jnp


def _compute_lagrange_basis(tau, taus):
    """
    Compute Lagrange basis polynomials and their first two derivatives.

    This is a simplified JAX-compatible implementation. For better performance
    and numerical stability, consider using barycentric weights.

    Parameters
    ----------
    tau : float or ArrayLike
        The value(s) of the independent variable where interpolation is requested.
        Scalar or shape: (vec_size,)
    taus : ArrayLike
        Node locations for the polynomial basis.
        Shape: (n,)

    Returns
    -------
    l : ArrayLike
        Lagrange basis polynomial values at tau.
        Shape: (vec_size, n)
    dl_dtau : ArrayLike
        First derivative of basis polynomials.
        Shape: (vec_size, n)
    d2l_dtau2 : ArrayLike
        Second derivative of basis polynomials.
        Shape: (vec_size, n)

    Notes
    -----
    The Lagrange basis function for node i is:
        l_i(tau) = prod_{j != i} (tau - tau_j) / (tau_i - tau_j)
    """
    # Ensure tau is at least 1D for consistent handling
    tau = jnp.atleast_1d(tau)
    vec_size = len(tau)
    n = len(taus)

    # Initialize outputs
    l = jnp.zeros((vec_size, n))
    dl_dtau = jnp.zeros((vec_size, n))
    d2l_dtau2 = jnp.zeros((vec_size, n))

    # For each basis function i
    for i in range(n):
        # Compute numerator: prod_{j != i} (tau - tau_j)
        numerator = jnp.ones(vec_size)
        for j in range(n):
            if j != i:
                numerator = numerator * (tau - taus[j])

        # Compute denominator: prod_{j != i} (tau_i - tau_j)
        denominator = 1.0
        for j in range(n):
            if j != i:
                denominator = denominator * (taus[i] - taus[j])

        # Lagrange basis function
        l = l.at[:, i].set(numerator / denominator)

        # First derivative: sum over k of [ prod_{j != i, j != k} (tau - tau_j) / (tau_i - tau_j) ]
        dl_i = jnp.zeros(vec_size)
        for k in range(n):
            if k != i:
                term = jnp.ones(vec_size)
                for j in range(n):
                    if j != i and j != k:
                        term = term * (tau - taus[j])
                dl_i = dl_i + term / denominator

        dl_dtau = dl_dtau.at[:, i].set(dl_i)

        # Second derivative: sum over k, m (k != m) of [ prod_{j != i, j != k, j != m} (tau - tau_j) / (tau_i - tau_j) ]
        d2l_i = jnp.zeros(vec_size)
        for k in range(n):
            if k != i:
                for m in range(k + 1, n):
                    if m != i:
                        term = jnp.ones(vec_size)
                        for j in range(n):
                            if j != i and j != k and j != m:
                                term = term * (tau - taus[j])
                        d2l_i = d2l_i + 2.0 * term / denominator  # Factor of 2 for symmetry

        d2l_dtau2 = d2l_dtau2.at[:, i].set(d2l_i)

    return l, dl_dtau, d2l_dtau2


def barycentric_control_interp(u_input, stau, dstau_dt, input_to_disc_map,
                               disc_node_indices, taus_seg, w_b):
    """
    Interpolate control values using barycentric Lagrange interpolation.

    Barycentric Lagrange interpolation is a numerically stable method for polynomial
    interpolation that uses barycentric weights to evaluate the interpolant efficiently.

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
        Shape: (n,) where n is the number of nodes in the segment
    taus_seg : ArrayLike
        Segment tau locations of the discretization nodes.
        Shape: (n,)
    w_b : ArrayLike
        Barycentric weights for the interpolation nodes.
        Shape: (n, n) - diagonal matrix of weights

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

    The barycentric form provides better numerical stability than direct Vandermonde
    or Lagrange interpolation, especially for higher-order polynomials.
    """
    vec_size = len(stau)
    control_shape = u_input.shape[1:]
    n = len(taus_seg)

    # Step 1: Map input nodes to discretization nodes
    u_disc = u_input[input_to_disc_map]

    # Step 2: Extract values at nodes in current segment
    u_seg = u_disc[disc_node_indices]

    # Flatten control for matrix operations
    u_seg_flat = jnp.reshape(u_seg, (n, -1))

    # Step 3: Compute Lagrange basis functions and derivatives
    l, dl_dstau, d2l_dstau2 = _compute_lagrange_basis(stau, taus_seg)
    # l has shape (vec_size, n), dl_dstau has shape (vec_size, n), etc.

    # Step 4: Apply barycentric weights
    # wbuhat = w_b @ u_seg (element-wise weighting)
    # w_b is typically a diagonal matrix, so we can do element-wise multiplication
    if w_b.ndim == 2:
        # If w_b is a matrix, apply it
        wbuhat = w_b @ u_seg_flat
    else:
        # If w_b is a vector of diagonal elements
        wbuhat = w_b[:, jnp.newaxis] * u_seg_flat

    # Step 5: Compute interpolated values and derivatives
    # u = l^T @ wbuhat (matrix multiply across nodes)
    # l has shape (vec_size, n), wbuhat has shape (n, num_elements)
    # Result should be (vec_size, num_elements)
    u_flat = l @ wbuhat
    u_dot_flat = dl_dstau @ wbuhat
    u_ddot_flat = d2l_dstau2 @ wbuhat

    # Scale derivatives by dstau_dt
    if jnp.ndim(dstau_dt) == 0:
        dstau_dt_exp = dstau_dt
        dstau_dt2 = dstau_dt ** 2
    else:
        dstau_dt_exp = dstau_dt[:, jnp.newaxis]
        dstau_dt2 = dstau_dt_exp ** 2

    u_dot_flat = u_dot_flat * dstau_dt_exp
    u_ddot_flat = u_ddot_flat * dstau_dt2

    # Step 6: Reshape back to control shape
    u = jnp.reshape(u_flat, (vec_size,) + control_shape)
    u_dot = jnp.reshape(u_dot_flat, (vec_size,) + control_shape)
    u_ddot = jnp.reshape(u_ddot_flat, (vec_size,) + control_shape)

    return u, u_dot, u_ddot
