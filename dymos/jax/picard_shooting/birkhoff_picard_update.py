"""JAX function for Birkhoff-based Picard state updates."""
import jax.numpy as jnp


def birkhoff_picard_update_forward(f_computed, dt_dstau, x_0, B, seg_repeats):
    """
    Compute updated states using Birkhoff integration for forward shooting.

    Integrates the state derivatives using Birkhoff interpolation to produce
    updated state values during Picard iteration. Integrates forward in time
    from segment initial conditions.

    Parameters
    ----------
    f_computed : ArrayLike
        Computed state derivatives (dx/dt) from ODE at all nodes.
        Shape: (num_nodes, *state_shape)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at each node.
        Shape: (num_nodes,)
    x_0 : ArrayLike
        Initial state value for each segment.
        Shape: (num_segments, *state_shape)
    B : ArrayLike
        Block-diagonal Birkhoff integration matrix (dense).
        Shape: (num_nodes, num_nodes)
    seg_repeats : ArrayLike
        Number of nodes per segment (for repeating x_0).
        Shape: (num_segments,)

    Returns
    -------
    x_hat : ArrayLike
        Updated state values at all nodes.
        Shape: (num_nodes, *state_shape)
    x_b : ArrayLike
        Final state value (last node of x_hat).
        Shape: (1, *state_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The integration is performed as:
        x_hat = x_0_repeated + B @ (f_computed * dt_dstau)

    where x_0_repeated broadcasts the segment initial conditions to all nodes
    within each segment, and B performs Birkhoff quadrature to integrate the
    state derivatives.

    Note: For use in Picard iteration with NLBGS, the derivative computation
    through this function should use custom_root to avoid differentiating
    through the iterative solver.
    """
    num_nodes = f_computed.shape[0]
    state_shape = f_computed.shape[1:]

    # Convert state rates from dx/dt to dx/dstau
    dt_dstau_expanded = jnp.reshape(dt_dstau, (num_nodes,) + (1,) * len(state_shape))
    f_dstau = f_computed * dt_dstau_expanded

    # Flatten for matrix multiplication
    f_flat = jnp.reshape(f_dstau, (num_nodes, -1))

    # Repeat x_0 for each node in its segment
    # Use total_repeat_length to make JIT-compatible
    x_0_repeated = jnp.repeat(x_0, seg_repeats, axis=0, total_repeat_length=num_nodes)
    x_0_flat = jnp.reshape(x_0_repeated, (num_nodes, -1))

    # Integrate: x_hat = x_0 + integral(f * dt_dstau)
    integrated_flat = B @ f_flat
    x_hat_flat = x_0_flat + integrated_flat

    # Reshape to original state shape
    x_hat = jnp.reshape(x_hat_flat, (num_nodes,) + state_shape)

    # Extract final state (last node)
    x_b = x_hat[-1:, ...]

    return x_hat, x_b


def birkhoff_picard_update_backward(f_computed, dt_dstau, x_f, B, seg_repeats):
    """
    Compute updated states using Birkhoff integration for backward shooting.

    Integrates the state derivatives using Birkhoff interpolation to produce
    updated state values during Picard iteration. Integrates backward in time
    from segment final conditions.

    Parameters
    ----------
    f_computed : ArrayLike
        Computed state derivatives (dx/dt) from ODE at all nodes.
        Shape: (num_nodes, *state_shape)
    dt_dstau : ArrayLike
        Ratio of time derivative to segment tau derivative at each node.
        Shape: (num_nodes,)
    x_f : ArrayLike
        Final state value for each segment.
        Shape: (num_segments, *state_shape)
    B : ArrayLike
        Block-diagonal Birkhoff integration matrix (dense).
        Shape: (num_nodes, num_nodes)
    seg_repeats : ArrayLike
        Number of nodes per segment (for repeating x_f).
        Shape: (num_segments,)

    Returns
    -------
    x_hat : ArrayLike
        Updated state values at all nodes.
        Shape: (num_nodes, *state_shape)
    x_a : ArrayLike
        Initial state value (first node of x_hat).
        Shape: (1, *state_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The integration is performed as:
        x_hat = x_f_repeated - B_flipped @ (f_computed_flipped * dt_dstau_flipped)

    where the flipped arrays reverse time direction for backward integration.

    Note: For use in Picard iteration with NLBGS, the derivative computation
    through this function should use custom_root to avoid differentiating
    through the iterative solver.
    """
    num_nodes = f_computed.shape[0]
    state_shape = f_computed.shape[1:]

    # Convert state rates from dx/dt to dx/dstau
    dt_dstau_expanded = jnp.reshape(dt_dstau, (num_nodes,) + (1,) * len(state_shape))
    f_dstau = f_computed * dt_dstau_expanded

    # Flatten for matrix multiplication
    f_flat = jnp.reshape(f_dstau, (num_nodes, -1))

    # Repeat x_f for each node in its segment
    # Use total_repeat_length to make JIT-compatible
    x_f_repeated = jnp.repeat(x_f, seg_repeats, axis=0, total_repeat_length=num_nodes)
    x_f_flat = jnp.reshape(x_f_repeated, (num_nodes, -1))

    # Flip arrays for backward integration
    # In backward mode, we integrate from final to initial
    f_flat_flip = f_flat[::-1, :]
    B_flip = B[::-1, ::-1]

    # Integrate: x_hat = x_f - integral_backward(f * dt_dstau)
    integrated_flat = B_flip @ f_flat_flip
    x_hat_flat = x_f_flat - jnp.reshape(integrated_flat, (num_nodes, -1))

    # Reshape to original state shape
    x_hat = jnp.reshape(x_hat_flat, (num_nodes,) + state_shape)

    # Extract initial state (first node)
    x_a = x_hat[:1, ...]

    return x_hat, x_a
