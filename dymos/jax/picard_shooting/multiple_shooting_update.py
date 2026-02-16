"""JAX function for multiple shooting state updates."""
import jax.numpy as jnp


def multiple_shooting_update_forward(x, x_a, seg_end_indices):
    """
    Compute segment initial states for forward multiple shooting.

    Given the phase initial state and the full state vector, extract the
    appropriate values to use as initial conditions for each segment during
    forward propagation in Picard iteration.

    Parameters
    ----------
    x : ArrayLike
        State values at all nodes.
        Shape: (num_nodes, *state_shape)
    x_a : ArrayLike
        Desired initial state value for the phase.
        Shape: (1, *state_shape)
    seg_end_indices : ArrayLike
        Indices of segment end nodes (odd indices: 1, 3, 5, ...).
        Shape: (num_segments - 1,)

    Returns
    -------
    x_0 : ArrayLike
        Initial state value for each segment.
        Shape: (num_segments, *state_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The segment initial states are constructed as:
    - First segment: uses x_a (phase initial condition)
    - Subsequent segments: use the final state from the previous segment (x[seg_end_indices[i-1]])

    This function is used in the forward shooting direction where we integrate
    from the initial state forward in time.
    """
    num_segs = len(seg_end_indices) + 1
    state_shape = x.shape[1:]

    # Initialize output array
    x_0 = jnp.zeros((num_segs,) + state_shape, dtype=x.dtype)

    # First segment gets the phase initial condition
    x_0 = x_0.at[0].set(x_a[0])

    # Subsequent segments get the final state from the previous segment
    if num_segs > 1:
        # seg_end_indices contains indices [1, 3, 5, ...] for segment ends
        # These become initial conditions for segments [1, 2, 3, ...]
        x_seg_ends = x[seg_end_indices]
        x_0 = x_0.at[1:].set(x_seg_ends)

    return x_0


def multiple_shooting_update_backward(x, x_b, seg_start_indices):
    """
    Compute segment final states for backward multiple shooting.

    Given the phase final state and the full state vector, extract the
    appropriate values to use as final conditions for each segment during
    backward propagation in Picard iteration.

    Parameters
    ----------
    x : ArrayLike
        State values at all nodes.
        Shape: (num_nodes, *state_shape)
    x_b : ArrayLike
        Desired final state value for the phase.
        Shape: (1, *state_shape)
    seg_start_indices : ArrayLike
        Indices of segment start nodes for segments 1..N-1 (even indices: 2, 4, 6, ...).
        Shape: (num_segments - 1,)

    Returns
    -------
    x_f : ArrayLike
        Final state value for each segment.
        Shape: (num_segments, *state_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The segment final states are constructed as:
    - Last segment: uses x_b (phase final condition)
    - Previous segments: use the initial state from the next segment (x[seg_start_indices[i]])

    This function is used in the backward shooting direction where we integrate
    from the final state backward in time.
    """
    num_segs = len(seg_start_indices) + 1
    state_shape = x.shape[1:]

    # Initialize output array
    x_f = jnp.zeros((num_segs,) + state_shape, dtype=x.dtype)

    # Last segment gets the phase final condition
    x_f = x_f.at[-1].set(x_b[0])

    # Previous segments get the initial state from the next segment
    if num_segs > 1:
        # seg_start_indices contains indices [2, 4, 6, ...] for segment starts (of segments 1, 2, 3, ...)
        # These become final conditions for segments [0, 1, 2, ...]
        x_seg_starts = x[seg_start_indices]
        x_f = x_f.at[:-1].set(x_seg_starts)

    return x_f
