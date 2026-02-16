"""JAX functions for continuity defect computation."""
import jax.numpy as jnp


def continuity_defect(values, segment_end_indices=None, dt_dptau=None, deriv_order=0):
    """
    Compute continuity defects at segment boundaries.

    This function computes the difference (jump) in a variable's value between the end
    of one segment and the start of the next segment. This defect should be constrained
    to zero to ensure continuity across segment boundaries.

    Parameters
    ----------
    values : ArrayLike
        Values of the variable at segment endpoint nodes.
        Shape: (num_segment_end_nodes, *var_shape)
    segment_end_indices : ArrayLike, optional
        Indices of segment end nodes. If None, assumes values are already at segment ends.
        For extracting from full node array: values[segment_end_indices]
        Shape: (num_segment_end_nodes,)
    dt_dptau : float, optional
        Scaling factor dt/dptau = t_duration / 2.0. Required when deriv_order > 0.
        Used to scale derivative continuity defects.
    deriv_order : int, optional
        Order of derivative being checked for continuity:
        - 0: value continuity (default)
        - 1: first derivative (rate) continuity
        - 2: second derivative (rate2) continuity

    Returns
    -------
    defect : ArrayLike
        Continuity defect at each segment boundary.
        Shape: (num_segments - 1, *var_shape)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless or in
    consistent units chosen by the caller.

    The segment endpoint nodes are ordered as:
        [seg0_start, seg0_end, seg1_start, seg1_end, seg2_start, seg2_end, ...]

    So the indices are:
        - seg0_end: index 1
        - seg1_start: index 2
        - seg1_end: index 3
        - seg2_start: index 4
        - etc.

    The defect at boundary i is:
        defect[i] = values[seg_i+1_start] - values[seg_i_end]
                  = values[2*(i+1)] - values[2*i + 1]

    For derivative continuity, the defect is scaled by (dt/dptau)^deriv_order.
    """
    # Extract values at segment endpoints if indices provided
    if segment_end_indices is not None:
        seg_values = values[segment_end_indices]
    else:
        seg_values = values

    # Extract end values: indices 1, 3, 5, ... (odd indices starting from 1)
    # These are the end of each segment (except the last segment)
    end_vals = seg_values[1:-1:2, ...]

    # Extract start values: indices 2, 4, 6, ... (even indices starting from 2)
    # These are the start of each segment (except the first segment)
    start_vals = seg_values[2::2, ...]

    # Compute defect: start of next segment minus end of previous segment
    defect = start_vals - end_vals

    # Scale by dt/dptau for derivative continuity
    if deriv_order > 0:
        if dt_dptau is None:
            raise ValueError("dt_dptau must be provided for derivative continuity")
        defect = defect * (dt_dptau ** deriv_order)

    return defect


def state_continuity_defect(states, segment_end_indices=None):
    """
    Compute state value continuity defects at segment boundaries.

    Convenience function for state continuity (deriv_order=0).

    Parameters
    ----------
    states : ArrayLike
        State values at segment endpoint nodes.
        Shape: (num_segment_end_nodes, *state_shape)
    segment_end_indices : ArrayLike, optional
        Indices of segment end nodes in the full state array.

    Returns
    -------
    defect : ArrayLike
        State continuity defect at each segment boundary.
        Shape: (num_segments - 1, *state_shape)

    Notes
    -----
    This is equivalent to:
        continuity_defect(states, segment_end_indices, deriv_order=0)
    """
    return continuity_defect(states, segment_end_indices, deriv_order=0)


def control_continuity_defect(controls, segment_end_indices=None):
    """
    Compute control value continuity defects at segment boundaries.

    Convenience function for control value continuity (deriv_order=0).

    Parameters
    ----------
    controls : ArrayLike
        Control values at segment endpoint nodes.
        Shape: (num_segment_end_nodes, *control_shape)
    segment_end_indices : ArrayLike, optional
        Indices of segment end nodes in the full control array.

    Returns
    -------
    defect : ArrayLike
        Control continuity defect at each segment boundary.
        Shape: (num_segments - 1, *control_shape)

    Notes
    -----
    This is equivalent to:
        continuity_defect(controls, segment_end_indices, deriv_order=0)
    """
    return continuity_defect(controls, segment_end_indices, deriv_order=0)


def control_rate_continuity_defect(control_rates, t_duration, segment_end_indices=None):
    """
    Compute control rate continuity defects at segment boundaries.

    Parameters
    ----------
    control_rates : ArrayLike
        Control rate values at segment endpoint nodes.
        Shape: (num_segment_end_nodes, *control_shape)
    t_duration : float
        Phase time duration, used to compute dt/dptau scaling.
    segment_end_indices : ArrayLike, optional
        Indices of segment end nodes in the full control rate array.

    Returns
    -------
    defect : ArrayLike
        Control rate continuity defect at each segment boundary.
        Shape: (num_segments - 1, *control_shape)

    Notes
    -----
    This is equivalent to:
        continuity_defect(control_rates, segment_end_indices,
                         dt_dptau=t_duration/2.0, deriv_order=1)
    """
    dt_dptau = t_duration / 2.0
    return continuity_defect(control_rates, segment_end_indices,
                           dt_dptau=dt_dptau, deriv_order=1)


def control_rate2_continuity_defect(control_rate2s, t_duration, segment_end_indices=None):
    """
    Compute control second derivative continuity defects at segment boundaries.

    Parameters
    ----------
    control_rate2s : ArrayLike
        Control second derivative values at segment endpoint nodes.
        Shape: (num_segment_end_nodes, *control_shape)
    t_duration : float
        Phase time duration, used to compute dt/dptau scaling.
    segment_end_indices : ArrayLike, optional
        Indices of segment end nodes in the full control rate2 array.

    Returns
    -------
    defect : ArrayLike
        Control second derivative continuity defect at each segment boundary.
        Shape: (num_segments - 1, *control_shape)

    Notes
    -----
    This is equivalent to:
        continuity_defect(control_rate2s, segment_end_indices,
                         dt_dptau=t_duration/2.0, deriv_order=2)
    """
    dt_dptau = t_duration / 2.0
    return continuity_defect(control_rate2s, segment_end_indices,
                           dt_dptau=dt_dptau, deriv_order=2)
