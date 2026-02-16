def time(t_initial, t_duration, node_ptau, node_dptau_dstau):
    """
    Compute aspects of time for the phase.

    Parameters
    ----------
    t_initial : float
        Time at which the phase begins.
    t_duration : float
        Duration of the phase.
    node_ptau : ArrayLike
        The non-dimensional phase "tau" time at which each node in the phase occurs.
    node_dptau_dstau : ArrayLike
        For each node, the ratio of the total phase length to the length of the node's containing segment.

    Returns
    -------
    t : ArrayLike
        Time at each of the nodes in the phase.
    t_phase : ArrayLike
        Time since the start of the phase for each node in the phase.
    dt_dstau : ArrayLike
        For each node, the ratio of its parent segment time duration to its nondimensional duration.
    """
    t = t_initial + 0.5 * (node_ptau + 1) * t_duration
    t_phase = 0.5 * (node_ptau + 1) * t_duration
    dt_dstau = 0.5 * t_duration * node_dptau_dstau

    return t, t_phase, dt_dstau
