"""JAX function for tau computation in explicit shooting."""


def tau(t, t_initial, t_duration, ptau0_seg, ptauf_seg):
    """
    Compute phase tau, segment tau, and dstau/dt based on time.

    Parameters
    ----------
    t : ArrayLike
        Time values at each node.
    t_initial : float
        Initial time of the phase.
    t_duration : float
        Duration of the phase.
    ptau0_seg : float
        Phase tau at the start of the segment.
    ptauf_seg : float
        Phase tau at the end of the segment.

    Returns
    -------
    ptau : ArrayLike
        Phase tau (normalized time from -1 to 1 over the phase).
    stau : ArrayLike
        Segment tau (normalized time from -1 to 1 within the segment).
    dstau_dt : float
        Derivative of segment tau with respect to time.
    t_phase : ArrayLike
        Time since the start of the phase.
    """
    # Compute phase tau
    ptau = 2.0 * (t - t_initial) / t_duration - 1.0

    # Compute segment duration in ptau space
    td_seg = ptauf_seg - ptau0_seg

    # Compute segment tau
    stau = 2.0 * (ptau - ptau0_seg) / td_seg - 1.0

    # Compute derivative of stau with respect to time
    dstau_dt = 4.0 / (t_duration * td_seg)

    # Compute time since start of phase
    t_phase = t - t_initial

    return ptau, stau, dstau_dt, t_phase
