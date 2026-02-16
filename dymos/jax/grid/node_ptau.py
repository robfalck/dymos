"""JAX function for computing node_ptau from grid parameters."""
import jax.numpy as jnp


def node_ptau(segment_ends, node_stau, nodes_per_segment, total_nodes=None):
    """
    Compute phase tau (ptau) coordinates for all nodes from segment structure.

    This function maps nodes from segment-local tau coordinates (stau ∈ [-1, 1])
    to phase-global tau coordinates (ptau). Each segment occupies a portion of
    the phase tau space, and nodes within each segment are positioned according
    to their segment tau locations.

    Parameters
    ----------
    segment_ends : ArrayLike
        Phase tau values at segment boundaries.
        Shape: (num_segments + 1,)
        For example, [-1.0, 0.0, 1.0] represents 2 segments of equal length.
        First value is phase start, last value is phase end.
    node_stau : ArrayLike
        Segment tau coordinate for each node (stau ∈ [-1, 1] within each segment).
        Shape: (total_num_nodes,)
        Nodes are ordered by segment (all segment 0 nodes, then segment 1, etc.)
    nodes_per_segment : ArrayLike
        Number of nodes in each segment.
        Shape: (num_segments,)
        For example, [5, 5] means 5 nodes in each of 2 segments.
    total_nodes : int, optional
        Total number of nodes (sum of nodes_per_segment).
        If provided, improves JIT compilation efficiency. If None, computed
        automatically. Should be marked as static_argnames for JIT.

    Returns
    -------
    node_ptau : ArrayLike
        Phase tau coordinate for each node.
        Shape: (total_num_nodes,)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless.

    The transformation from segment tau to phase tau is:
        ptau = ptau_start + (stau + 1) * (ptau_end - ptau_start) / 2

    where ptau_start and ptau_end are the boundaries of the segment in phase space.

    This mapping ensures:
    - When stau = -1: ptau = ptau_start (segment beginning)
    - When stau = +1: ptau = ptau_end (segment ending)
    - When stau = 0: ptau = (ptau_start + ptau_end) / 2 (segment midpoint)

    Making segment_ends differentiable enables grid optimization where segment
    spacing can be tuned to improve solution accuracy or convergence.

    Examples
    --------
    >>> # Two equal segments, each with 3 nodes at stau = [-1, 0, 1]
    >>> segment_ends = jnp.array([-1.0, 0.0, 1.0])
    >>> node_stau = jnp.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    >>> nodes_per_segment = jnp.array([3, 3])
    >>> ptau = node_ptau(segment_ends, node_stau, nodes_per_segment)
    >>> # Segment 0 maps [-1, 0, 1] in stau to [-1.0, -0.5, 0.0] in ptau
    >>> # Segment 1 maps [-1, 0, 1] in stau to [0.0, 0.5, 1.0] in ptau
    >>> # Result: [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0]

    >>> # Two unequal segments: first is 75% of phase
    >>> segment_ends = jnp.array([-1.0, 0.5, 1.0])
    >>> node_stau = jnp.array([-1.0, 1.0, -1.0, 1.0])
    >>> nodes_per_segment = jnp.array([2, 2])
    >>> ptau = node_ptau(segment_ends, node_stau, nodes_per_segment)
    >>> # Segment 0: stau=-1 -> ptau=-1.0, stau=1 -> ptau=0.5
    >>> # Segment 1: stau=-1 -> ptau=0.5, stau=1 -> ptau=1.0
    >>> # Result: [-1.0, 0.5, 0.5, 1.0]
    """
    num_segments = len(nodes_per_segment)

    # Build cumulative indices to know where each segment starts
    cumsum_nodes = jnp.cumsum(nodes_per_segment)

    # Use provided total_nodes or compute it
    if total_nodes is None:
        total_nodes = cumsum_nodes[-1]

    # For each node, determine which segment it belongs to
    node_indices = jnp.arange(total_nodes)
    segment_for_node = jnp.searchsorted(cumsum_nodes, node_indices, side='right')

    # Get segment boundaries for each node's segment
    ptau_start = segment_ends[segment_for_node]
    ptau_end = segment_ends[segment_for_node + 1]

    # Map from segment tau to phase tau
    # ptau = ptau_start + (stau + 1) * (ptau_end - ptau_start) / 2
    # When stau = -1: ptau = ptau_start
    # When stau = +1: ptau = ptau_end
    node_ptau_vals = ptau_start + (node_stau + 1.0) * (ptau_end - ptau_start) / 2.0

    return node_ptau_vals
