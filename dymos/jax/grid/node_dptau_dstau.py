"""JAX function for computing node_dptau_dstau from grid parameters."""
import jax.numpy as jnp


def node_dptau_dstau(segment_ends, nodes_per_segment, total_nodes=None):
    """
    Compute dptau/dstau at each node given segment endpoints and node distribution.

    This function computes the derivative of phase tau (ptau) with respect to
    segment tau (stau) at each node. This mapping is needed to transform between
    segment-local coordinates (stau ∈ [-1, 1] for each segment) and phase-global
    coordinates (ptau spanning the entire phase).

    Parameters
    ----------
    segment_ends : ArrayLike
        Phase tau values at segment boundaries.
        Shape: (num_segments + 1,)
        For example, [-1.0, 0.0, 1.0] represents 2 segments of equal length.
        First value is phase start, last value is phase end, intermediate values
        are segment boundaries.
    nodes_per_segment : ArrayLike
        Number of nodes in each segment.
        Shape: (num_segments,)
        For example, [5, 5] means 5 nodes in each of 2 segments.
    total_nodes : int, optional
        Total number of nodes (sum of nodes_per_segment).
        If provided, improves JIT compilation efficiency. If None, computed
        automatically.

    Returns
    -------
    node_dptau_dstau : ArrayLike
        Derivative dptau/dstau at each node.
        Shape: (total_num_nodes,)
        where total_num_nodes = sum(nodes_per_segment)

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, jax.vmap, etc.
    Units are not handled - all inputs/outputs are assumed dimensionless.

    For segment i spanning from ptau_start to ptau_end in phase tau space:
        dptau/dstau = (ptau_end - ptau_start) / 2.0

    The division by 2 is because segment tau spans [-1, 1] (a range of 2),
    so a unit change in stau corresponds to (ptau_end - ptau_start) / 2 in ptau.

    This value is constant within each segment and repeated for all nodes in
    that segment.

    Making segment_ends differentiable enables grid optimization where segment
    spacing can be tuned to improve solution accuracy or convergence.

    Examples
    --------
    >>> # Two equal segments with 3 nodes each
    >>> segment_ends = jnp.array([-1.0, 0.0, 1.0])
    >>> nodes_per_segment = jnp.array([3, 3])
    >>> result = node_dptau_dstau(segment_ends, nodes_per_segment)
    >>> # Each segment has dptau_dstau = (0 - (-1))/2 = 0.5 or (1 - 0)/2 = 0.5
    >>> # Result: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    >>> # Two unequal segments: first is 75% of phase, second is 25%
    >>> segment_ends = jnp.array([-1.0, 0.5, 1.0])
    >>> nodes_per_segment = jnp.array([4, 2])
    >>> result = node_dptau_dstau(segment_ends, nodes_per_segment)
    >>> # Segment 0: dptau_dstau = (0.5 - (-1))/2 = 0.75
    >>> # Segment 1: dptau_dstau = (1.0 - 0.5)/2 = 0.25
    >>> # Result: [0.75, 0.75, 0.75, 0.75, 0.25, 0.25]
    """
    num_segments = len(nodes_per_segment)

    # Compute dptau_dstau for each segment
    # For segment i: dptau_dstau = (segment_ends[i+1] - segment_ends[i]) / 2.0
    segment_starts = segment_ends[:-1]  # First num_segments elements
    segment_ends_vals = segment_ends[1:]  # Last num_segments elements

    seg_dptau_dstau = (segment_ends_vals - segment_starts) / 2.0

    # Repeat each segment's dptau_dstau value for all nodes in that segment
    # Use manual construction to build the repeated array
    # Build cumulative indices to know where each segment starts
    cumsum_nodes = jnp.cumsum(nodes_per_segment)

    # Use provided total_nodes or compute it
    if total_nodes is None:
        total_nodes = cumsum_nodes[-1]

    # Create array of segment indices for each node
    # Node i belongs to segment j where cumsum_nodes[j-1] <= i < cumsum_nodes[j]
    node_indices = jnp.arange(total_nodes)

    # For each node, determine which segment it belongs to
    # by finding the first cumsum value that exceeds its index
    segment_for_node = jnp.searchsorted(cumsum_nodes, node_indices, side='right')

    # Index into seg_dptau_dstau to get the value for each node
    node_vals = seg_dptau_dstau[segment_for_node]

    return node_vals
