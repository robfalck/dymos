"""
Radau Pseudospectral Phase for Brachistochrone Problem using JAX.

This module demonstrates a complete phase assembly using pure JAX functions
for the brachistochrone optimal control problem with Radau pseudospectral
collocation.
"""
import jax.numpy as jnp
from dymos.jax.common.time import time
from dymos.jax.grid.node_ptau import node_ptau
from dymos.jax.grid.node_dptau_dstau import node_dptau_dstau
from dymos.jax.pseudospectral.components.state_interp import state_interp_radau
from dymos.jax.pseudospectral.components.radau_defect import radau_defect
from dymos.jax.examples.brachistochrone_ode import brachistochrone_ode


def radau_brachistochrone_phase(design_vars, grid_data, options):
    """
    Assemble complete Radau pseudospectral phase for brachistochrone problem.

    This function combines all components needed for a Radau transcription:
    - Grid computations (ptau, dptau_dstau, time)
    - State interpolation at collocation nodes
    - ODE evaluation
    - Defect computation
    - Boundary conditions

    Parameters
    ----------
    design_vars : dict
        Design variables for the optimization problem:
        - 'states:x' : array (num_disc_nodes,) - x at discretization nodes
        - 'states:y' : array (num_disc_nodes,) - y at discretization nodes
        - 'states:v' : array (num_disc_nodes,) - v at discretization nodes
        - 'controls:theta' : array (num_all_nodes,) - theta at all nodes
        - 't_initial' : float - initial time
        - 't_duration' : float - phase duration
        - 'x_initial' : float (optional) - initial x constraint
        - 'y_initial' : float (optional) - initial y constraint
        - 'v_initial' : float (optional) - initial v constraint
        - 'x_final' : float (optional) - final x constraint
        - 'y_final' : float (optional) - final y constraint
        - 'v_final' : float (optional) - final v constraint

    grid_data : dict
        Static grid structure information:
        - 'segment_ends' : array (num_segments + 1,) - segment boundaries in ptau
        - 'nodes_per_segment' : array (num_segments,) - nodes per segment
        - 'node_stau' : array (num_all_nodes,) - segment tau for all nodes
        - 'disc_indices' : array (num_disc_nodes,) - indices of discretization nodes
        - 'col_indices' : array (num_col_nodes,) - indices of collocation nodes
        - 'seg_end_indices' : array (2 * num_segments,) - segment endpoint indices
        - 'D_matrix' : array (num_col_nodes, num_disc_nodes) - differentiation matrix
        - 'A_matrix' : array (num_col_nodes, num_disc_nodes) - interpolation matrix

    options : dict
        Problem options:
        - 'g' : float - gravitational acceleration (default: 9.80665)
        - 'enforce_initial' : dict - which initial states to constrain
        - 'enforce_final' : dict - which final states to constrain

    Returns
    -------
    residuals : dict
        Constraint residuals and objectives:
        - 'defect:x' : array (num_col_nodes,) - x defects at collocation nodes
        - 'defect:y' : array (num_col_nodes,) - y defects
        - 'defect:v' : array (num_col_nodes,) - v defects
        - 'initial:x' : float - initial x constraint (if enforced)
        - 'initial:y' : float - initial y constraint (if enforced)
        - 'initial:v' : float - initial v constraint (if enforced)
        - 'final:x' : float - final x constraint (if enforced)
        - 'final:y' : float - final y constraint (if enforced)
        - 'final:v' : float - final v constraint (if enforced)
        - 'continuity:x' : array (num_segments - 1,) - continuity defects
        - 'continuity:y' : array (num_segments - 1,)
        - 'continuity:v' : array (num_segments - 1,)
        - 'objective' : float - time to minimize

    Notes
    -----
    This is a pure function suitable for use with jax.jit, jax.grad, etc.
    All inputs should be JAX arrays or compatible types.

    The Radau pseudospectral method uses:
    - State values defined at discretization nodes (segment endpoints)
    - ODE evaluated at collocation nodes (Legendre-Gauss-Radau points)
    - Polynomial interpolation to connect them
    - Defect constraints to enforce dynamics

    Examples
    --------
    >>> # Setup grid for 2 segments with 3 Radau nodes each
    >>> num_segments = 2
    >>> segment_ends = jnp.array([-1.0, 0.0, 1.0])
    >>> nodes_per_segment = jnp.array([4, 4])  # 3 LGR + 1 endpoint each
    >>> # ... (setup other grid_data)
    >>>
    >>> # Design variables
    >>> design_vars = {
    >>>     'states:x': jnp.linspace(0, 10, 5),  # 5 disc nodes
    >>>     'states:y': jnp.linspace(0, -10, 5),
    >>>     'states:v': jnp.linspace(0, 14, 5),
    >>>     'controls:theta': jnp.ones(8) * 0.5,  # 8 total nodes
    >>>     't_initial': 0.0,
    >>>     't_duration': 1.8,
    >>> }
    >>>
    >>> # Evaluate phase
    >>> residuals = radau_brachistochrone_phase(design_vars, grid_data, options)
    """
    # Extract design variables
    x_disc = design_vars['states:x']
    y_disc = design_vars['states:y']
    v_disc = design_vars['states:v']
    theta_all = design_vars['controls:theta']
    t_initial = design_vars['t_initial']
    t_duration = design_vars['t_duration']

    # Extract grid data
    segment_ends = grid_data['segment_ends']
    nodes_per_segment = grid_data['nodes_per_segment']
    node_stau = grid_data['node_stau']
    disc_indices = grid_data['disc_indices']
    col_indices = grid_data['col_indices']
    seg_end_indices = grid_data['seg_end_indices']
    D_matrix = grid_data['D_matrix']
    A_matrix = grid_data['A_matrix']

    # Extract options
    g = options.get('g', 9.80665)
    enforce_initial = options.get('enforce_initial', {})
    enforce_final = options.get('enforce_final', {})

    # Compute total nodes for static argument
    num_all_nodes = len(node_stau)
    num_disc_nodes = len(disc_indices)
    num_col_nodes = len(col_indices)

    # ========================================================================
    # 1. Grid Computations
    # ========================================================================

    # Compute phase tau coordinates for all nodes
    ptau_all = node_ptau(segment_ends, node_stau, nodes_per_segment,
                        total_nodes=num_all_nodes)

    # Compute dptau/dstau for all nodes
    dptau_dstau_all = node_dptau_dstau(segment_ends, nodes_per_segment,
                                       total_nodes=num_all_nodes)

    # Compute time at all nodes
    t_all, t_phase_all, dt_dstau_all = time(t_initial, t_duration,
                                             ptau_all, dptau_dstau_all)

    # Extract time values at collocation nodes
    dt_dstau_col = dt_dstau_all[col_indices]

    # ========================================================================
    # 2. State Interpolation at Collocation Nodes
    # ========================================================================

    # Interpolate states from discretization nodes to collocation nodes
    x_col = state_interp_radau(x_disc, dt_dstau_col, A_matrix)
    y_col = state_interp_radau(y_disc, dt_dstau_col, A_matrix)
    v_col = state_interp_radau(v_disc, dt_dstau_col, A_matrix)

    # Extract controls at collocation nodes
    theta_col = theta_all[col_indices]

    # ========================================================================
    # 3. ODE Evaluation at Collocation Nodes
    # ========================================================================

    # Evaluate brachistochrone ODE
    x_dot_computed, y_dot_computed, v_dot_computed = brachistochrone_ode(
        x_col, y_col, v_col, theta_col, g
    )

    # Stack state rates for defect computation
    # Shape: (num_col_nodes, 3) for the 3 states
    f_computed_x = x_dot_computed
    f_computed_y = y_dot_computed
    f_computed_v = v_dot_computed

    # ========================================================================
    # 4. Initial and Final State Values
    # ========================================================================

    # For Radau, initial and final values are at discretization nodes
    x_initial_val = x_disc[0:1]  # Shape (1,)
    y_initial_val = y_disc[0:1]
    v_initial_val = v_disc[0:1]

    x_final_val = x_disc[-1:]  # Shape (1,)
    y_final_val = y_disc[-1:]
    v_final_val = v_disc[-1:]

    # ========================================================================
    # 5. Radau Defect Computation
    # ========================================================================

    # Compute defects for each state separately
    # Each call returns: (rate_defect, initial_defect, final_defect, continuity_defect)

    defect_x = radau_defect(
        x_disc, f_computed_x, dt_dstau_col, D_matrix,
        x_initial_val, x_final_val, seg_end_indices
    )

    defect_y = radau_defect(
        y_disc, f_computed_y, dt_dstau_col, D_matrix,
        y_initial_val, y_final_val, seg_end_indices
    )

    defect_v = radau_defect(
        v_disc, f_computed_v, dt_dstau_col, D_matrix,
        v_initial_val, v_final_val, seg_end_indices
    )

    # Unpack defects
    rate_defect_x, init_defect_x, final_defect_x, cont_defect_x = defect_x
    rate_defect_y, init_defect_y, final_defect_y, cont_defect_y = defect_y
    rate_defect_v, init_defect_v, final_defect_v, cont_defect_v = defect_v

    # ========================================================================
    # 6. Assemble Residuals
    # ========================================================================

    residuals = {
        # Collocation defects (main dynamics constraints)
        'defect:x': rate_defect_x.ravel(),
        'defect:y': rate_defect_y.ravel(),
        'defect:v': rate_defect_v.ravel(),

        # Continuity defects across segments
        'continuity:x': cont_defect_x.ravel() if cont_defect_x.size > 0 else jnp.array([]),
        'continuity:y': cont_defect_y.ravel() if cont_defect_y.size > 0 else jnp.array([]),
        'continuity:v': cont_defect_v.ravel() if cont_defect_v.size > 0 else jnp.array([]),

        # Objective: minimize time
        'objective': t_duration,
    }

    # Add initial boundary constraints if enforced
    if enforce_initial.get('x', False):
        target = design_vars.get('x_initial', 0.0)
        residuals['initial:x'] = x_initial_val[0] - target

    if enforce_initial.get('y', False):
        target = design_vars.get('y_initial', 0.0)
        residuals['initial:y'] = y_initial_val[0] - target

    if enforce_initial.get('v', False):
        target = design_vars.get('v_initial', 0.0)
        residuals['initial:v'] = v_initial_val[0] - target

    # Add final boundary constraints if enforced
    if enforce_final.get('x', False):
        target = design_vars.get('x_final', 10.0)
        residuals['final:x'] = x_final_val[0] - target

    if enforce_final.get('y', False):
        target = design_vars.get('y_final', -10.0)
        residuals['final:y'] = y_final_val[0] - target

    if enforce_final.get('v', False):
        target = design_vars.get('v_final', 0.0)
        residuals['final:v'] = v_final_val[0] - target

    # Note: initial_defect and final_defect from radau_defect are typically
    # zero for Radau transcription, but included for completeness
    residuals['_debug_initial_defect_x'] = init_defect_x
    residuals['_debug_final_defect_x'] = final_defect_x

    return residuals


def create_radau_grid_data(num_segments, order=3):
    """
    Create grid data structure for Radau pseudospectral transcription.

    This is a helper function to generate the grid_data dictionary needed
    by radau_brachistochrone_phase.

    Parameters
    ----------
    num_segments : int
        Number of segments in the phase
    order : int, optional
        Polynomial order (number of LGR points per segment). Default: 3

    Returns
    -------
    grid_data : dict
        Grid data structure with all required fields

    Notes
    -----
    For Radau transcription:
    - Each segment has 'order' Legendre-Gauss-Radau (LGR) collocation points
    - Plus 1 discretization point at the segment start
    - Segment endpoints serve as discretization nodes
    - Total discretization nodes = num_segments + 1
    - Total collocation nodes = num_segments * order
    - Total all nodes = num_segments + 1 + num_segments * order

    This is a simplified version. Full implementation would use proper
    LGR node locations and compute exact Lagrange matrices.
    """
    from dymos.utils.lgl import lgl
    from dymos.utils.lagrange import lagrange_matrices
    import numpy as np
    from scipy.linalg import block_diag

    # Use NumPy for grid data creation (this is static, precomputed data)
    # Convert to JAX arrays only at the end
    segment_ends = np.linspace(-1.0, 1.0, num_segments + 1)
    nodes_per_segment = np.ones(num_segments, dtype=int) * (order + 1)

    # Get Legendre-Gauss-Radau nodes
    # LGR nodes are in (-1, 1] (includes right endpoint)
    lgr_nodes, _ = lgl(order + 1)  # Using LGL as approximation for now
    lgr_nodes = lgr_nodes[:-1]  # Remove left endpoint to get LGR-like

    # Build node_stau for all nodes
    # For each segment: discretization node at -1, then 'order' collocation nodes
    node_stau_list = []
    disc_indices_list = []
    col_indices_list = []
    node_counter = 0

    for seg_idx in range(num_segments):
        # Discretization node at segment start
        node_stau_list.append(-1.0)
        disc_indices_list.append(node_counter)
        node_counter += 1

        # Collocation nodes (LGR points, excluding left endpoint)
        for lgr_node in lgr_nodes:
            node_stau_list.append(lgr_node)
            col_indices_list.append(node_counter)
            node_counter += 1

    # Add final discretization node (at end of last segment)
    node_stau_list.append(1.0)
    disc_indices_list.append(node_counter)

    node_stau = np.array(node_stau_list)
    disc_indices = np.array(disc_indices_list)
    col_indices = np.array(col_indices_list)

    # Segment endpoint indices (for continuity)
    seg_end_indices = np.array([i * (order + 1) for i in range(num_segments + 1)])

    # Build Lagrange interpolation and differentiation matrices
    # From global discretization nodes to collocation nodes
    # For Radau, each segment's collocation nodes depend on the 2 boundary disc nodes
    num_disc = len(disc_indices)
    num_col = len(col_indices)

    # Discretization nodes at segment boundaries in segment coordinates
    disc_nodes_stau = np.array([-1.0, 1.0])

    # Build global matrices
    A_matrix = np.zeros((num_col, num_disc))
    D_matrix = np.zeros((num_col, num_disc))

    col_offset = 0
    for seg_idx in range(num_segments):
        # Collocation nodes for this segment (in segment coordinates)
        seg_col_nodes = lgr_nodes
        num_col_per_seg = len(seg_col_nodes)

        # Lagrange matrices from segment endpoints to collocation nodes
        A_seg, D_seg = lagrange_matrices(disc_nodes_stau, seg_col_nodes)

        # Place in global matrix: segment i uses disc nodes i and i+1
        A_matrix[col_offset:col_offset + num_col_per_seg, seg_idx:seg_idx + 2] = A_seg
        D_matrix[col_offset:col_offset + num_col_per_seg, seg_idx:seg_idx + 2] = D_seg

        col_offset += num_col_per_seg

    # Convert all arrays to JAX arrays for use in JAX functions
    grid_data = {
        'segment_ends': jnp.array(segment_ends),
        'nodes_per_segment': jnp.array(nodes_per_segment),
        'node_stau': jnp.array(node_stau),
        'disc_indices': jnp.array(disc_indices),
        'col_indices': jnp.array(col_indices),
        'seg_end_indices': jnp.array(seg_end_indices),
        'D_matrix': jnp.array(D_matrix),
        'A_matrix': jnp.array(A_matrix),
        'num_all_nodes': len(node_stau),
        'num_disc_nodes': len(disc_indices),
        'num_col_nodes': len(col_indices),
    }

    return grid_data
