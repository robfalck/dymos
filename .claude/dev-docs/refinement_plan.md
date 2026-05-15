# Birkhoff Grid Refinement — Developer Reference

## Overview

Birkhoff is a single-segment, global-polynomial transcription. Refinement means
increasing `num_nodes` (the polynomial degree). There is no h-refinement (no
segment splitting).

## Error Estimation Algorithm

### Why not re-run the ODE?

After a Birkhoff solve, state values x_i and ODE rates f_i = dx/dt|_{t_i} are
already available at all N nodes. A Hermite polynomial through these 2N pieces
of data has degree 2N-1 — far more accurate than the N-1 degree Lagrange polynomial
through state values alone. Their difference at midpoints is an O(h^N) error
indicator with zero extra ODE evaluations.

### Lagrange interpolation at midpoints

Given N nodes tau_0 < tau_1 < ... < tau_{N-1} in stau in [-1,1]:

    midpoints m_k = (tau_k + tau_{k+1}) / 2,  k = 0,...,N-2

Build L_mid (shape N-1 x N) via `lagrange_matrices(nodes, midpoints)`:

    x_lagrange[k] = sum_i L_mid[k,i] * x[i]

### Hermite interpolation at midpoints (no extra ODE calls)

The Hermite polynomial H_{2N-1} passing through (tau_i, x_i) with slope
xdot_stau_i = f_i * t_duration/2 (converting dx/dt to dx/dtau) can be written as:

    H(m) = sum_i h_i(m) * x_i  +  sum_i H_i(m) * xdot_stau_i

where (using the Lagrange basis L_i(tau) and diagonal of the differentiation
matrix at nodes, d_i = dL_i/dtau|_{tau_i}):

    h_i(m) = (1 - 2*d_i*(m - tau_i)) * L_i(m)^2
    H_i(m) = (m - tau_i) * L_i(m)^2

Build matrices H_y and H_yd (each N-1 x N) from L_mid and diag(D_nodes):

    H_y[k,i]  = (1 - 2*diag(D)[i]*(midpoints[k] - nodes[i])) * L_mid[k,i]^2
    H_yd[k,i] = (midpoints[k] - nodes[i]) * L_mid[k,i]^2

Then:

    x_hermite = H_y @ x + H_yd @ (f * t_duration/2)

These three matrices are computed once per phase in `_hermite_lagrange_matrices()`
in `error_estimation.py`.

### Error indicator

    abs_error = |x_hermite - x_lagrange|            # shape (N-1, *state_shape)
    rel_error = max(abs_error) / (1 + max(|x|))     # relative, per Patterson 2015

If the maximum over all states exceeds `phase.refine_options['tolerance']`,
refinement is needed.

### Error order

The indicator detects O(h^N) error where h ~ 2/(N-1) is the node spacing. This
aligns with the Lagrange interpolation error for a degree-(N-1) polynomial through
N nodes. The Hermite interpolant contributes O(h^{2N}), which is negligible.
For global polynomial (spectral) methods, h decreases as N grows, so convergence
is spectral: the error indicator drops very rapidly once N is large enough.

## Refinement Strategy

If refinement is needed:

    delta_N = max(5, ceil(log(error / tolerance)))
    new_N   = min(current_N + delta_N, max_order)

The `log` term gives a rough estimate based on the magnitude of the violation;
the floor of 5 ensures progress even for mild violations.

`max_order` in `refine_options` acts as the node count cap. The default is 14,
which is too small for most Birkhoff problems — users should raise it (e.g.,
50-100) when using Birkhoff with grid refinement.

## Variable Access Paths

After a Birkhoff solve, at the phase level:

    phase.get_val('states:{name}')       # (N, *state_shape), state values
    phase.get_val('state_rates:{name}')  # (N, *state_shape), dx/dt
    tx.grid_data.node_stau               # (N,), node positions in [-1, 1]
    phase.get_val('t_duration')[0]       # scalar, physical time duration

Both `states:` and `state_rates:` are promoted to the phase level from the
Birkhoff iter group (see `birkhoff_iter_group.py` and `birkhoff_defect_comp.py`).
State rates are design variables (when `solve_segments=False`, the default).

## Key Code Locations

    dymos/grid_refinement/error_estimation.py
        _hermite_lagrange_matrices()   — builds L_mid, H_y, H_yd
        _check_error_birkhoff()        — per-phase Birkhoff error check
        check_error()                  — dispatches to Birkhoff branch

    dymos/grid_refinement/birkhoff_adaptive/birkhoff_adaptive.py
        BirkhoffAdaptive               — computes new num_nodes and calls tx.init_grid()

    dymos/grid_refinement/refinement.py
        _refine_iter()                 — splits phases by type; calls both refiners

    dymos/transcriptions/grid_data.py
        BirkhoffGrid                   — always num_segments=1; transcription_order=num_nodes

    dymos/transcriptions/pseudospectral/birkhoff.py
        Birkhoff.init_grid()           — creates BirkhoffGrid(num_nodes, grid_type)

## Integration with the Refinement Loop

`_refine_iter()` splits `phases` into Birkhoff and non-Birkhoff dicts:

    birkhoff_phases = phases where gd.transcription == 'birkhoff'
    other_phases    = all remaining phases

`check_error()` handles both: the Birkhoff branch calls `_check_error_birkhoff()`
and skips the `eval_ode_on_grid` + `compute_state_quadratures` path entirely.

`BirkhoffAdaptive` and the hp/ph refiner are called independently on their
respective phase subsets. Both populate `refine_results` with the standard keys:
`new_order`, `new_num_segments`, `new_segment_ends`.
