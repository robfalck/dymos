import numpy as np


class BirkhoffAdaptive:
    """
    Grid refinement object for the Birkhoff transcription.

    Birkhoff is a single-segment, global-polynomial method.  Refinement increases
    the number of nodes (polynomial degree).  Error is estimated by comparing a
    Lagrange interpolant (state values only) against a Hermite interpolant (state
    values + ODE rates) at midpoints between nodes; see check_error() in
    error_estimation.py for details.

    Parameters
    ----------
    phases : dict
        Dict mapping phase paths to Phase objects.
    """

    def __init__(self, phases):
        self.phases = phases

    def refine(self, refine_results, iter_number):
        """
        Increase num_nodes for Birkhoff phases that require refinement.

        Parameters
        ----------
        refine_results : dict
            A dictionary where each key is the path to a phase in the problem, and the
            associated value are various properties of that phase needed by the refinement
            algorithm.  refine_results is returned by check_error.  This method modifies it
            in place, adding the new_num_segments, new_order, and new_segment_ends.
        iter_number : int
            Current iteration of the grid refinement (unused; present for interface parity).
        """
        for phase_path, results in refine_results.items():
            phase = self.phases[phase_path]
            tx = phase.options['transcription']
            gd = tx.grid_data

            need_refine = results['need_refinement']
            if not phase.refine_options['refine'] or not np.any(need_refine):
                results['new_order'] = gd.transcription_order
                results['new_num_segments'] = 1
                results['new_segment_ends'] = gd.segment_ends
                continue

            max_order = phase.refine_options['max_order']
            error = results['max_rel_error'][0]
            tol = phase.refine_options['tolerance']
            current_N = tx.options['num_nodes']

            # Estimate increment: at least 5; larger when error significantly exceeds tol
            delta_N = max(5, int(np.ceil(np.log(error / tol))))
            new_N = min(current_N + delta_N, max_order)

            tx.options['num_nodes'] = new_N
            tx.init_grid()

            results['new_order'] = np.array([new_N])
            results['new_num_segments'] = 1
            results['new_segment_ends'] = gd.segment_ends
