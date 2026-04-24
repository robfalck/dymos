"""Tests for GaussLobattoIterGroup."""
import unittest

import numpy as np

import openmdao.api as om

import dymos
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.misc import GroupWrapperConfig
from dymos.transcriptions.pseudospectral.components.gauss_lobatto_iter_group import (
    GaussLobattoIterGroup,
)
from dymos.phase.options import StateOptionsDictionary, TimeOptionsDictionary
from dymos.transcriptions.grid_data import GaussLobattoGrid
from dymos.utils.testing_utils import PhaseStub, SimpleODE, SimpleVectorizedODE


GaussLobattoIterGroup = GroupWrapperConfig(GaussLobattoIterGroup, [PhaseStub()])


@use_tempdirs
class TestGaussLobattoIterGroup(unittest.TestCase):

    def _make_state_options(self, name='x', shape=(1,), units='s**2',
                            targets=None, rate_source='x_dot', solve_segments=False):
        """Build a minimal StateOptionsDictionary for testing."""
        state_options = {name: StateOptionsDictionary()}
        state_options[name]['shape'] = shape
        state_options[name]['units'] = units
        state_options[name]['targets'] = targets if targets is not None else [name]
        state_options[name]['initial_bounds'] = (None, None)
        state_options[name]['final_bounds'] = (None, None)
        state_options[name]['solve_segments'] = solve_segments
        state_options[name]['rate_source'] = rate_source
        state_options[name]['opt'] = False  # no design vars in standalone test context
        return state_options

    def test_nlbgs_algebraic_loop(self):
        """Verify the NLBGS resolves the algebraic loop: col states are Hermite-consistent."""
        with dymos.options.temporary(include_check_partials=True):
            for compressed in [True, False]:
                with self.subTest(msg=f'{compressed=}'):

                    state_options = self._make_state_options(
                        name='x', shape=(1,), units='s**2',
                        targets=['x'], rate_source='x_dot', solve_segments=False)

                    time_options = TimeOptionsDictionary()
                    grid_data = GaussLobattoGrid(num_segments=2, nodes_per_seg=5,
                                                 compressed=compressed)
                    nn = grid_data.num_nodes
                    ode_class = SimpleODE

                    p = om.Problem()
                    p.model.add_subsystem(
                        'gl',
                        GaussLobattoIterGroup(state_options=state_options,
                                              time_options=time_options,
                                              grid_data=grid_data,
                                              ode_class=ode_class))

                    gl_grp = p.model._get_subsystem('gl')
                    gl_grp.nonlinear_solver = om.NonlinearBlockGS(maxiter=10, iprint=0)

                    p.setup(force_alloc_complex=True)

                    # Map ptau ∈ [-1, 1] to times ∈ [0, 2]
                    times = grid_data.node_ptau + 1.0
                    # Analytical solution: x(t) = t^2 + 2t + 1 - 0.5*exp(t)
                    solution = times ** 2 + 2 * times + 1 - 0.5 * np.exp(times)
                    solution = solution.reshape((nn, 1))

                    # dt_dstau at all nodes: (tf - t0)/2 * d(ptau)/d(stau)
                    dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

                    disc_idxs = grid_data.subset_node_indices['state_disc']
                    col_idxs = grid_data.subset_node_indices['col']
                    n_input = grid_data.subset_num_nodes['state_input']
                    n_disc = grid_data.subset_num_nodes['state_disc']

                    p.set_val('gl.dt_dstau', dt_dstau_all)
                    p.set_val('gl.states:x', solution[disc_idxs[:n_input]])
                    p.set_val('gl.ode_all.t', times)
                    p.set_val('gl.ode_all.p', 1.0)

                    p.run_model()

                    # After NLBGS, col states must be consistent with Hermite interpolation:
                    # col_val = Ai @ xd + Bi @ fd_disc * dt_dstau_col
                    # Verify this algebraic consistency, not convergence to analytical solution.
                    from scipy.sparse import csr_matrix
                    Ai, Bi, _, _ = grid_data.phase_hermite_matrices('state_disc', 'col',
                                                                     sparse=True)
                    states_all = p.get_val('gl.states_all:x')
                    xd = states_all[disc_idxs, 0]   # disc values from passthrough
                    fd = p.get_val('gl.ode_all.x_dot')[disc_idxs]  # disc ODE rates
                    dt_col = dt_dstau_all[col_idxs]  # dt_dstau at col nodes

                    col_expected = Ai.dot(xd) + Bi.dot(fd) * dt_col
                    assert_near_equal(states_all[col_idxs, 0], col_expected,
                                      tolerance=1.0e-10)

    def test_picard_check_partials(self):
        """Verify partial derivatives for GaussLobattoIterGroup without solve_segments."""
        with dymos.options.temporary(include_check_partials=True):

            state_options = self._make_state_options(
                name='x', shape=(1,), units='s**2',
                targets=['x'], rate_source='x_dot', solve_segments=False)

            time_options = TimeOptionsDictionary()
            grid_data = GaussLobattoGrid(num_segments=2, nodes_per_seg=3, compressed=True)
            nn = grid_data.num_nodes
            ode_class = SimpleODE

            p = om.Problem()
            p.model.add_subsystem(
                'gl',
                GaussLobattoIterGroup(state_options=state_options,
                                      time_options=time_options,
                                      grid_data=grid_data,
                                      ode_class=ode_class))

            gl_grp = p.model._get_subsystem('gl')
            gl_grp.nonlinear_solver = om.NonlinearBlockGS(maxiter=10, iprint=0)

            p.setup(force_alloc_complex=True)

            times = grid_data.node_ptau + 1.0
            solution = times ** 2 + 2 * times + 1 - 0.5 * np.exp(times)
            dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

            disc_idxs = grid_data.subset_node_indices['state_disc']
            n_input = grid_data.subset_num_nodes['state_input']

            p.set_val('gl.dt_dstau', dt_dstau_all)
            p.set_val('gl.states:x', solution[disc_idxs[:n_input]].reshape(-1, 1))
            p.set_val('gl.ode_all.t', times)
            p.set_val('gl.ode_all.p', 1.0)

            p.run_model()

            cpd = p.check_partials(method='cs', compact_print=False,
                                   show_only_incorrect=True, out_stream=None)
            assert_check_partials(cpd, atol=1.0e-5, rtol=1.0)

    def test_picard_vector_states(self):
        """Verify Picard convergence for vector-valued states."""
        with dymos.options.temporary(include_check_partials=True):

            state_options = self._make_state_options(
                name='z', shape=(2,), units='s**2',
                targets=['z'], rate_source='z_dot', solve_segments=False)

            time_options = TimeOptionsDictionary()
            grid_data = GaussLobattoGrid(num_segments=2, nodes_per_seg=5, compressed=True)
            nn = grid_data.num_nodes
            ode_class = SimpleVectorizedODE

            p = om.Problem()
            p.model.add_subsystem(
                'gl',
                GaussLobattoIterGroup(state_options=state_options,
                                      time_options=time_options,
                                      grid_data=grid_data,
                                      ode_class=ode_class))

            gl_grp = p.model._get_subsystem('gl')
            gl_grp.nonlinear_solver = om.NonlinearBlockGS(maxiter=10, iprint=0)

            p.setup(force_alloc_complex=True)

            times = grid_data.node_ptau + 1.0
            dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

            disc_idxs = grid_data.subset_node_indices['state_disc']
            n_input = grid_data.subset_num_nodes['state_input']

            # Simple non-trivial initial condition
            solution_z0 = (times ** 2 + 2 * times + 1 - 0.5 * np.exp(times)).reshape(nn, 1)
            solution_z1 = (5 * times).reshape(nn, 1)
            solution = np.hstack([solution_z0, solution_z1])

            p.set_val('gl.dt_dstau', dt_dstau_all)
            p.set_val('gl.states:z', solution[disc_idxs[:n_input]])
            p.set_val('gl.ode_all.t', times)
            p.set_val('gl.ode_all.p', 1.0)

            p.run_model()

            cpd = p.check_partials(method='cs', compact_print=False,
                                   show_only_incorrect=True, out_stream=None)
            assert_check_partials(cpd, atol=1.0e-5, rtol=1.0)


if __name__ == '__main__':
    unittest.main()
