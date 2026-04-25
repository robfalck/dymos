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


def _make_state_options(name='x', shape=(1,), units='s**2',
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
    return state_options


@use_tempdirs
class TestGaussLobattoIterGroup(unittest.TestCase):

    def test_solve_segments(self):
        """Test forward/backward solve_segments with scalar state and analytical solution."""
        with dymos.options.temporary(include_check_partials=True):
            for direction in ['forward', 'backward']:
                for compressed in [True, False]:
                    with self.subTest(msg=f'{direction=} {compressed=}'):

                        state_options = _make_state_options(
                            name='x', shape=(1,), units='s**2',
                            targets=['x'], rate_source='x_dot',
                            solve_segments=direction)

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
                        gl_grp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                                   maxiter=100, iprint=-1)
                        gl_grp.linear_solver = om.DirectSolver()

                        p.setup(force_alloc_complex=True)

                        # Map ptau ∈ [-1, 1] → times ∈ [0, 2]
                        times = grid_data.node_ptau + 1.0
                        # Analytical solution: x(t) = t^2 + 2t + 1 - 0.5*exp(t), x(0) = 0.5
                        solution = (times**2 + 2*times + 1 - 0.5*np.exp(times)).reshape(nn, 1)

                        dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

                        state_input_idxs = grid_data.subset_node_indices['state_input'] 

                        p.set_val('gl.dt_dstau', dt_dstau_all)
                        p.set_val('gl.states:x', 0.0)
                        p.set_val('gl.states_all:x', 0.0)
                        p.set_val('gl.ode.t', times)
                        p.set_val('gl.ode.p', 1.0)

                        if direction == 'forward':
                            p.set_val('gl.initial_states:x', 0.5)
                        else:
                            p.set_val('gl.final_states:x', solution[-1])

                        p.run_model()

                        x = p.get_val('gl.states:x')
                        x_0 = p.get_val('gl.initial_states:x')
                        x_f = p.get_val('gl.final_states:x')

                        assert_near_equal(solution[state_input_idxs], x, tolerance=1.0E-5)
                        assert_near_equal(solution[np.newaxis, 0], x_0, tolerance=1.0E-7)
                        assert_near_equal(solution[np.newaxis, -1], x_f, tolerance=1.0E-7)

                        cpd = p.check_partials(method='cs', compact_print=False, out_stream=None)
                        assert_check_partials(cpd)

    def test_solve_segments_vector_states(self):
        """Test forward/backward solve_segments with vector-valued states."""
        with dymos.options.temporary(include_check_partials=True):
            for direction in ['forward', 'backward']:
                with self.subTest(msg=f'{direction=}'):

                    state_options = _make_state_options(
                        name='z', shape=(2,), units='s**2',
                        targets=['z'], rate_source='z_dot',
                        solve_segments=direction)

                    time_options = TimeOptionsDictionary()
                    grid_data = GaussLobattoGrid(num_segments=5, nodes_per_seg=3,
                                                 compressed=False)
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
                    gl_grp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                               maxiter=100, iprint=-1,
                                                               atol=1.0E-10, rtol=1.0E-10)
                    gl_grp.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
                        bound_enforcement='vector')
                    gl_grp.linear_solver = om.DirectSolver()

                    p.setup(force_alloc_complex=True)

                    times = grid_data.node_ptau + 1.0
                    solution = (times**2 + 2*times + 1 - 0.5*np.exp(times)).reshape(nn, 1)

                    dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

                    disc_idxs = grid_data.subset_node_indices['state_disc']
                    n_input = grid_data.subset_num_nodes['state_input']

                    p.set_val('gl.dt_dstau', dt_dstau_all)
                    p.set_val('gl.states:z', 0.0)
                    p.set_val('gl.states_all:z', 0.0)
                    p.set_val('gl.ode.t', times)
                    p.set_val('gl.ode.p', 1.0)

                    if direction == 'forward':
                        p.set_val('gl.initial_states:z', [0.5, 0.0])
                    else:
                        p.set_val('gl.final_states:z', solution[-1], indices=om.slicer[:, 0])
                        p.set_val('gl.final_states:z', 20.0, indices=om.slicer[:, 1])

                    p.run_model()

                    x = p.get_val('gl.states:z')
                    x_0 = p.get_val('gl.initial_states:z')
                    x_f = p.get_val('gl.final_states:z')

                    assert_near_equal(solution[disc_idxs[:n_input]], x[np.newaxis, :, 0].T,
                                      tolerance=1.0E-4)
                    assert_near_equal(solution[0], x_0[:, 0], tolerance=1.0E-4)
                    assert_near_equal(solution[-1], x_f[:, 0], tolerance=1.0E-4)

                    with np.printoptions(linewidth=1024, edgeitems=1024):
                        cpd = p.check_partials(method='cs', compact_print=False,
                                               show_only_incorrect=True, out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0)

    def test_autoivc_no_solve(self):
        """Test solve_segments=False: feed exact solution, verify residuals and partials."""
        with dymos.options.temporary(include_check_partials=True):
            for compressed in [True, False]:
                with self.subTest(msg=f'{compressed=}'):

                    state_options = _make_state_options(
                        name='x', shape=(1,), units='s**2',
                        targets=['x'], rate_source='x_dot',
                        solve_segments=False)

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

                    times = grid_data.node_ptau + 1.0
                    solution = (times**2 + 2*times + 1 - 0.5*np.exp(times)).reshape(nn, 1)

                    dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

                    disc_idxs = grid_data.subset_node_indices['state_disc']
                    n_input = grid_data.subset_num_nodes['state_input']

                    p.set_val('gl.dt_dstau', dt_dstau_all)
                    p.set_val('gl.states:x', solution[disc_idxs[:n_input]])
                    p.set_val('gl.states_all:x', solution)
                    p.set_val('gl.initial_states:x', solution[np.newaxis, 0])
                    p.set_val('gl.final_states:x', solution[np.newaxis, -1])
                    p.set_val('gl.ode.t', times)
                    p.set_val('gl.ode.p', 1.0)

                    p.run_model()

                    x = p.get_val('gl.states:x')
                    x_0 = p.get_val('gl.initial_states:x')
                    x_f = p.get_val('gl.final_states:x')

                    assert_near_equal(solution[disc_idxs[:n_input]], x, tolerance=1.0E-5)
                    assert_near_equal(solution[np.newaxis, 0], x_0, tolerance=1.0E-7)
                    assert_near_equal(solution[np.newaxis, -1], x_f, tolerance=1.0E-7)

                    cpd = p.check_partials(method='cs', compact_print=False, out_stream=None)
                    assert_check_partials(cpd)

    def test_nlbgs_algebraic_loop(self):
        """Verify NLBGS resolves the algebraic loop: col states are Hermite-consistent."""
        with dymos.options.temporary(include_check_partials=True):
            for compressed in [True, False]:
                with self.subTest(msg=f'{compressed=}'):

                    state_options = _make_state_options(
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

                    times = grid_data.node_ptau + 1.0
                    solution = (times**2 + 2*times + 1 - 0.5*np.exp(times)).reshape(nn, 1)

                    dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

                    disc_idxs = grid_data.subset_node_indices['state_disc']
                    col_idxs = grid_data.subset_node_indices['col']
                    n_input = grid_data.subset_num_nodes['state_input']

                    p.set_val('gl.dt_dstau', dt_dstau_all)
                    p.set_val('gl.states:x', solution[disc_idxs[:n_input]])
                    p.set_val('gl.states_all:x', solution)
                    p.set_val('gl.ode.t', times)
                    p.set_val('gl.ode.p', 1.0)

                    p.run_model()

                    # Verify col states are consistent with Hermite interpolation from disc values.
                    Ai, Bi, _, _ = grid_data.phase_hermite_matrices('state_disc', 'col',
                                                                     sparse=True)
                    states_all = p.get_val('gl.states_all:x')
                    xd = states_all[disc_idxs, 0]
                    fd = p.get_val('gl.ode.x_dot')[disc_idxs]
                    dt_col = dt_dstau_all[col_idxs]

                    col_expected = Ai.dot(xd) + Bi.dot(fd) * dt_col
                    assert_near_equal(states_all[col_idxs, 0], col_expected, tolerance=1.0e-10)

    def test_picard_check_partials(self):
        """Verify partial derivatives for GaussLobattoIterGroup without solve_segments."""
        with dymos.options.temporary(include_check_partials=True):

            state_options = _make_state_options(
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
            solution = (times**2 + 2*times + 1 - 0.5*np.exp(times)).reshape(nn, 1)
            dt_dstau_all = (times[-1] - times[0]) / 2.0 * grid_data.node_dptau_dstau

            disc_idxs = grid_data.subset_node_indices['state_disc']
            n_input = grid_data.subset_num_nodes['state_input']

            p.set_val('gl.dt_dstau', dt_dstau_all)
            p.set_val('gl.states:x', solution[disc_idxs[:n_input]])
            p.set_val('gl.states_all:x', solution)
            p.set_val('gl.ode.t', times)
            p.set_val('gl.ode.p', 1.0)

            p.run_model()

            cpd = p.check_partials(method='cs', compact_print=False,
                                   show_only_incorrect=True, out_stream=None)
            assert_check_partials(cpd, atol=1.0e-5, rtol=1.0)

    def test_picard_vector_states(self):
        """Verify Picard convergence for vector-valued states without solve_segments."""
        with dymos.options.temporary(include_check_partials=True):

            state_options = _make_state_options(
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

            solution_z0 = (times**2 + 2*times + 1 - 0.5*np.exp(times)).reshape(nn, 1)
            solution_z1 = (5*times).reshape(nn, 1)
            solution = np.hstack([solution_z0, solution_z1])

            p.set_val('gl.dt_dstau', dt_dstau_all)
            p.set_val('gl.states:z', solution[disc_idxs[:n_input]])
            p.set_val('gl.states_all:z', solution)
            p.set_val('gl.ode.t', times)
            p.set_val('gl.ode.p', 1.0)

            p.run_model()

            cpd = p.check_partials(method='cs', compact_print=False,
                                   show_only_incorrect=True, out_stream=None)
            assert_check_partials(cpd, atol=1.0e-5, rtol=1.0)


if __name__ == '__main__':
    unittest.main()
