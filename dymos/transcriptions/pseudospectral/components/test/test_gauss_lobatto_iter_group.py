import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos
from dymos.phase.options import StateOptionsDictionary, TimeOptionsDictionary
from dymos.transcriptions.grid_data import GaussLobattoGrid
from dymos.transcriptions.pseudospectral.components import GaussLobattoIterGroup
from dymos.utils.misc import GroupWrapperConfig
from dymos.utils.testing_utils import SimpleODE, _PhaseStub

GaussLobattoIterGroup = GroupWrapperConfig(GaussLobattoIterGroup, [_PhaseStub()])


# @use_tempdirs
class TestGaussLobattoIterGroup(unittest.TestCase):

    def test_solve_segments(self):
        with dymos.options.temporary(include_check_partials=True):
            for direction in ['forward', 'backward']:
                for compressed in [False, True]:
                    with self.subTest(msg=f'{direction=} {compressed=}'):

                        state_options = {'x': StateOptionsDictionary()}

                        state_options['x']['shape'] = (1,)
                        state_options['x']['units'] = 's**2'
                        state_options['x']['targets'] = ['x']
                        state_options['x']['initial_bounds'] = (None, None)
                        state_options['x']['final_bounds'] = (None, None)
                        state_options['x']['solve_segments'] = direction
                        state_options['x']['rate_source'] = 'x_dot'

                        time_options = TimeOptionsDictionary()
                        grid_data = GaussLobattoGrid(num_segments=7, nodes_per_seg=5, compressed=compressed)
                        nn = grid_data.subset_num_nodes['all']
                        ode_class = SimpleODE

                        p = om.Problem()

                        times = grid_data.node_ptau + 1
                        time_comp = p.model.add_subsystem('time_comp', om.IndepVarComp())
                        time_comp.add_output('t', val=times, units='s')

                        p.model.add_subsystem('iter_group', GaussLobattoIterGroup(state_options=state_options,
                                                                                  time_options=time_options,
                                                                                  grid_data=grid_data,
                                                                                  ode_class=ode_class))

                        p.model.connect('time_comp.t', 'iter_group.ode_disc.t',
                                        src_indices=grid_data.subset_node_indices['state_disc'])

                        p.model.connect('time_comp.t', 'iter_group.ode_col.t',
                                        src_indices=grid_data.subset_node_indices['col'])

                        iter_group = p.model._get_subsystem('iter_group')

                        iter_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                        iter_group.linear_solver = om.DirectSolver()

                        p.setup(force_alloc_complex=True)

                        p.final_setup()
                        solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))
                        dsolution_dt = np.reshape(2 * times + 2 - 0.5 * np.exp(times), (nn, 1))

                        # Each segment is of the same length, so dt_dstau is constant.
                        # dt_dstau is (tf - t0) / 2.0 / num_seg
                        p.set_val('iter_group.dt_dstau', (times[-1] / 2.0 / grid_data.num_segments))

                        if direction == 'forward':
                            p.set_val('iter_group.initial_states:x', 0.5)
                        else:
                            p.set_val('iter_group.final_states:x', solution[-1])

                        p.set_val('iter_group.states:x', 0.0)
                        p.set_val('iter_group.ode_disc.t', times[grid_data.subset_node_indices['state_disc']])
                        p.set_val('iter_group.ode_disc.p', 1.0)
                        p.set_val('iter_group.ode_col.t', times[grid_data.subset_node_indices['col']])
                        p.set_val('iter_group.ode_col.p', 1.0)

                        p.model.run_apply_nonlinear()

                        with np.printoptions(linewidth=10000, edgeitems=1000):
                            cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
                        assert_check_partials(cpd)

                        p.set_solver_print(level=2)
                        p.run_model()

                        x_i = p.get_val('iter_group.states:x')
                        xdot_i = p.get_val('iter_group.ode_disc.x_dot')

                        x_c = p.get_val('iter_group.state_col:x')
                        xdot_c = p.get_val('iter_group.ode_col.x_dot')

                        x_0 = p.get_val('iter_group.initial_states:x')
                        x_f = p.get_val('iter_group.final_states:x')

                        c_idxs = grid_data.subset_node_indices['col']
                        i_idxs = grid_data.subset_node_indices['state_input']
                        d_idxs = grid_data.subset_node_indices['state_disc']
                        assert_near_equal(solution[i_idxs], x_i, tolerance=1.0E-5)
                        assert_near_equal(solution[c_idxs], x_c, tolerance=1.0E-5)
                        assert_near_equal(dsolution_dt[d_idxs, 0], xdot_i, tolerance=1.0E-5)
                        assert_near_equal(dsolution_dt[c_idxs, 0], xdot_c, tolerance=1.0E-5)
                        assert_near_equal(solution[np.newaxis, 0], x_0, tolerance=1.0E-7)
                        assert_near_equal(solution[np.newaxis, -1], x_f, tolerance=1.0E-7)


if __name__ == '__main__':
    unittest.main()
