import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.pseudospectral.components.birkhoff_collocation_comp import BirkhoffCollocationComp
from dymos.transcriptions.grid_data import BirkhoffGrid

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
CollocationComp = CompWrapperConfig(BirkhoffCollocationComp)


class TestCollocationComp(unittest.TestCase):

    def make_problem(self, grid_type='lgl'):
        dm.options['include_check_partials'] = True

        gd = BirkhoffGrid(num_segments=1, segment_ends=np.array([0., 10.]),
                          nodes_per_seg=21, grid_type='lgl')
        n = gd.subset_num_nodes['col']
        tau = gd.node_stau
        t = 5 * tau + 5

        self.p = om.Problem(model=om.Group())

        state_options = {'x': {'units': 'm', 'shape': (1,), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': False,
                               'input_initial': False},
                         'y': {'units': 'm', 'shape': (2, 2), 'fix_initial': False,
                               'fix_final': True, 'solve_segments': False,
                               'input_initial': False}}

        indep_comp = om.IndepVarComp()
        self.p.model.add_subsystem('indep', indep_comp, promotes_outputs=['*'])

        # Testing the basic ODE xdot = -x, x(0) = 10
        # Solution is x(t) = 10*exp(-t)

        x_val = 10*np.exp(-t)
        y_val = np.zeros((n, 2, 2))
        y_val[:, 0, 0] = x_val
        y_val[:, 1, 1] = x_val

        indep_comp.add_output(
            'dt_dstau',
            val=np.ones(n)*5, units='s')
        indep_comp.add_output(
            'state_value:x',
            val=x_val, units='m')
        indep_comp.add_output(
            'f_value:x',
            val=-x_val*5, units='m')
        indep_comp.add_output(
            'f_computed:x',
            val=-x_val, units='m/s')
        indep_comp.add_output(
            'state_value:y',
            val=y_val, units='m')
        indep_comp.add_output(
            'f_value:y',
            val=-y_val * 5, units='m')
        indep_comp.add_output(
            'f_computed:y',
            val=-y_val, units='m/s')

        self.p.model.add_subsystem('defect_comp',
                                   subsys=CollocationComp(grid_data=gd,
                                                          state_options=state_options,
                                                          time_units='s'))

        if grid_type == 'radau-ps':
            src_indices = om.slicer[:-1]
        else:
            src_indices = om.slicer[:]

        self.p.model.connect('state_value:x', 'defect_comp.states:x', src_indices=src_indices)
        self.p.model.connect('f_value:x', 'defect_comp.state_rates:x', src_indices=src_indices)
        self.p.model.connect('state_value:y', 'defect_comp.states:y')
        self.p.model.connect('f_value:y', 'defect_comp.state_rates:y')
        self.p.model.connect('f_computed:x', 'defect_comp.f_computed:x', src_indices=src_indices)
        self.p.model.connect('f_computed:y', 'defect_comp.f_computed:y')
        self.p.model.connect('dt_dstau', 'defect_comp.dt_dstau')

        self.p.setup(force_alloc_complex=True)

        self.p.set_val('defect_comp.initial_states:x', 10.0)
        self.p.set_val('defect_comp.final_states:x', x_val[-1])

        self.p.set_val('defect_comp.initial_states:y', np.array([[10.0, 0.0], [0.0, 10.0]]))
        self.p.set_val('defect_comp.final_states:y', np.array([[x_val[-1], 0.0], [0.0, x_val[-1]]]))

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results_lgr_grid(self):
        self.make_problem(grid_type='lgr')
        assert_almost_equal(self.p['defect_comp.state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_defects:y'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:y'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:y'], 0.0)

    def test_results_lgl_grid(self):
        self.make_problem(grid_type='lgl')
        assert_almost_equal(self.p['defect_comp.state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_defects:y'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:y'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:y'], 0.0)

    def test_results_cgl_grid(self):
        self.make_problem(grid_type='cgl')
        assert_almost_equal(self.p['defect_comp.state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_defects:y'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:y'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:y'], 0.0)

    def test_partials(self):
        self.make_problem(grid_type='lgl')
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='fd')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()