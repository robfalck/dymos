"""Tests for GaussLobattoInterpComp."""
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import dymos as dm
from dymos.utils.misc import CompWrapperConfig
from dymos.transcriptions.pseudospectral.components.gauss_lobatto_interp_comp import (
    GaussLobattoInterpComp,
)

GaussLobattoInterpComp = CompWrapperConfig(GaussLobattoInterpComp)


# Analytical functions used as test states (quadratic and cubic polynomials)
def x(t):
    return t ** 2


def f_x(t):
    return 2.0 * t


def v(t):
    return t ** 3 - 10.0 * t ** 2


def f_v(t):
    return 3.0 * t ** 2 - 20.0 * t


def _make_dt_dstau(gd, segends):
    """Compute dt/d(stau) at collocation nodes from grid and segment endpoints.

    For each col node in segment j:
        dt_dstau = (segends[j+1] - segends[j]) / 2
    This equals (tf - t0)/2 * d(ptau)/d(stau) at each node.
    """
    col_idxs = gd.subset_node_indices['col']
    t0 = segends[0]
    tf = segends[-1]
    dt_dstau = (tf - t0) / 2.0 * gd.node_dptau_dstau[col_idxs]
    return dt_dstau


class TestGaussLobattoInterpComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def _make_problem(self, gd, states):
        """Build a Problem containing only GaussLobattoInterpComp."""
        p = om.Problem()

        n_disc = gd.subset_num_nodes['state_disc']
        n_col = gd.subset_num_nodes['col']

        ivc = om.IndepVarComp()
        for name, opts in states.items():
            shape = opts['shape']
            units = opts['units']
            rate_units = opts.get('rate_units', f'{units}/s')
            ivc.add_output(f'state_disc:{name}', val=np.zeros((n_disc,) + shape), units=units)
            ivc.add_output(f'staterate_disc:{name}',
                           val=np.zeros((n_disc,) + shape), units=rate_units)

        ivc.add_output('dt_dstau', val=np.ones(n_col), units='s')

        p.model.add_subsystem('ivc', ivc, promotes=['*'])
        p.model.add_subsystem(
            'interp',
            subsys=GaussLobattoInterpComp(grid_data=gd, state_options=states,
                                             time_units='s'),
            promotes=['*'])

        p.setup(force_alloc_complex=True)
        return p

    def test_states_all_disc_passthrough(self):
        """Disc-node values in states_all must equal the state_disc input exactly."""
        segends = np.array([0.0, 3.0, 10.0])
        gd = dm.GaussLobattoGrid(num_segments=2, nodes_per_seg=3, segment_ends=segends)

        states = {'x': {'units': 'm', 'shape': (1,), 'rate_units': 'm/s'},
                  'v': {'units': 'm/s', 'shape': (1,), 'rate_units': 'm/s**2'}}

        p = self._make_problem(gd, states)

        disc_idxs = gd.subset_node_indices['state_disc']
        t0, tf = segends[0], segends[-1]
        t_all = (tf - t0) / 2.0 * (gd.node_ptau + 1.0) + t0

        xd = x(t_all[disc_idxs]).reshape(-1, 1)
        p.set_val('state_disc:x', xd)
        p.set_val('staterate_disc:x', f_x(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('dt_dstau', _make_dt_dstau(gd, segends))
        p.run_model()

        states_all = p.get_val('states_all:x')
        assert_almost_equal(states_all[disc_idxs], xd, decimal=12)

    def test_hermite_interpolation_accuracy(self):
        """Hermite interpolation should be exact for polynomials within the interpolation degree."""
        segends = np.array([0.0, 3.0, 10.0])
        gd = dm.GaussLobattoGrid(num_segments=2, nodes_per_seg=3, segment_ends=segends)

        states = {'x': {'units': 'm', 'shape': (1,), 'rate_units': 'm/s'},
                  'v': {'units': 'm/s', 'shape': (1,), 'rate_units': 'm/s**2'}}

        p = self._make_problem(gd, states)

        disc_idxs = gd.subset_node_indices['state_disc']
        col_idxs = gd.subset_node_indices['col']
        t0, tf = segends[0], segends[-1]
        t_all = (tf - t0) / 2.0 * (gd.node_ptau + 1.0) + t0

        for name, func, dfunc in [('x', x, f_x), ('v', v, f_v)]:
            p.set_val(f'state_disc:{name}', func(t_all[disc_idxs]).reshape(-1, 1))
            p.set_val(f'staterate_disc:{name}', dfunc(t_all[disc_idxs]).reshape(-1, 1))

        p.set_val('dt_dstau', _make_dt_dstau(gd, segends))
        p.run_model()

        for name, func in [('x', x), ('v', v)]:
            states_all = p.get_val(f'states_all:{name}')
            exact_col = func(t_all[col_idxs]).reshape(-1, 1)
            assert_almost_equal(states_all[col_idxs], exact_col, decimal=8,
                                 err_msg=f'Interpolated col state {name} does not match exact')

    def test_staterate_col_accuracy(self):
        """Interpolated rates at col nodes should be accurate for low-degree polynomials."""
        segends = np.array([0.0, 3.0, 10.0])
        gd = dm.GaussLobattoGrid(num_segments=2, nodes_per_seg=3, segment_ends=segends)

        states = {'x': {'units': 'm', 'shape': (1,), 'rate_units': 'm/s'}}
        p = self._make_problem(gd, states)

        disc_idxs = gd.subset_node_indices['state_disc']
        col_idxs = gd.subset_node_indices['col']
        t0, tf = segends[0], segends[-1]
        t_all = (tf - t0) / 2.0 * (gd.node_ptau + 1.0) + t0

        p.set_val('state_disc:x', x(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('staterate_disc:x', f_x(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('dt_dstau', _make_dt_dstau(gd, segends))
        p.run_model()

        staterate_col = p.get_val('staterate_col:x')
        exact_rate_col = f_x(t_all[col_idxs]).reshape(-1, 1)
        assert_almost_equal(staterate_col, exact_rate_col, decimal=8,
                             err_msg='Interpolated col rate does not match exact')

    def test_check_partials_scalar_state(self):
        """Verify analytic partials against finite differences for scalar state."""
        segends = np.array([0.0, 3.0, 10.0])
        gd = dm.GaussLobattoGrid(num_segments=2, nodes_per_seg=3, segment_ends=segends)

        states = {'x': {'units': 'm', 'shape': (1,), 'rate_units': 'm/s'},
                  'v': {'units': 'm/s', 'shape': (1,), 'rate_units': 'm/s**2'}}

        p = self._make_problem(gd, states)

        disc_idxs = gd.subset_node_indices['state_disc']
        t0, tf = segends[0], segends[-1]
        t_all = (tf - t0) / 2.0 * (gd.node_ptau + 1.0) + t0

        p.set_val('state_disc:x', x(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('staterate_disc:x', f_x(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('state_disc:v', v(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('staterate_disc:v', f_v(t_all[disc_idxs]).reshape(-1, 1))
        p.set_val('dt_dstau', _make_dt_dstau(gd, segends))

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=5.0e-8, rtol=1.0e-6)

    def test_check_partials_vector_state(self):
        """Verify analytic partials for a vector-valued state."""
        segends = np.array([0.0, 5.0])
        gd = dm.GaussLobattoGrid(num_segments=1, nodes_per_seg=5, segment_ends=segends)

        states = {'z': {'units': 'm', 'shape': (2,), 'rate_units': 'm/s'}}
        p = self._make_problem(gd, states)

        n_disc = gd.subset_num_nodes['state_disc']
        n_col = gd.subset_num_nodes['col']

        np.random.seed(42)
        p.set_val('state_disc:z', np.random.randn(n_disc, 2))
        p.set_val('staterate_disc:z', np.random.randn(n_disc, 2))
        p.set_val('dt_dstau', _make_dt_dstau(gd, segends))

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs', out_stream=None)
        assert_check_partials(cpd, atol=5.0e-8, rtol=1.0e-6)


if __name__ == '__main__':
    unittest.main()
