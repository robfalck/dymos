import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except Exception:
    pyOptSparseDriver = None

import dymos as dm


class ODEComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # z is the state vector, a nn x 2 x 2 in the form of [[x, y], [vx, vy]]
        self.add_input('param', shape=(2,), units=None)
        self.add_input('z', shape=(nn, 2, 2), units=None)
        self.add_output('zdot', shape=(nn, 2, 2), units=None)

        self.declare_coloring(wrt='*', method='cs', num_full_jacs=5, tol=1.0E-12)

    def compute(self, inputs, outputs):
        outputs['zdot'][:, 0, 0] = inputs['z'][:, 1, 0] # xdot
        outputs['zdot'][:, 0, 1] = inputs['z'][:, 1, 1] # ydot
        outputs['zdot'][:, 1, 0] = inputs['param'][0] # vx_dot
        outputs['zdot'][:, 1, 1] = -inputs['param'][1] # vy_dot


def add_parameter_test(testShape=None):
    p = om.Problem()

    tx = dm.Birkhoff(num_segments=1, num_nodes=20)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=ODEComp, transcription=tx)

    if testShape is None:
        phase.add_parameter('param', static_target=True)
    else:
        phase.add_parameter('param', shape=testShape, static_target=True)

    traj.add_phase('phase', phase)

    p.model.add_subsystem('traj', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(2, 20), units=None)
    phase.add_state('z', rate_source='zdot', fix_initial=True, units=None, scaler=1.0, defect_ref=1000)

    phase.add_boundary_constraint('z', loc='final', lower=-100, upper=-90, indices=[1])
    phase.add_objective('time', loc='final')

    p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['max_iter'] = 500
    p.driver.declare_coloring(tol=1.0E-12)

    p.setup()

    phase.set_time_val(initial=0, duration=5.0)
    phase.set_parameter_val('param', [0.0, 9.80665])
    phase.set_state_val('z', [[[0, 0], [10, 10]], [[10, 0], [10, -10]]])

    res = dm.run_problem(p, simulate=True, make_plots=True)

    return res

# @use_tempdirs
class TestParameterTypes(unittest.TestCase):
    def test_tuple(self):
        res = add_parameter_test((2, ))
        self.assertEqual(res['exit_status'], 'SUCCESS')

    def test_list(self):
        res = add_parameter_test([2, ])
        self.assertEqual(res['exit_status'], 'SUCCESS')

    def test_scaler(self):
        res = add_parameter_test(2)
        self.assertEqual(res['exit_status'], 'SUCCESS')

    def test_nothing(self):
        res = add_parameter_test()
        self.assertEqual(res['exit_status'], 'SUCCESS')
