import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.misc import om_version

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode import BrachistochroneVectorStatesODE


@use_tempdirs
class TestAddBoundaryConstraint(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_simple_no_exception(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        # test add_boundary_constraint with arrays:
        expected = np.array([10, 5])
        phase.add_boundary_constraint(name='pos', loc='final', equals=expected)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)
        p.final_setup()

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 1.8016

        pos0 = [0, 10]
        posf = [10, 5]

        p['traj0.phase0.states:pos'] = phase.interp('pos', [pos0, posf])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj0.phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('traj0.phase0.timeseries.pos')[-1, ...],
                          [10, 5], tolerance=1.0E-5)

    def test_invalid_expression(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # test add_boundary_constraint with arrays:
        expected = 'The expression provided `pos**2` has invalid format. ' \
                   'Expression may be a single variable or an equation ' \
                   'of the form `constraint_name = func(vars)`'
        with self.assertRaises(ValueError) as e:
            phase.add_boundary_constraint(name='pos**2', loc='final', equals=np.array([10, 5]))

        self.assertEqual(expected, str(e.exception))

    def test_duplicate_name(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        phase.add_boundary_constraint(name='pos', loc='final', equals=np.array([10, 5]))

        # test add_boundary_constraint with arrays:
        with self.assertRaises(ValueError) as e:
            phase.add_boundary_constraint(name='pos=v**2', loc='final', equals=np.array([10, 5]))

        expected = 'Cannot add new final boundary constraint named `pos` and indices None.' \
                   ' The name `pos` is already in use as a final boundary constraint'
        self.assertEqual(expected, str(e.exception))

    def test_duplicate_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        phase.add_boundary_constraint(name='pos', loc='final', equals=np.array([10, 5]))

        # test add_boundary_constraint with arrays:
        with self.assertRaises(ValueError) as e:
            phase.add_boundary_constraint(name='pos', loc='final', equals=np.array([10, 5]))

        expected = 'Cannot add new final boundary constraint for variable `pos` and indices None. One already exists.'
        self.assertEqual(expected, str(e.exception))


if __name__ == '__main__':
    unittest.main()
