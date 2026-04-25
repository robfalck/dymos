"""Integration tests for the GaussLobattoNew transcription."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.transcriptions.pseudospectral.gauss_lobatto_new import GaussLobattoNew
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestGaussLobattoNew(unittest.TestCase):

    def test_brachistochrone_basic(self):
        """
        Solve the brachistochrone OCP with GaussLobattoNew and verify the known optimal time.

        The analytical answer is t* ≈ 1.8016 s for the standard brachistochrone
        (x: 0→10 m, y: 10→5 m, v(0)=0, g=9.80665 m/s²).
        """
        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['tol'] = 1e-8
        p.driver.options['maxiter'] = 300

        tx = GaussLobattoNew(num_segments=10, order=3, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        traj = dm.Trajectory()
        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        phase.add_state('x', fix_initial=True, fix_final=False,
                        rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=False,
                        rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, fix_final=False,
                        rate_source='vdot', units='m/s')

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)

        phase.add_boundary_constraint('x', loc='final', equals=10.0, units='m')
        phase.add_boundary_constraint('y', loc='final', equals=5.0, units='m')

        phase.add_objective('time', loc='final', scaler=10.0)

        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=False)

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5.0, 100.5], units='deg')

        dm.run_problem(p, run_driver=True, simulate=False)

        t_duration = p.get_val('traj.phase0.t_duration')[0]
        assert_near_equal(t_duration, 1.8016, tolerance=1.0e-3)

    def test_brachistochrone_fix_final(self):
        """
        Solve brachistochrone with fix_final=True on x and y, verifying fix_initial and
        fix_final boundary constraints both work.
        """
        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['tol'] = 1e-8
        p.driver.options['maxiter'] = 300

        tx = GaussLobattoNew(num_segments=10, order=3, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        traj = dm.Trajectory()
        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        phase.add_state('x', fix_initial=True, fix_final=True,
                        rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=True,
                        rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, fix_final=False,
                        rate_source='vdot', units='m/s')

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)

        phase.add_objective('time', loc='final', scaler=10.0)

        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=False)

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5.0, 100.5], units='deg')

        dm.run_problem(p, run_driver=True, simulate=False)

        t_duration = p.get_val('traj.phase0.t_duration')[0]
        assert_near_equal(t_duration, 1.8016, tolerance=1.0e-3)

    def test_brachistochrone_timeseries(self):
        """Verify that timeseries outputs are accessible and have the right shape."""
        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['tol'] = 1e-8
        p.driver.options['maxiter'] = 300

        tx = GaussLobattoNew(num_segments=5, order=3, compressed=True)
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        traj = dm.Trajectory()
        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        phase.add_state('x', fix_initial=True, fix_final=False,
                        rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=False,
                        rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, fix_final=False,
                        rate_source='vdot', units='m/s')

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)

        phase.add_boundary_constraint('x', loc='final', equals=10.0, units='m')
        phase.add_boundary_constraint('y', loc='final', equals=5.0, units='m')

        phase.add_objective('time', loc='final', scaler=10.0)

        p.model.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=False)

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5.0, 100.5], units='deg')

        dm.run_problem(p, run_driver=True, simulate=False)

        # 5 segments * 3 nodes = 15 total nodes
        nn = tx.grid_data.num_nodes
        x_ts = p.get_val('traj.phase0.timeseries.x')
        y_ts = p.get_val('traj.phase0.timeseries.y')
        t_ts = p.get_val('traj.phase0.timeseries.time')

        self.assertEqual(x_ts.shape[0], nn)
        self.assertEqual(y_ts.shape[0], nn)
        self.assertEqual(t_ts.shape[0], nn)

        assert_near_equal(p.get_val('traj.phase0.t_duration')[0], 1.8016, tolerance=1.0e-3)


if __name__ == '__main__':
    unittest.main()
