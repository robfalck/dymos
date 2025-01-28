import unittest
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.general_utils import set_pyoptsparse_opt
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=True)


@use_tempdirs
class TestBrachistochroneVaryingOrderControlSimulation(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_results(self):

        transcriptions = {'GaussLobatto': dm.GaussLobatto(num_segments=20,
                                                          order=[3, 5]*10,
                                                          compressed=True),
                          'Radau': dm.Radau(num_segments=20,
                                            order=[3, 5] * 10,
                                            compressed=True)}

        for tx_name, tx in transcriptions.items():
            with self.subTest(f'{tx_name}'):
                p = om.Problem(model=om.Group())

                p.driver = om.pyOptSparseDriver()
                p.driver.options['optimizer'] = 'SLSQP'
                p.driver.declare_coloring()

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                p.model.add_subsystem('phase0', phase)

                phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

                phase.add_state('x', fix_initial=True, fix_final=False)
                phase.add_state('y', fix_initial=True, fix_final=False)
                phase.add_state('v', fix_initial=True, fix_final=False)

                phase.add_control('theta', continuity=True, rate_continuity=True,
                                units='deg', lower=0.01, upper=179.9)

                phase.add_parameter('g', units='m/s**2', val=9.80665)

                phase.add_boundary_constraint('x', loc='final', equals=10)
                phase.add_boundary_constraint('y', loc='final', equals=5)
                # Minimize time at the end of the phase
                phase.add_objective('time_phase', loc='final', scaler=10)

                p.model.linear_solver = om.DirectSolver()
                p.setup(check=True)

                phase.set_time_val(initial=0.0, duration=2.0)

                phase.set_state_val('x', [0, 10])
                phase.set_state_val('y', [10, 5])
                phase.set_state_val('v', [0, 9.9])

                phase.set_control_val('theta', [5, 100])
                phase.set_parameter_val('g', 9.80665)

                p.run_driver()

                exp_out = phase.simulate()

                t_initial = p.get_val('phase0.timeseries.time')[0]
                tf = p.get_val('phase0.timeseries.time')[-1]

                x0 = p.get_val('phase0.timeseries.x')[0]
                xf = p.get_val('phase0.timeseries.x')[-1]

                y0 = p.get_val('phase0.timeseries.y')[0]
                yf = p.get_val('phase0.timeseries.y')[-1]

                v0 = p.get_val('phase0.timeseries.v')[0]
                vf = p.get_val('phase0.timeseries.v')[-1]

                g = p.get_val('phase0.parameter_vals:g')[0]

                thetaf = exp_out.get_val('phase0.timeseries.theta')[-1]

                assert_almost_equal(t_initial, 0.0)
                assert_almost_equal(x0, 0.0)
                assert_almost_equal(y0, 10.0)
                assert_almost_equal(v0, 0.0)

                assert_almost_equal(tf, 1.8016, decimal=3)
                assert_almost_equal(xf, 10.0, decimal=3)
                assert_almost_equal(yf, 5.0, decimal=3)
                assert_almost_equal(vf, 9.902, decimal=3)
                assert_almost_equal(g, 9.80665, decimal=3)

                assert_almost_equal(thetaf, 100.12, decimal=0)
