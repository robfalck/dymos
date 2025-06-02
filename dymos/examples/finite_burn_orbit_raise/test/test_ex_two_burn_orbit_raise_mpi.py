import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem
from dymos.utils.misc import om_version


@require_pyoptsparse(optimizer='IPOPT')
@unittest.skipUnless(MPI, "MPI is required.")
# @use_tempdirs
class TestExampleTwoBurnOrbitRaiseMPI(unittest.TestCase):
    N_PROCS = 3

    def test_set_val_mpi_bug(self):
        p = om.Problem()
        par_group = om.ParallelGroup()

        c1 = om.ExecComp('y1 = x1 ** 2', x1={'shape': (1,)}, y1={'copy_shape': 'x1'})
        g1 = om.Group()
        g1.add_subsystem('c1', c1, promotes=['*'])

        c2 = om.ExecComp('g2 = x2', x2={'shape': (1,)}, g2={'copy_shape': 'x2'})
        g2 = om.Group()
        g2.add_subsystem('c2', c2, promotes=['*'])

        c3 = om.ExecComp('g3 = x3', x3={'shape': (1,)}, g3={'copy_shape': 'x3'})
        g3 = om.Group()
        g3.add_subsystem('c3', c3, promotes=['*'])

        par_group.add_subsystem('g1', g1, promotes=['*'])
        par_group.add_subsystem('g2', g2, promotes=['*'])
        par_group.add_subsystem('g3', g3, promotes=['*'])

        p.model.add_objective('y1')
        p.model.add_design_var('x1', lower=2, upper=5)
        p.model.add_design_var('x2', lower=2, upper=5)
        p.model.add_design_var('x3', lower=2, upper=5)
        p.model.add_constraint('g2', lower=3.)
        p.model.add_constraint('g3', lower=3.)

        p.model.add_subsystem('par_group', par_group, promotes=['*'])

        p.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        p.setup()

        g1.set_val('x1', 2.5)
        g2.set_val('x2', 2.5)
        g3.set_val('x3', 2.5)

        p.final_setup()
        p.list_driver_vars()

    def test_ex_two_burn_orbit_raise_mpi(self):
        optimizer = 'IPOPT'

        CONNECTED = False

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=True)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = p.model.traj.sim_prob.get_outputs_dir() / sim_db

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        # The last phase in this case is run in reverse time if CONNECTED=True,
        # so grab the correct index to test the resulting delta-V.
        end_idx = 0 if CONNECTED else -1

        assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)
        assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)

    def test_ex_two_burn_orbit_raise_connected_mpi(self):
        optimizer = 'IPOPT'

        CONNECTED = True

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=False)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = p.model.traj.sim_prob.get_outputs_dir() / sim_db

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        # The last phase in this case is run in reverse time if CONNECTED=True,
        # so grab the correct index to test the resulting delta-V.
        end_idx = 0 if CONNECTED else -1

        assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)
        assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
