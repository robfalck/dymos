import unittest
import openmdao.api as om
import dymos as dm


from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from dymos.examples.cannonball.cannonball_ode import CannonballODE
from dymos.examples.cannonball.size_comp import CannonballSizeComp


@use_tempdirs
class TestCannonballDOE(unittest.TestCase):

    def make_problem(self, connected=False):

        #############################################
        # Setup the Dymos problem
        #############################################

        p = om.Problem(model=om.Group())

        p.driver = om.DOEDriver()


        p.model.add_subsystem('size_comp', CannonballSizeComp(),
                              promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10,
                               ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Birkhoff(grid=dm.BirkhoffGrid(num_segments=1, nodes_per_seg=10),
                                    solve_segments='forward')
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription, ode_init_kwargs={'use_tags': True})

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero
        # so that the phase ends at apogee).
        # The output of the ODE which provides the rate source for each state
        # is obtained from the tags used on those outputs in the ODE.
        # The units of the states are automatically inferred by multiplying the units
        # of those rates by the time units.
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                                duration_ref=100, units='s')
        ascent.set_state_options('r', fix_initial=True, fix_final=False)
        ascent.set_state_options('h', fix_initial=True, fix_final=False, lower=0, upper=100_000)
        ascent.set_state_options('gam', fix_initial=False, fix_final=False)
        ascent.set_state_options('v', fix_initial=False, fix_final=False, lower=0.1)

        ascent.add_parameter('S', units='m**2', static_target=True)
        ascent.add_parameter('m', units='kg', static_target=True)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # A duration balance is added to set the flight path angle to zero at the end of the phase (apogee).
        ascent.set_duration_balance('gam', val=0.0)

        # Second Phase (descent)
        transcription = dm.Birkhoff(grid=dm.BirkhoffGrid(num_segments=1, nodes_per_seg=10),
                                    solve_segments='forward')
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription, ode_init_kwargs={'use_tags': True})

        traj.add_phase('descent', descent)

        # All initial states and time are free, since
        #    they will be linked to the final states of ascent.
        # Final altitude is fixed, because we will set
        #    it to zero so that the phase ends at ground impact)
        descent.set_time_options(duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r')
        descent.add_state('h', fix_initial=False, fix_final=False, lower=0, upper=100_000)
        descent.add_state('gam', fix_initial=False, fix_final=False)
        descent.add_state('v', fix_initial=False, fix_final=False)

        descent.add_parameter('S', units='m**2', static_target=True)
        descent.add_parameter('m', units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        descent.set_duration_balance('h', val=0.0)


        ascent.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, maxiter=1000, stall_limit=3)
        ascent.linear_solver = om.DirectSolver()
        ascent.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

        descent.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, maxiter=1000, stall_limit=3)
        descent.linear_solver = om.DirectSolver()
        descent.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, static_target=True)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters
        # named 'm' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'm', 'descent': 'm'}, static_target=True)

        # In this case, by omitting targets, we're connecting these
        # parameters to parameters with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'], connected=True)  # Phases must be directly connected for solver.

        # Issue Connections
        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # A linear solver at the top level can improve performance.
        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        #############################################
        # Set constants and initial guesses
        #############################################
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.initial_states:r', 0.0)
        p.set_val('traj.ascent.initial_states:h', 0.0)
        p.set_val('traj.ascent.initial_states:gam', 25.0, units='deg')
        p.set_val('traj.ascent.initial_states:v', 200.0)
        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.ascent.final_states:r', 0.0)
        p.set_val('traj.ascent.final_states:h', 0.0)
        p.set_val('traj.ascent.final_states:gam', 0.0, units='deg')
        p.set_val('traj.ascent.final_states:v', 150.0)

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')
        p.set_val('traj.descent.final_states:h', 0.0)

        return p

    def test_two_phase_cannonball_connected_doe(self):
        p = self.make_problem(connected=True)
        p.run_model()

        p.list_problem_vars(print_arrays=True)

        p.model.list_outputs(print_arrays=True, explicit=False, implicit=True)

    # def test_two_phase_cannonball_birkhoff_connected(self):
    #     self.make_problem(connected=True)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
